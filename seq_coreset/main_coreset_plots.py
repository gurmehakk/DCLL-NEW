import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class CoresetLoader:
    """
    A combined class that builds and loads a coreset
    """
    def __init__(self, dataset, model, device, sample_size=1000, epsilon=0.1, R=1.0, L=10.0, max_coreset_size=2000):
        """
        Initialize the coreset loader
        
        Args:
            dataset: Original dataset
            model: Model used to compute gradients
            device: Device to perform computations
            sample_size: Number of samples to use for estimating layer partitioning
            epsilon: Coreset approximation parameter
            R: Region radius
            L: Lipschitz constant
            max_coreset_size: Maximum size of the coreset
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.sample_size = sample_size
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.max_coreset_size = max_coreset_size
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm
        self.N = math.ceil(math.log(self.n)) #we take log as per algo
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norm(self, data, target):
        """Compute gradient norm for a single data point"""
        self.model.zero_grad()
        output = self.model(data.unsqueeze(0))
        loss = F.nll_loss(output, target.unsqueeze(0))
        loss.backward()
        
        # Collect and flatten gradients
        grad_norm = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += torch.sum(param.grad ** 2)
        
        return torch.sqrt(grad_norm).item()
    
    def _compute_model_loss(self):
        """Compute the overall loss at the anchor point (F(β_anc))"""
        self.model.eval()
        sample_size = min(100, self.n)  # Use a small sample to estimate loss
        indices = np.random.choice(self.n, sample_size, replace=False)
        
        total_loss = 0.0
        with torch.no_grad():
            for idx in indices:
                data, target = self.dataset[idx]
                data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
                output = self.model(data.unsqueeze(0))
                loss = F.nll_loss(output, target.unsqueeze(0))
                total_loss += loss.item()
        
        return total_loss / sample_size
    
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1
        """
        print("Building coreset...")
        start_time = time.time()
        
        # Step 1: Initialize as per Algorithm 1
        H = self._compute_model_loss()  # This is F(β_anc) in Algorithm 1
        
        # Sample data points to estimate gradient distribution
        indices = np.random.choice(self.n, min(self.sample_size, self.n), replace=False)
        
        # Compute gradient norms for sampled points
        grad_norms = []
        self.model.eval()
        for idx in indices:
            data, target = self.dataset[idx]
            data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
            norm = self._compute_grad_norm(data, target)
            grad_norms.append((idx, norm))
        
        # Step 2: Partition into layers based on gradient norms
        layers = [[] for _ in range(self.N + 1)]
        
        # Sort by gradient norm
        grad_norms.sort(key=lambda x: x[1])
        
        # Determine layer thresholds - equally spaced in log scale
        if len(grad_norms) > 0:
            min_norm = max(grad_norms[0][1], 1e-6)
            max_norm = max(grad_norms[-1][1], min_norm * 2)
            # Create N+1 layers with exponentially increasing thresholds
            thresholds = [min_norm * (max_norm/min_norm)**(j/self.N) for j in range(self.N+1)]
        else:
            thresholds = [0] * (self.N+1)
        
        # Assign samples to layers according to Equation (6) and (7) in the algorithm
        for idx, norm in grad_norms:
            for j in range(self.N+1):
                if j == self.N or norm <= thresholds[j]:
                    layers[j].append(idx)
                    break
        
        # Step 3: For each layer, sample and assign weights
        all_selected_indices = []
        all_weights = []
        
        # Calculate total number of samples to select based on epsilon
        base_size = int(self.n * self.epsilon)
        actual_coreset_size = min(base_size, self.max_coreset_size)
        
        # Distribute samples across layers proportionally
        total_layer_samples = sum(len(layer) for layer in layers if len(layer) > 0)
        remaining_samples = actual_coreset_size
        
        for j in range(self.N+1):
            if len(layers[j]) > 0:
                # Calculate sample size for this layer
                layer_proportion = len(layers[j]) / total_layer_samples
                size = max(int(layer_proportion * actual_coreset_size), 1)
                size = min(size, len(layers[j]), remaining_samples)
                if size == 0:
                    continue
                
                # Sample from this layer
                selected = np.random.choice(layers[j], size, replace=False)
                
                # Assign weights according to step 3(b) in Algorithm 1
                weight = len(layers[j]) / size  # This is |Pj|/|Qj|
                
                all_selected_indices.extend(selected)
                all_weights.extend([weight] * size)
                
                remaining_samples -= size
        
        elapsed_time = time.time() - start_time
        print(f"Created coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        print(f"Coreset construction time: {elapsed_time:.2f} seconds")
        
        # Load all selected data into memory
        all_data = []
        all_targets = []
        
        for idx in all_selected_indices:
            data, target = self.dataset[idx]
            all_data.append(data)
            all_targets.append(target)
        
        # Create and store coreset data tensors
        if all_data:
            self.data = torch.stack(all_data).to(self.device)
        else:
            self.data = torch.tensor([]).to(self.device) 
            
        if all_targets:
            self.targets = torch.tensor(all_targets, dtype=torch.long).to(self.device)
        else:
            self.targets = torch.tensor([], dtype=torch.long).to(self.device)
            
        self.weights = torch.tensor(all_weights, dtype=torch.float).to(self.device)
        
        # Store coreset size for reference
        self.coreset_size = len(all_selected_indices)
    
    def __iter__(self):
        """
        Iterator that yields the entire coreset once
        """
        yield self.data, self.targets, self.weights
    
    def __len__(self):
        """
        Always returns 1 since there's only one batch (the entire coreset)
        """
        return 1


def train(model, coreset_loader, optimizer):
    """
    Train the model using the coreset loader
    
    Args:
        model: The neural network model
        coreset_loader: CoresetLoader containing the coreset data
        optimizer: The optimizer to use
        
    Returns:
        Tuple of (average loss, training accuracy)
    """
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    # Since our CoresetLoader only has one batch (the entire coreset),
    # this loop will execute only once
    for data, target, weights in coreset_loader:
        optimizer.zero_grad()
        output = model(data)
        
        # Compute weighted loss
        individual_losses = F.nll_loss(output, target, reduction='none')
        loss = (individual_losses * weights).mean()
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(target)
    
    accuracy = 100. * correct / total if total > 0 else 0
    return total_loss, accuracy


def test(model, device, test_loader):
    """
    Test the model on the test dataset
    
    Args:
        model: The neural network model
        device: Device to perform testing
        test_loader: DataLoader for test data
        
    Returns:
        Tuple of (average loss, test accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.1f}%)')
    
    return test_loss, accuracy


def run_experiment(coreset_size, epochs=10):
    """
    Run a training experiment with a specific coreset size
    
    Args:
        coreset_size: Maximum size of the coreset
        epochs: Number of training epochs
        
    Returns:
        Dict containing training history
    """
    # Setup
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset)
    
    # Create model
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=1.0,momentum=0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'coreset_size': coreset_size,
        'epoch_times': []
    }
    
    # Training loop
    print(f"\n=== Running experiment with coreset size: {coreset_size} ===")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Build/rebuild coreset
        if epoch == 1 or epoch % 1 == 0:  # Rebuild every epoch
            coreset_loader = CoresetLoader(
                train_dataset, 
                model, 
                device,
                sample_size=1000,
                epsilon=0.1,
                R=1.0,
                L=10.0,
                max_coreset_size=coreset_size
            )
        
        # Train and test
        train_loss, train_acc = train(model, coreset_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.1f}%, Time: {epoch_time:.2f}s")
    
    return history


def compare_coreset_sizes(sizes=[500, 1000, 2000, 5000], epochs=10):
    """
    Compare model performance with different coreset sizes
    
    Args:
        sizes: List of coreset sizes to compare
        epochs: Number of training epochs
    """
    results = []
    
    for size in sizes:
        result = run_experiment(size, epochs)
        results.append(result)
    
    # Plot train/test accuracies
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    for result in results:
        plt.plot(range(1, epochs+1), result['train_acc'], 
                 label=f"Coreset size: {result['coreset_size']}")
    plt.title('Training Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    for result in results:
        plt.plot(range(1, epochs+1), result['test_acc'], 
                 label=f"Coreset size: {result['coreset_size']}")
    plt.title('Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot train/test losses
    plt.subplot(2, 2, 3)
    for result in results:
        plt.plot(range(1, epochs+1), result['train_loss'], 
                 label=f"Coreset size: {result['coreset_size']}")
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    for result in results:
        plt.plot(range(1, epochs+1), result['test_loss'], 
                 label=f"Coreset size: {result['coreset_size']}")
    plt.title('Test Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/coreset_size_comparison_0.png')
    plt.show()
    
    # Plot final performance vs coreset size
    plt.figure(figsize=(12, 10))
    
    # Get the data
    sizes_list = [r['coreset_size'] for r in results]
    final_train_acc = [r['train_acc'][-1] for r in results]
    final_test_acc = [r['test_acc'][-1] for r in results]
    final_train_loss = [r['train_loss'][-1] for r in results]
    final_test_loss = [r['test_loss'][-1] for r in results]
    avg_epoch_times = [sum(r['epoch_times'])/len(r['epoch_times']) for r in results]
    
    # Plot Accuracy vs Coreset Size
    plt.subplot(2, 2, 1)
    plt.plot(sizes_list, final_train_acc, 'o-', label='Training Accuracy')
    plt.plot(sizes_list, final_test_acc, 's-', label='Test Accuracy')
    plt.title('Final Accuracy vs Coreset Size')
    plt.xlabel('Coreset Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss vs Coreset Size
    plt.subplot(2, 2, 2)
    plt.plot(sizes_list, final_train_loss, 'o-', label='Training Loss')
    plt.plot(sizes_list, final_test_loss, 's-', label='Test Loss')
    plt.title('Final Loss vs Coreset Size')
    plt.xlabel('Coreset Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Average Epoch Time vs Coreset Size
    plt.subplot(2, 2, 3)
    plt.plot(sizes_list, avg_epoch_times, 'o-')
    plt.title('Average Epoch Time vs Coreset Size')
    plt.xlabel('Coreset Size')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    
    # Plot Train-Test Accuracy Gap vs Coreset Size
    plt.subplot(2, 2, 4)
    accuracy_gap = [t - v for t, v in zip(final_train_acc, final_test_acc)]
    plt.plot(sizes_list, accuracy_gap, 'o-')
    plt.title('Train-Test Accuracy Gap vs Coreset Size')
    plt.xlabel('Coreset Size')
    plt.ylabel('Gap (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/coreset_size_performance_0.png')
    plt.show()
    
    # Print summary table
    print("\n=== Performance Summary ===")
    print(f"{'Coreset Size':<15} {'Final Train Acc':<15} {'Final Test Acc':<15} {'Train Loss':<15} {'Test Loss':<15} {'Avg Epoch Time':<15}")
    print("-" * 90)
    for i, size in enumerate(sizes_list):
        print(f"{size:<15} {final_train_acc[i]:<15.2f} {final_test_acc[i]:<15.2f} {final_train_loss[i]:<15.4f} {final_test_loss[i]:<15.4f} {avg_epoch_times[i]:<15.2f}")


def main():
    # Run comparison experiment with various coreset sizes
    # Using smaller epochs for demonstration
    compare_coreset_sizes(sizes=[500, 1000, 2000, 3000, 5000, 7500, 10000], epochs=500)

if __name__ == '__main__':
    main()