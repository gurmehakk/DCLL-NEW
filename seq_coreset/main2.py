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
import json


# Create output directory for plots, results, and model checkpoints
base_dir = 'mnist_lr/30/same'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/models', exist_ok=True)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        # For MNIST: 28x28 = 784 input features, 10 classes
        self.linear = nn.Linear(784, 10)
    
    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)
        # Apply linear layer and log softmax
        output = F.log_softmax(self.linear(x), dim=1)
        return output


import math
import time
import torch
import numpy as np
import torch.nn.functional as F

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

class CoresetLoader:
    """
    A combined class that builds and loads a coreset based on Algorithm 1
    for Local ε-Coreset Construction - Optimized for Speed
    """
    def __init__(self, dataset, model, device, epsilon=0.1, R=1.0, L=10.0, target_coreset_size=5000, 
                 anchor_loss=None, sample_size=5000):
        """
        Initialize the coreset loader
        
        Args:
            dataset: Original dataset
            model: Model used to compute losses
            device: Device to perform computations
            epsilon: Coreset approximation parameter (ε)
            R: Region radius
            L: Lipschitz constant
            target_coreset_size: Target size of the coreset
            anchor_loss: Pre-computed H value (F(β_anc)) if available
            sample_size: Number of samples to use for estimation (smaller = faster)
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.target_coreset_size = target_coreset_size
        self.anchor_loss = anchor_loss
        self.sample_size = min(sample_size, len(dataset))
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm: N = ⌈log n⌉
        self.N = math.ceil(math.log(self.n))
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_batch_losses(self, dataloader):
        """Compute losses for a batch of data points - much faster than one by one"""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                # Compute loss for each example in the batch
                for i in range(len(data)):
                    loss = F.nll_loss(output[i:i+1], target[i:i+1])
                    losses.append(loss.item())
        
        return losses
    
    def _compute_anchor_loss(self, dataloader):
        """Compute the loss at the anchor point (F(β_anc)) using batched processing"""
        self.model.eval()
        total_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.nll_loss(output, target, reduction='sum')
                total_loss += loss.item()
                sample_count += len(data)
        
        return total_loss / sample_count if sample_count > 0 else 0.0
    
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1 for Local ε-Coreset Construction
        Optimized for speed with batched processing
        """
        print(f"Building coreset for dataset size {self.n} with target size {self.target_coreset_size}")
        print(f"Parameters: ε={self.epsilon}, R={self.R}, L={self.L}, N={self.N}")
        start_time = time.time()
        
        # Step 1: Initialize as per Algorithm 1
        
        # Use pre-computed anchor loss if provided, otherwise compute it
         # Use the value from dry run
        
        
        # Step 2: Estimate layer distribution using sampling
        # Create a subset of the dataset for estimation
        indices = np.random.choice(self.n, self.sample_size, replace=False)
        
        # Get data loader for sampled indices
        from torch.utils.data import DataLoader, Subset
        sampled_dataset = Subset(self.dataset, indices)
        
        # Compute batch size based on GPU memory
        # Smaller models can use larger batch sizes
        batch_size = 128  # Adjust based on your model and GPU
        
        # Create dataloader for efficient batch processing
        sampled_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=False)
        
        H = self._compute_anchor_loss(sampled_loader) 
        print(f"H (F(β_anc)) = {H}")
        # Compute losses for the entire sample batch-wise
        sample_losses = []
        idx_map = []  # To map back to original dataset indices
        
        # Process in batches to exploit GPU parallelism
        for batch_idx, (data, target) in enumerate(sampled_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Compute loss for each example in batch
            self.model.eval()
            with torch.no_grad():
                output = self.model(data)
                for i in range(len(data)):
                    loss = F.nll_loss(output[i:i+1], target[i:i+1])
                    sample_losses.append(loss.item())
                    idx_map.append(indices[batch_idx * batch_size + i])
        
        # Define layer thresholds based on H with exponential scaling
        thresholds = [H * (2**j) for j in range(self.N + 1)]
        
        # Create layers and assign points directly - no need for nested loops
        layers = [[] for _ in range(self.N + 1)]
        for idx, loss in enumerate(sample_losses):
            # Binary search to find correct layer - much faster than linear search
            left, right = 0, self.N
            while left < right:
                mid = (left + right) // 2
                if loss <= thresholds[mid]:
                    right = mid
                else:
                    left = mid + 1
            layers[left].append(idx)
        
        # Get estimated layer sizes
        sample_layer_counts = [len(layer) for layer in layers]
        print(f"sample_layer_count: {sample_layer_counts}")
        # Extrapolate to full dataset - use float for more accurate calculations
        estimated_full_layer_sizes = [int(count * (self.n / len(sample_losses))) for count in sample_layer_counts]
        print(f"estimated_full_layer_sizes: {estimated_full_layer_sizes}")
        # Skip detailed logging for better performance
        
        # Step 3: For each non-empty layer, sample and assign weights
        all_selected_indices = []
        all_weights = []
        
        # Pre-calculate total points for efficiency
        total_points = sum(size for size in estimated_full_layer_sizes if size > 0)
        
        # Pre-calculate samples per layer - do all calculations at once
        non_empty_layers = [(j, estimated_full_layer_sizes[j]) for j in range(self.N + 1) if estimated_full_layer_sizes[j] > 0]
        total_layer_weight = sum(size for _, size in non_empty_layers)
        
        # Calculate sample allocation all at once
        sample_allocation = {}
        for j, size in non_empty_layers:
            # Proportional allocation
            allocation = max(1, int((size / total_layer_weight) * self.target_coreset_size))
            # Cap at layer size
            allocation = min(allocation, len(layers[j]))
            sample_allocation[j] = allocation
        
        # Adjust allocations to match target size
        target_sum = sum(sample_allocation.values())
        if target_sum < self.target_coreset_size:
            # Add samples to largest layers
            sorted_layers = sorted(non_empty_layers, key=lambda x: x[1], reverse=True)
            for j, _ in sorted_layers:
                if sum(sample_allocation.values()) >= self.target_coreset_size:
                    break
                if sample_allocation[j] < len(layers[j]):
                    sample_allocation[j] += 1
        
        # Sample from each layer and assign weights
        for j, size in non_empty_layers:
            if sample_allocation[j] > 0:
                # Get indices for this layer
                layer_indices = layers[j]
                
                # Sample using numpy - much faster than loop-based sampling
                selected = np.random.choice(layer_indices, sample_allocation[j], replace=False)
                
                # Calculate weight once
                weight = estimated_full_layer_sizes[j] / sample_allocation[j]
                
                # Use extend instead of append in loop
                all_selected_indices.extend([idx_map[idx] for idx in selected])
                all_weights.extend([weight] * len(selected))
        
        elapsed_time = time.time() - start_time
        print(f"Created coreset with {len(all_selected_indices)} samples in {elapsed_time:.2f} seconds")
        
        # Load all selected data into memory using batch loading
        # Sort indices to improve cache locality when accessing dataset
        sorted_indices = sorted(all_selected_indices)
        sorted_weights = [all_weights[all_selected_indices.index(idx)] for idx in sorted_indices]
        
        # Process in batches to reduce memory pressure
        all_data, all_targets = [], []
        batch_size = 500  # Adjust based on available memory
        for i in range(0, len(sorted_indices), batch_size):
            batch_indices = sorted_indices[i:i+batch_size]
            batch_data, batch_targets = [], []
            
            # Load a batch of data
            for idx in batch_indices:
                data, target = self.dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            # Add to our collection
            all_data.extend(batch_data)
            all_targets.extend(batch_targets)
        
        # Create tensors from the data
        if all_data:
            self.data = torch.stack(all_data).to(self.device)
            self.targets = torch.tensor(all_targets, dtype=torch.long).to(self.device)
            self.weights = torch.tensor(sorted_weights, dtype=torch.float).to(self.device)
        else:
            self.data = torch.tensor([]).to(self.device)
            self.targets = torch.tensor([], dtype=torch.long).to(self.device)
            self.weights = torch.tensor([], dtype=torch.float).to(self.device)
        
        # Store coreset size
        self.coreset_size = len(all_selected_indices)
        print(f"Final coreset size: {self.coreset_size}")
    
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

# Add new function to train without coreset
def train_without_coreset(epochs=3, momentum_value=0.0):
    """Train model on full dataset without coreset"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # Setup model and optimizer
    model = LogisticRegression().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum_value)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'epoch_times': []
    }
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / len(train_loader.dataset)
        
        # Testing
        test_loss, test_acc = test(model, device, test_loader)
        
        epoch_time = time.time() - epoch_start_time
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        
        scheduler.step()
    
    return history

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


def run_experiment(coreset_size, momentum_value, epochs=10):
    """
    Run a training experiment with a specific coreset size and momentum value
    
    Args:
        coreset_size: Maximum size of the coreset
        momentum_value: Value for SGD momentum parameter
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
    
    # Create model (Logistic Regression instead of CNN)
    model = LogisticRegression().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum_value)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'coreset_size': coreset_size,
        'momentum': momentum_value,
        'epoch_times': []
    }
    
    # Training loop
    print(f"\n=== Running experiment with coreset size: {coreset_size}, momentum: {momentum_value} ===")
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        
        # Build/rebuild coreset
        if epoch == 1 or epoch % 1 == 0:  # Rebuild every epoch
            coreset_loader = CoresetLoader(
                train_dataset, 
                model, 
                device,
                sample_size=coreset_size,
                epsilon=0.1,
                R=1.0,
                L=10.0,
                target_coreset_size=coreset_size
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
    
    # Save model state
    model_path = f"models/logistic_model_size_{coreset_size}_momentum_{momentum_value}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return history


def compare_momentum_values(sizes=[500, 1000, 2000, 5000], momentum_values=[0.0, 0.9], epochs=10):
    """Modified comparison function with separate metric saving"""
    all_results = {}
    
    # First run without coreset
    print("\n=== Running baseline without coreset ===")
    for momentum in momentum_values:
        baseline_result = train_without_coreset(epochs=3, momentum_value=momentum)
        # Save baseline results
        torch.save(baseline_result, f'{base_dir}/results/baseline_momentum_{momentum}.pt')
    
    # Run with different coreset sizes
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        all_results[momentum_key] = []
        
        for size in sizes:
            # epochs = 25 if size == 500 else 200
            result = run_experiment(size, momentum, epochs)
            all_results[momentum_key].append(result)
            
            # Save individual metrics separately
            metrics = {
                'train_acc': result['train_acc'],
                'test_acc': result['test_acc'],
                'train_loss': result['train_loss'],
                'test_loss': result['test_loss'],
                'epoch_times': result['epoch_times']
            }
            
            for metric_name, metric_values in metrics.items():
                torch.save(
                    metric_values,
                    f'{base_dir}/results/coreset_{size}_momentum_{momentum}_{metric_name}.pt'
                )
            
            # Save full result
            torch.save(result, f'{base_dir}/results/coreset_{size}_momentum_{momentum}_full.pt')
    
    # Save combined results
    torch.save(all_results, f'{base_dir}/results/combined_results.pt')
    
    # Create plots
    plot_all_in_one_line(all_results, momentum_values, sizes)

    
    # Print summary table
    print("\n=== Performance Summary ===")
    print(f"{'Momentum':<10} {'Coreset Size':<15} {'Final Train Loss':<15} {'Final Test Loss':<15} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 85)
    
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        results = all_results[momentum_key]
        
        for i, result in enumerate(results):
            print(f"{momentum:<10} {result['coreset_size']:<15} {result['train_loss'][-1]:<15.4f} {result['test_loss'][-1]:<15.4f} {result['train_acc'][-1]:<15.2f} {result['test_acc'][-1]:<15.2f}")


def plot_all_in_one_line(all_results, momentum_values, sizes):
    """
    Create plots with all metrics in one line (4 subplots in a row)
    
    Args:
        all_results: Dictionary containing results for all experiments
        momentum_values: List of momentum values used
        sizes: List of coreset sizes used
    """
    # Define plot styles for different momentum values
    styles = {
        0.0: {'marker': 'o', 'linestyle': '-', 'color': 'blue'},
        0.9: {'marker': 's', 'linestyle': '--', 'color': 'red'}
    }
    
    # 1. Create one figure with all metrics vs coreset size in one line
    plt.figure(figsize=(20, 5))
    
    # Training Loss vs Coreset Size
    plt.subplot(1, 4, 1)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        results = all_results[momentum_key]
        
        sizes_list = [r['coreset_size'] for r in results]
        final_train_loss = [r['train_loss'][-1] for r in results]
        
        style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
        plt.plot(sizes_list, final_train_loss, label=f'M={momentum}', **style)
    
    plt.title('Final Training Loss')
    plt.xlabel('Coreset Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Test Loss vs Coreset Size
    plt.subplot(1, 4, 2)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        results = all_results[momentum_key]
        
        sizes_list = [r['coreset_size'] for r in results]
        final_test_loss = [r['test_loss'][-1] for r in results]
        
        style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
        plt.plot(sizes_list, final_test_loss, label=f'M={momentum}', **style)
    
    plt.title('Final Test Loss')
    plt.xlabel('Coreset Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Training Accuracy vs Coreset Size
    plt.subplot(1, 4, 3)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        results = all_results[momentum_key]
        
        sizes_list = [r['coreset_size'] for r in results]
        final_train_acc = [r['train_acc'][-1] for r in results]
        
        style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
        plt.plot(sizes_list, final_train_acc, label=f'M={momentum}', **style)
    
    plt.title('Final Training Accuracy')
    plt.xlabel('Coreset Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    # Test Accuracy vs Coreset Size
    plt.subplot(1, 4, 4)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        results = all_results[momentum_key]
        
        sizes_list = [r['coreset_size'] for r in results]
        final_test_acc = [r['test_acc'][-1] for r in results]
        
        style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
        plt.plot(sizes_list, final_test_acc, label=f'M={momentum}', **style)
    
    plt.title('Final Test Accuracy')
    plt.xlabel('Coreset Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/plots/metrics_mnist_lr.png')
    
    # 2. For each size, create plots with epoch-wise training curves in one line
    for size in sizes:
        # Get the results for this size
        size_results = {}
        for momentum in momentum_values:
            momentum_key = f"momentum_{momentum}"
            for result in all_results[momentum_key]:
                if result['coreset_size'] == size:
                    size_results[momentum] = result
                    break
        
        if not size_results:
            continue
            
        # Create plot with 4 subplots in one line for this coreset size
        plt.figure(figsize=(20, 5))
        
        # Training Loss vs Epochs
        plt.subplot(1, 4, 1)
        for momentum, result in size_results.items():
            style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
            plt.plot(range(1, len(result['train_loss'])+1), result['train_loss'], 
                     label=f'M={momentum}', **style)
        
        plt.title(f'Training Loss (Size={size})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Test Loss vs Epochs
        plt.subplot(1, 4, 2)
        for momentum, result in size_results.items():
            style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
            plt.plot(range(1, len(result['test_loss'])+1), result['test_loss'], 
                     label=f'M={momentum}', **style)
        
        plt.title(f'Test Loss (Size={size})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Training Accuracy vs Epochs
        plt.subplot(1, 4, 3)
        for momentum, result in size_results.items():
            style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
            plt.plot(range(1, len(result['train_acc'])+1), result['train_acc'], 
                     label=f'M={momentum}', **style)
        
        plt.title(f'Training Accuracy (Size={size})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        # Test Accuracy vs Epochs
        plt.subplot(1, 4, 4)
        for momentum, result in size_results.items():
            style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
            plt.plot(range(1, len(result['test_acc'])+1), result['test_acc'], 
                     label=f'M={momentum}', **style)
        
        plt.title(f'Test Accuracy (Size={size})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{base_dir}/plots/size_{size}_metrics_mnist_lr.png')


def load_and_plot_results(result_file=None):
    """
    Load previously saved results and plot them
    
    Args:
        result_file: Path to the saved results file
    """
    if result_file is None:
        result_file = f'{base_dir}/results/combined_results.pt'
    all_results = torch.load(result_file)
    
    # Extract momentum values from keys
    momentum_values = [float(k.split('_')[1]) for k in all_results.keys()]
    
    # Get coreset sizes
    sizes = []
    for momentum_key in all_results:
        for result in all_results[momentum_key]:
            if result['coreset_size'] not in sizes:
                sizes.append(result['coreset_size'])
    sizes.sort()
    
    # Create plots using the loaded data
    plot_all_in_one_line(all_results, momentum_values, sizes)
    
    print("Loaded results and created plots from saved data.")

#[1000, 2000, 4000, 800, 16000]

def main():
    # Run comparison experiment with various coreset sizes and momentum values
    compare_momentum_values(
        sizes=[1000, 2000, 4000, 8000, 16000], 
        momentum_values=[0.0, 0.9], 
        epochs=30
    )
    
    # Alternatively, if results are already saved, just load and plot them
    # load_and_plot_results()

if __name__ == '__main__':
    main()