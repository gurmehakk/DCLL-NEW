import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


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
        self.N = math.ceil(math.log(self.n))
        
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
                
                # Sample from this layer
                selected = np.random.choice(layers[j], size, replace=False)
                
                # Assign weights according to step 3(b) in Algorithm 1
                weight = len(layers[j]) / size  # This is |Pj|/|Qj|
                
                all_selected_indices.extend(selected)
                all_weights.extend([weight] * size)
                
                remaining_samples -= size
        
        print(f"Created coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        
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
        The average loss for this epoch
    """
    model.train()
    
    total_loss = 0
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
    
    return total_loss


def test(model, device, test_loader):
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

    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def main(
    test_batch_size=1000,
    epochs=14,
    lr=1.0,
    gamma=0.7,
    use_cuda=True,
    seed=1,
    save_model=False,
    coreset_epsilon=0.1,
    coreset_radius=1.0,
    lipschitz_constant=10.0,
    coreset_size=2000,
    rebuild_coreset=1
):
    """
    Main training function with normal arguments instead of argparse
    
    Args:
        test_batch_size: Input batch size for testing
        epochs: Number of epochs to train
        lr: Learning rate
        gamma: Learning rate step gamma
        use_cuda: Whether to use CUDA if available
        seed: Random seed
        save_model: Whether to save the model after training
        coreset_epsilon: Epsilon parameter for coreset
        coreset_radius: Radius parameter for coreset
        lipschitz_constant: Lipschitz constant L
        coreset_size: Maximum size of coreset
        rebuild_coreset: Number of epochs after which to rebuild the coreset
    """
    # Setup
    use_cuda = use_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)
    
    # Create model
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Rebuild coreset periodically
        if epoch == 1 or epoch % rebuild_coreset == 0:
            # Create coreset loader which builds the coreset at initialization
            coreset_loader = CoresetLoader(
                train_dataset, 
                model, 
                device,
                sample_size=1000,
                epsilon=coreset_epsilon,
                R=coreset_radius,
                L=lipschitz_constant,
                max_coreset_size=coreset_size
            )
            
        # Train model with coreset loader
        loss = train(model, coreset_loader, optimizer)
        print(f'Epoch: {epoch}, Loss: {loss:.6f}')
        
        # Test
        test(model, device, test_loader)
        
        # Update learning rate
        scheduler.step()
    
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn_local_coreset.pt")


if __name__ == '__main__':
    # Call main with default parameters
    # You can customize any parameter here
    main(
        test_batch_size=1000,
        epochs=14,
        lr=1.0,
        gamma=0.7,
        use_cuda=True,
        seed=1,
        save_model=False,
        coreset_epsilon=0.1,
        coreset_radius=1.0,
        lipschitz_constant=10.0,
        coreset_size=3000,
        rebuild_coreset=1
    )