import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor

# Create output directory for plots, results, and model checkpoints
base_dir = './'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/models', exist_ok=True)


class ResNet18Model(nn.Module):
    """ResNet18 model for CIFAR-10"""
    def __init__(self, num_classes=10):
        super(ResNet18Model, self).__init__()
        # Start with a fresh model (no pretrained weights)
        model = models.resnet18(pretrained=False)
        
        # CRITICAL FIX: Replace the first convolutional layer with a smaller kernel and stride
        # Original ResNet-18 uses 7x7 kernels with stride 2, which is too aggressive for 32x32 CIFAR images
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # CRITICAL FIX: Remove the max pooling layer which would further reduce the spatial dimensions
        model.maxpool = nn.Identity()
        
        # Final classification layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        self.model = model
        
    def forward(self, x):
        return self.model(x)


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
        self.N = math.ceil(math.log(self.n))  # we take log as per algo
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norm(self, data, target):
        """Compute gradient norm for a single data point"""
        self.model.zero_grad()
        output = self.model(data.unsqueeze(0))
        loss = F.cross_entropy(output, target.unsqueeze(0))
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
                loss = F.cross_entropy(output, target.unsqueeze(0))
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
        
        # Calculate total number of samples to select
        actual_coreset_size = min(self.n, self.max_coreset_size)
        
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
        print(f"weights: {self.weights.shape}")
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
        0.0: {'marker': 'o', 'linestyle': '-', 'color': 'green'},
        0.9: {'marker': 's', 'linestyle': '--', 'color': 'darkorange'}
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
    plt.savefig(f'{base_dir}/plots/metrics_cifar10_resnet18.png')
    
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
        plt.savefig(f'{base_dir}/plots/size_{size}_metrics_cifar10_resnet18.png')


def train(model, coreset_loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, target, weights in coreset_loader:
        optimizer.zero_grad()
        output = model(data)
        individual_losses = F.cross_entropy(output, target, reduction='none')
        loss = (individual_losses * weights).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(target)

    accuracy = 100. * correct / total if total > 0 else 0
    return total_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    torch.cuda.empty_cache()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def run_experiment(coreset_size, momentum_value, epochs=10):
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
    torch.manual_seed(1)
    np.random.seed(1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Data transformations for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # Initialize ResNet18 model
    model = ResNet18Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum_value, weight_decay=5e-4)
    # scheduler = StepLR(optimizer, step_size=20, gamma=0.5)  # Adjusted for CIFAR-10
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'coreset_size': coreset_size,
        'momentum': momentum_value,
        'epoch_times': []
    }

    print(f"\n=== Running experiment with coreset size: {coreset_size}, momentum: {momentum_value} ===")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        coreset_loader = CoresetLoader(train_dataset, model, device,
                                     sample_size=min(coreset_size*2, 5000),  # Sample size for gradient estimation
                                     epsilon=0.1,
                                     R=1.0, L=10.0, max_coreset_size=coreset_size)

        train_loss, train_acc = train(model, coreset_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(time.time() - epoch_start)
        
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"Time: {history['epoch_times'][-1]:.2f}s")

    # Save the final model
    torch.save(model.state_dict(), f'{base_dir}/models/resnet18_coreset_{coreset_size}_momentum_{momentum_value}.pt')
    return history


def save_results(result, coreset_size, momentum_value):
    metrics = {
        'train_acc': result['train_acc'],
        'test_acc': result['test_acc'],
        'train_loss': result['train_loss'],
        'test_loss': result['test_loss'],
        'epoch_times': result['epoch_times']
    }
    for metric_name, values in metrics.items():
        torch.save(values, f'{base_dir}/results/coreset_{coreset_size}_momentum_{momentum_value}_{metric_name}.pt')
    torch.save(result, f'{base_dir}/results/coreset_{coreset_size}_momentum_{momentum_value}_full.pt')


def compare_momentum_values_parallel(sizes=[500, 1000, 2000, 5000], momentum_values=[0.0, 0.9], epochs=10):
    all_results = {}

    def run_and_save(size, momentum):
        result = run_experiment(size, momentum, epochs)
        save_results(result, size, momentum)
        return momentum, size, result

    # Determine max workers based on available CPUs and GPUs
    max_workers = min(len(sizes) * len(momentum_values), 1)
        
    print(f"Running with {max_workers} worker threads")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_and_save, size, momentum)
                   for momentum in momentum_values for size in sizes]

        for future in futures:
            momentum, size, result = future.result()
            key = f"momentum_{momentum}"
            if key not in all_results:
                all_results[key] = []
            all_results[key].append(result)

    torch.save(all_results, f'{base_dir}/results/combined_results.pt')
    # Create plots
    plot_all_in_one_line(all_results, momentum_values, sizes)
    print_summary(all_results, momentum_values)


def print_summary(all_results, momentum_values):
    print("\n=== Performance Summary ===")
    print(f"{'Momentum':<10} {'Coreset Size':<15} {'Final Train Loss':<15} {'Final Test Loss':<15} {'Train Acc':<15} {'Test Acc':<15}")
    print("-" * 85)

    for momentum in momentum_values:
        results = all_results[f"momentum_{momentum}"]
        for res in sorted(results, key=lambda x: x['coreset_size']):
            print(f"{momentum:<10} {res['coreset_size']:<15} {res['train_loss'][-1]:<15.4f} "
                  f"{res['test_loss'][-1]:<15.4f} {res['train_acc'][-1]:<15.2f} {res['test_acc'][-1]:<15.2f}")


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
    print_summary(all_results, momentum_values)
    
    print("Loaded results and created plots from saved data.")


def main():
    # Run comparison experiment with various coreset sizes and momentum values
    # Reduced sizes and epochs due to CIFAR-10 + ResNet18 being more computationally intensive
    compare_momentum_values_parallel(
        sizes=[500], 
        momentum_values=[0.0, 0.9],
        epochs=500  # Reduced from 100 to 50 for faster execution
    )
    
    # Alternatively, if results are already saved, just load and plot them
    # load_and_plot_results()


if __name__ == '__main__':
    main()