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
base_dir = '/home/gurmehak/DCLL/seq_coreset/mnist_cnn/diff/lr250'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/models', exist_ok=True)


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(28 * 28, 10)  

    def forward(self, x):
        x = x.view(x.size(0), -1)  
        return self.linear(x)



import math
import time
import torch
import numpy as np
import torch.nn.functional as F

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
        # base_size = int(self.n * self.epsilon)
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

# #[1000, 2000, 4000, 800, 16000]

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


import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import time
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt



base_dir = '/home/gurmehak/DCLL/seq_coreset/mnist_cnn/diff/parallel'
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)

def train(model, coreset_loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, target, weights in coreset_loader:
        optimizer.zero_grad()
        output = model(data)
        individual_losses = F.nll_loss(output, target, reduction='none')
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
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def run_experiment(coreset_size, momentum_value, epochs=10):
    use_cuda = False
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda:1" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset)

    model = LogisticRegression().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=momentum_value)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

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
                                       sample_size=coreset_size, epsilon=0.1,
                                       R=1.0, L=10.0, max_coreset_size=coreset_size)

        train_loss, train_acc = train(model, coreset_loader, optimizer)
        test_loss, test_acc = test(model, device, test_loader)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(time.time() - epoch_start)

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

    with ThreadPoolExecutor(max_workers=min(len(sizes) * len(momentum_values), 8)) as executor:
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
        for res in results:
            print(f"{momentum:<10} {res['coreset_size']:<15} {res['train_loss'][-1]:<15.4f} "
                  f"{res['test_loss'][-1]:<15.4f} {res['train_acc'][-1]:<15.2f} {res['test_acc'][-1]:<15.2f}")

def main():
    # Run comparison experiment with various coreset sizes and momentum values
    compare_momentum_values_parallel(
    sizes=[250,500, 1000, 2000, 4000, 8000, 16000],
    momentum_values=[0.0, 0.9],
    epochs=100
)

    
    # Alternatively, if results are already saved, just load and plot them
    # load_and_plot_results()

if __name__ == '__main__':
    main()