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
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback


# Create output directory for plots, results, and model checkpoints
base_dir = '/home/gurmehak/DCLL/seq_coreset/mnist_cnn/diff/fixed'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/models', exist_ok=True)


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
    A combined class that builds and loads a coreset with memory optimization
    """
    def __init__(self, dataset, model, device, sample_size=1000, epsilon=0.1, R=1.0, L=10.0, max_coreset_size=2000):
        """
        Initialize the coreset loader
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.sample_size = min(sample_size, len(dataset))  # Ensure we don't exceed dataset size
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.max_coreset_size = max_coreset_size
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm
        self.N = math.ceil(math.log(max(self.n, 2)))  # Avoid log(0)
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norm(self, data, target):
        """Compute gradient norm for a single data point with memory optimization"""
        try:
            self.model.zero_grad()
            data_batch = data.unsqueeze(0)
            target_batch = target.unsqueeze(0)
            
            output = self.model(data_batch)
            loss = F.nll_loss(output, target_batch)
            loss.backward()
            
            # Collect and flatten gradients
            grad_norm = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += torch.sum(param.grad ** 2)
            
            # Clear gradients immediately to save memory
            self.model.zero_grad()
            
            return torch.sqrt(grad_norm).item()
        except Exception as e:
            print(f"Error computing gradient norm: {e}")
            return 0.0
    
    def _compute_model_loss(self):
        """Compute the overall loss at the anchor point (F(β_anc)) with memory optimization"""
        self.model.eval()
        sample_size = min(50, self.n)  # Reduced sample size to save memory
        indices = np.random.choice(self.n, sample_size, replace=False)
        
        total_loss = 0.0
        with torch.no_grad():
            for idx in indices:
                try:
                    data, target = self.dataset[idx]
                    data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
                    output = self.model(data.unsqueeze(0))
                    loss = F.nll_loss(output, target.unsqueeze(0))
                    total_loss += loss.item()
                except Exception as e:
                    print(f"Error in loss computation: {e}")
                    continue
        
        return total_loss / sample_size if sample_size > 0 else 0.0
    
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1 with memory optimization
        """
        print(f"Building coreset with max size {self.max_coreset_size}...")
        start_time = time.time()
        
        try:
            # Step 1: Initialize as per Algorithm 1
            H = self._compute_model_loss()
            
            # Use smaller sample size for gradient computation to avoid memory issues
            actual_sample_size = min(self.sample_size, 500)  # Reduced from original
            indices = np.random.choice(self.n, actual_sample_size, replace=False)
            
            # Compute gradient norms for sampled points with progress tracking
            grad_norms = []
            self.model.eval()
            
            print(f"Computing gradients for {len(indices)} samples...")
            for i, idx in enumerate(indices):
                if i % 100 == 0:
                    print(f"Processed {i}/{len(indices)} samples")
                    # Clear cache periodically
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                try:
                    data, target = self.dataset[idx]
                    data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
                    norm = self._compute_grad_norm(data, target)
                    grad_norms.append((idx, norm))
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue
            
            if not grad_norms:
                print("No valid gradient norms computed, using random sampling")
                selected_indices = np.random.choice(self.n, min(self.max_coreset_size, self.n), replace=False)
                weights = [1.0] * len(selected_indices)
            else:
                # Step 2: Partition into layers based on gradient norms
                layers = [[] for _ in range(self.N + 1)]
                
                # Sort by gradient norm
                grad_norms.sort(key=lambda x: x[1])
                
                # Determine layer thresholds
                min_norm = max(grad_norms[0][1], 1e-8)
                max_norm = max(grad_norms[-1][1], min_norm * 2)
                thresholds = [min_norm * (max_norm/min_norm)**(j/self.N) for j in range(self.N+1)]
                
                # Assign samples to layers
                for idx, norm in grad_norms:
                    for j in range(self.N+1):
                        if j == self.N or norm <= thresholds[j]:
                            layers[j].append(idx)
                            break
                
                # Step 3: Sample from each layer and assign weights
                selected_indices = []
                weights = []
                
                actual_coreset_size = min(self.n, self.max_coreset_size)
                total_layer_samples = sum(len(layer) for layer in layers if len(layer) > 0)
                remaining_samples = actual_coreset_size
                
                for j in range(self.N+1):
                    if len(layers[j]) > 0 and remaining_samples > 0:
                        layer_proportion = len(layers[j]) / total_layer_samples
                        size = max(int(layer_proportion * actual_coreset_size), 1)
                        size = min(size, len(layers[j]), remaining_samples)
                        
                        if size > 0:
                            selected = np.random.choice(layers[j], size, replace=False)
                            weight = len(layers[j]) / size
                            
                            selected_indices.extend(selected)
                            weights.extend([weight] * size)
                            remaining_samples -= size
            
            elapsed_time = time.time() - start_time
            print(f"Created coreset with {len(selected_indices)} samples from {self.n} original samples")
            print(f"Coreset construction time: {elapsed_time:.2f} seconds")
            
            # Load selected data with memory optimization
            all_data = []
            all_targets = []
            
            print("Loading coreset data...")
            for i, idx in enumerate(selected_indices):
                if i % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    data, target = self.dataset[idx]
                    all_data.append(data)
                    all_targets.append(target)
                except Exception as e:
                    print(f"Error loading sample {idx}: {e}")
                    continue
            
            # Create tensors
            if all_data:
                self.data = torch.stack(all_data)
                # Don't move to device immediately to save GPU memory
                if not torch.cuda.is_available():
                    self.data = self.data.to(self.device)
            else:
                self.data = torch.tensor([])
                
            if all_targets:
                self.targets = torch.tensor(all_targets, dtype=torch.long)
                if not torch.cuda.is_available():
                    self.targets = self.targets.to(self.device)
            else:
                self.targets = torch.tensor([], dtype=torch.long)
                
            self.weights = torch.tensor(weights, dtype=torch.float)
            if not torch.cuda.is_available():
                self.weights = self.weights.to(self.device)
            
            self.coreset_size = len(selected_indices)
            print(f"Coreset ready with {self.coreset_size} samples")
            
        except Exception as e:
            print(f"Error in coreset construction: {e}")
            traceback.print_exc()
            # Fallback to random sampling
            selected_indices = np.random.choice(self.n, min(100, self.n), replace=False)
            all_data = [self.dataset[idx][0] for idx in selected_indices]
            all_targets = [self.dataset[idx][1] for idx in selected_indices]
            
            self.data = torch.stack(all_data)
            self.targets = torch.tensor(all_targets, dtype=torch.long)
            self.weights = torch.ones(len(selected_indices), dtype=torch.float)
            self.coreset_size = len(selected_indices)
    
    def __iter__(self):
        """Iterator that yields the entire coreset once"""
        # Move data to device only when needed
        data = self.data.to(self.device) if not self.data.is_cuda else self.data
        targets = self.targets.to(self.device) if not self.targets.is_cuda else self.targets
        weights = self.weights.to(self.device) if not self.weights.is_cuda else self.weights
        yield data, targets, weights
    
    def __len__(self):
        return 1


def train(model, coreset_loader, optimizer):
    """Train the model using coreset with memory optimization"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    try:
        for data, target, weights in coreset_loader:
            optimizer.zero_grad()
            
            # Process in smaller batches if data is too large
            batch_size = min(64, len(data))
            if len(data) > batch_size:
                # Process in mini-batches
                indices = torch.randperm(len(data))
                for i in range(0, len(data), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_data = data[batch_indices]
                    batch_target = target[batch_indices]
                    batch_weights = weights[batch_indices]
                    
                    output = model(batch_data)
                    individual_losses = F.nll_loss(output, batch_target, reduction='none')
                    loss = (individual_losses * batch_weights).mean()
                    loss = loss / ((len(data) + batch_size - 1) // batch_size)  # Normalize by number of batches
                    
                    loss.backward()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(batch_target.view_as(pred)).sum().item()
                    total += len(batch_target)
                    total_loss += loss.item()
                
                optimizer.step()
            else:
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
        
    except Exception as e:
        print(f"Error in training: {e}")
        return 0.0, 0.0


def test(model, device, test_loader):
    """Test the model with memory optimization"""
    model.eval()
    test_loss = 0
    correct = 0
    
    try:
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
        
    except Exception as e:
        print(f"Error in testing: {e}")
        return float('inf'), 0.0


def run_single_experiment(coreset_size, momentum_value, epochs=10):
    """Run a single experiment with error handling"""
    try:
        # Force CPU usage to avoid CUDA memory issues
        device = torch.device("cpu")
        torch.manual_seed(42)  # Fixed seed for reproducibility
        np.random.seed(42)
        
        # Data setup
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # Model setup
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=momentum_value)  # Reduced LR
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

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

        print(f"\n=== Running experiment: Coreset Size={coreset_size}, Momentum={momentum_value} ===")
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            try:
                # Build coreset (rebuild every few epochs to adapt)
                if epoch == 1 or epoch % 5 == 0:
                    print(f"Building coreset for epoch {epoch}...")
                    coreset_loader = CoresetLoader(
                        train_dataset, 
                        model, 
                        device,
                        sample_size=min(coreset_size, 1000),  # Limit sample size
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
                
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                      f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%, Time={epoch_time:.2f}s")
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                # Fill with default values to maintain consistency
                history['train_loss'].append(float('inf'))
                history['train_acc'].append(0.0)
                history['test_loss'].append(float('inf'))
                history['test_acc'].append(0.0)
                history['epoch_times'].append(0.0)
                continue
        
        return history
        
    except Exception as e:
        print(f"Error in experiment (Size={coreset_size}, Momentum={momentum_value}): {e}")
        traceback.print_exc()
        return None


def save_results(result, coreset_size, momentum_value):
    """Save experiment results"""
    if result is None:
        return
        
    try:
        # Save individual metrics
        metrics = {
            'train_acc': result['train_acc'],
            'test_acc': result['test_acc'],
            'train_loss': result['train_loss'],
            'test_loss': result['test_loss'],
            'epoch_times': result['epoch_times']
        }
        
        for metric_name, values in metrics.items():
            torch.save(values, f'{base_dir}/results/coreset_{coreset_size}_momentum_{momentum_value}_{metric_name}.pt')
        
        # Save full result
        torch.save(result, f'{base_dir}/results/coreset_{coreset_size}_momentum_{momentum_value}_full.pt')
        print(f"Saved results for Size={coreset_size}, Momentum={momentum_value}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def run_experiments_sequential(sizes=[250, 500, 1000], momentum_values=[0.0, 0.9], epochs=20):
    """Run experiments sequentially to avoid memory issues"""
    all_results = {}
    
    # Initialize results structure
    for momentum in momentum_values:
        all_results[f"momentum_{momentum}"] = []
    
    # Run experiments sequentially
    total_experiments = len(sizes) * len(momentum_values)
    current_experiment = 0
    
    for momentum in momentum_values:
        for size in sizes:
            current_experiment += 1
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {current_experiment}/{total_experiments}")
            print(f"Coreset Size: {size}, Momentum: {momentum}")
            print(f"{'='*80}")
            
            # Run experiment
            result = run_single_experiment(size, momentum, epochs)
            
            if result is not None:
                # Save results
                save_results(result, size, momentum)
                
                # Add to combined results
                all_results[f"momentum_{momentum}"].append(result)
                
                print(f"✓ Completed experiment: Size={size}, Momentum={momentum}")
                
                # Print current results
                if len(result['test_acc']) > 0:
                    best_test_acc = max(result['test_acc'])
                    final_test_acc = result['test_acc'][-1]
                    print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
                    print(f"  Final Test Accuracy: {final_test_acc:.2f}%")
            else:
                print(f"✗ Failed experiment: Size={size}, Momentum={momentum}")
    
    # Save combined results
    torch.save(all_results, f'{base_dir}/results/combined_results.pt')
    
    # Create summary and plots
    print_comprehensive_summary(all_results, momentum_values)
    create_comparison_plots(all_results, momentum_values, sizes)
    
    return all_results


def print_comprehensive_summary(all_results, momentum_values):
    """Print a comprehensive summary of all results"""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print(f"{'='*100}")
    print(f"{'Momentum':<10} {'Size':<8} {'Final Train':<12} {'Final Test':<11} {'Best Test':<11} {'Train Acc':<11} {'Test Acc':<11} {'Best Acc':<11}")
    print(f"{'Value':<10} {'':<8} {'Loss':<12} {'Loss':<11} {'Loss':<11} {'(%)':<11} {'(%)':<11} {'(%)':<11}")
    print("-" * 100)
    
    best_overall = {'acc': 0, 'momentum': None, 'size': None}
    
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results:
            results = all_results[momentum_key]
            
            for result in results:
                if len(result['test_acc']) > 0:
                    final_train_loss = result['train_loss'][-1] if result['train_loss'] else float('inf')
                    final_test_loss = result['test_loss'][-1] if result['test_loss'] else float('inf')
                    best_test_loss = min(result['test_loss']) if result['test_loss'] else float('inf')
                    final_train_acc = result['train_acc'][-1] if result['train_acc'] else 0
                    final_test_acc = result['test_acc'][-1] if result['test_acc'] else 0
                    best_test_acc = max(result['test_acc']) if result['test_acc'] else 0
                    
                    print(f"{momentum:<10} {result['coreset_size']:<8} {final_train_loss:<12.4f} {final_test_loss:<11.4f} "
                          f"{best_test_loss:<11.4f} {final_train_acc:<11.2f} {final_test_acc:<11.2f} {best_test_acc:<11.2f}")
                    
                    # Track best overall
                    if best_test_acc > best_overall['acc']:
                        best_overall.update({
                            'acc': best_test_acc,
                            'momentum': momentum,
                            'size': result['coreset_size']
                        })
    
    print("-" * 100)
    print("OVERALL BEST RESULTS:")
    if best_overall['momentum'] is not None:
        print(f"Best Test Accuracy: {best_overall['acc']:.2f}% (Momentum={best_overall['momentum']}, Coreset Size={best_overall['size']})")
    print(f"{'='*100}")


def create_comparison_plots(all_results, momentum_values, sizes):
    """Create comparison plots for different momentum values"""
    try:
        # Set up the plot style
        plt.style.use('default')
        
        # Create a comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MNIST CNN Performance Comparison: Momentum 0.0 vs 0.9', fontsize=16)
        
        # Define colors and markers for different momentum values
        styles = {
            0.0: {'color': 'blue', 'marker': 'o', 'linestyle': '-', 'label': 'Momentum=0.0'},
            0.9: {'color': 'red', 'marker': 's', 'linestyle': '--', 'label': 'Momentum=0.9'}
        }
        
        # Collect data for plotting
        momentum_data = {}
        for momentum in momentum_values:
            momentum_key = f"momentum_{momentum}"
            if momentum_key in all_results:
                sizes_list = []
                final_train_loss = []
                final_test_loss = []
                final_train_acc = []
                final_test_acc = []
                
                for result in all_results[momentum_key]:
                    if len(result['test_acc']) > 0:
                        sizes_list.append(result['coreset_size'])
                        final_train_loss.append(result['train_loss'][-1] if result['train_loss'] else 0)
                        final_test_loss.append(result['test_loss'][-1] if result['test_loss'] else 0)
                        final_train_acc.append(result['train_acc'][-1] if result['train_acc'] else 0)
                        final_test_acc.append(result['test_acc'][-1] if result['test_acc'] else 0)
                
                momentum_data[momentum] = {
                    'sizes': sizes_list,
                    'train_loss': final_train_loss,
                    'test_loss': final_test_loss,
                    'train_acc': final_train_acc,
                    'test_acc': final_test_acc
                }
        
        # Plot 1: Training Loss vs Coreset Size
        ax = axes[0, 0]
        for momentum in momentum_values:
            if momentum in momentum_data:
                data = momentum_data[momentum]
                style = styles[momentum]
                ax.plot(data['sizes'], data['train_loss'], 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
        ax.set_xlabel('Coreset Size')
        ax.set_ylabel('Final Training Loss')
        ax.set_title('Training Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 2: Test Loss vs Coreset Size
        ax = axes[0, 1]
        for momentum in momentum_values:
            if momentum in momentum_data:
                data = momentum_data[momentum]
                style = styles[momentum]
                ax.plot(data['sizes'], data['test_loss'], 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
        ax.set_xlabel('Coreset Size')
        ax.set_ylabel('Final Test Loss')
        ax.set_title('Test Loss Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 3: Training Accuracy vs Coreset Size
        ax = axes[1, 0]
        for momentum in momentum_values:
            if momentum in momentum_data:
                data = momentum_data[momentum]
                style = styles[momentum]
                ax.plot(data['sizes'], data['train_acc'], 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
        ax.set_xlabel('Coreset Size')
        ax.set_ylabel('Final Training Accuracy (%)')
        ax.set_title('Training Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        # Plot 4: Test Accuracy vs Coreset Size
        ax = axes[1, 1]
        for momentum in momentum_values:
            if momentum in momentum_data:
                data = momentum_data[momentum]
                style = styles[momentum]
                ax.plot(data['sizes'], data['test_acc'], 
                       color=style['color'], marker=style['marker'], 
                       linestyle=style['linestyle'], label=style['label'])
        ax.set_xlabel('Coreset Size')
        ax.set_ylabel('Final Test Accuracy (%)')
        ax.set_title('Test Accuracy Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{base_dir}/plots/momentum_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {base_dir}/plots/")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        traceback.print_exc()


def main():
    """Main function to run all experiments"""
    print("Starting MNIST CNN Coreset Experiments with Momentum Comparison")
    print(f"Results will be saved to: {base_dir}")
    
    # Run experiments with a range of coreset sizes and both momentum values
    results = run_experiments_sequential(
        sizes=[100, 250, 500, 1000, 2000],  # Start with smaller sizes
        momentum_values=[0.0, 0.9], 
        epochs=10  # Reduced epochs for faster execution
    )
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")

    
    # Alternatively, if results are already saved, just load and plot them
    # load_and_plot_results()

if __name__ == '__main__':
    main()