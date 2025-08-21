import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import json
import gc
from concurrent.futures import ThreadPoolExecutor

# Create output directory for plots, results, and model checkpoints
base_dir = './'
os.makedirs(base_dir, exist_ok=True)
os.makedirs(f'{base_dir}/plots', exist_ok=True)
os.makedirs(f'{base_dir}/results', exist_ok=True)
os.makedirs(f'{base_dir}/models', exist_ok=True)


def clear_gpu_memory():
    """Comprehensive GPU memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class ImprovedResNet18(nn.Module):
    """Improved ResNet18 model for CIFAR-10 with better architecture"""
    def __init__(self, num_classes=10, dropout_rate=0.1):
        super(ImprovedResNet18, self).__init__()
        # Start with a fresh model (no pretrained weights)
        model = models.resnet18(pretrained=False)
        
        # IMPROVED: Better first convolutional layer for CIFAR-10
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Remove the max pooling layer (keeps spatial dimensions)
        model.maxpool = nn.Identity()
        
        # IMPROVED: Add dropout for regularization
        model.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate)
        )
        
        # Final classification layer with proper initialization
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.kaiming_normal_(model.fc.weight, mode='fan_out')
        nn.init.constant_(model.fc.bias, 0)
        
        self.model = model
        
        # Apply weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Better weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.model(x)


class CoresetLoader:
    """
    Improved coreset loader with better memory management
    """
    def __init__(self, dataset, model, device, sample_size=2000, epsilon=0.1, R=1.0, L=10.0, max_coreset_size=2000):
        """
        Initialize the coreset loader with improved parameters
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.sample_size = min(sample_size, len(dataset))
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.max_coreset_size = max_coreset_size
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers - reduced for better performance
        self.N = math.ceil(math.log(self.n))  
        
        # Initialize storage
        self.data = None
        self.targets = None
        self.weights = None
        self.coreset_size = 0
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norm_batch(self, data_batch, target_batch):
        """Compute gradient norms for a batch of data points - more memory efficient"""
        self.model.zero_grad()
        
        # Move batch to device
        if not data_batch.is_cuda and self.device.type == 'cuda':
            data_batch = data_batch.to(self.device, non_blocking=True)
        if not target_batch.is_cuda and self.device.type == 'cuda':
            target_batch = target_batch.to(self.device, non_blocking=True)
        
        # Forward pass for the batch
        output = self.model(data_batch)
        loss = F.cross_entropy(output, target_batch)
        
        # Backward pass
        loss.backward()
        
        # Collect and flatten gradients
        grad_norm_squared = 0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm_squared += torch.sum(param.grad ** 2).item()
        
        # Clear gradients immediately
        self.model.zero_grad()
        
        # Clear intermediate tensors
        del output, loss
        
        return math.sqrt(grad_norm_squared)
    
    def _compute_model_loss(self):
        """Compute the overall loss at the anchor point with better sampling"""
        self.model.eval()
        sample_size = min(50, self.n)  # Further reduced sample size for memory efficiency
        indices = np.random.choice(self.n, sample_size, replace=False)
        
        total_loss = 0.0
        batch_size = 16  # Smaller batch size
        
        with torch.no_grad():
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_data = []
                batch_targets = []
                
                for idx in batch_indices:
                    data, target = self.dataset[idx]
                    batch_data.append(data)
                    batch_targets.append(target)
                
                if batch_data:
                    batch_data = torch.stack(batch_data)
                    batch_targets = torch.tensor(batch_targets, dtype=torch.long)
                    
                    if self.device.type == 'cuda':
                        batch_data = batch_data.to(self.device, non_blocking=True)
                        batch_targets = batch_targets.to(self.device, non_blocking=True)
                    
                    output = self.model(batch_data)
                    loss = F.cross_entropy(output, batch_targets, reduction='sum')
                    total_loss += loss.item()
                    
                    del output, loss, batch_data, batch_targets
                
                clear_gpu_memory()
        
        return total_loss / sample_size
    
    def _build_coreset(self):
        """
        Build the coreset with improved memory management
        """
        print("Building coreset...")
        start_time = time.time()
        
        # Clear memory before starting
        clear_gpu_memory()
        
        # Step 1: Initialize
        H = self._compute_model_loss()
        
        # Better sampling strategy - use much smaller sample for gradient estimation
        sample_size = min(self.sample_size // 2, 1000)  # Reduced sample size
        indices = np.random.choice(self.n, sample_size, replace=False)
        
        # Compute gradient norms in smaller batches
        grad_norms = []
        self.model.eval()
        
        batch_size = 8  # Even smaller batch size for gradient computation
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_data = []
            batch_targets = []
            
            for idx in batch_indices:
                data, target = self.dataset[idx]
                batch_data.append(data)
                batch_targets.append(target)
            
            if batch_data:
                batch_data = torch.stack(batch_data)
                batch_targets = torch.tensor(batch_targets, dtype=torch.long)
                
                # Compute gradient norm for this batch
                norm = self._compute_grad_norm_batch(batch_data, batch_targets)
                
                # Assign the same norm to all samples in the batch (approximation)
                for j, idx in enumerate(batch_indices):
                    grad_norms.append((idx, norm))
                
                del batch_data, batch_targets
            
            if i % (batch_size * 10) == 0:
                clear_gpu_memory()
                print(f"Processed {i}/{len(indices)} samples for gradient computation")
        
        # Layer partitioning with better distribution
        layers = [[] for _ in range(self.N + 1)]
        
        if grad_norms:
            # Sort by gradient norm
            grad_norms.sort(key=lambda x: x[1])
            
            # Use percentile-based thresholds
            norms_only = [norm for _, norm in grad_norms]
            percentiles = np.linspace(0, 100, self.N + 1)
            thresholds = [np.percentile(norms_only, p) for p in percentiles]
            
            # Assign samples to layers
            for idx, norm in grad_norms:
                for j in range(self.N + 1):
                    if j == self.N or norm <= thresholds[j]:
                        layers[j].append(idx)
                        break
        
        # Sample distribution across layers
        all_selected_indices = []
        all_weights = []
        
        actual_coreset_size = min(self.n, self.max_coreset_size)
        non_empty_layers = [(j, layer) for j, layer in enumerate(layers) if len(layer) > 0]
        
        if not non_empty_layers:
            # Fallback: random sampling
            selected_indices = np.random.choice(self.n, actual_coreset_size, replace=False)
            all_selected_indices.extend(selected_indices)
            all_weights.extend([1.0] * len(selected_indices))
        else:
            # Distribute samples more evenly
            samples_per_layer = max(1, actual_coreset_size // len(non_empty_layers))
            remaining_samples = actual_coreset_size
            
            for j, layer in non_empty_layers:
                if remaining_samples <= 0:
                    break
                    
                size = min(samples_per_layer, len(layer), remaining_samples)
                if size == 0:
                    continue
                
                selected = np.random.choice(layer, size, replace=False)
                weight = len(layer) / size
                
                all_selected_indices.extend(selected)
                all_weights.extend([weight] * size)
                
                remaining_samples -= size
        
        elapsed_time = time.time() - start_time
        print(f"Created coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        print(f"Coreset construction time: {elapsed_time:.2f} seconds")
        
        # Load selected data in smaller batches
        all_data = []
        all_targets = []
        
        batch_size = 50  # Process in smaller batches
        for i in range(0, len(all_selected_indices), batch_size):
            batch_indices = all_selected_indices[i:i+batch_size]
            
            for idx in batch_indices:
                data, target = self.dataset[idx]
                all_data.append(data)
                all_targets.append(target)
            
            if i % (batch_size * 5) == 0:
                clear_gpu_memory()
        
        # Create coreset tensors - keep on CPU initially to save GPU memory
        if all_data:
            self.data = torch.stack(all_data)
            # Only move to GPU when needed during training
        else:
            self.data = torch.tensor([])
            
        if all_targets:
            self.targets = torch.tensor(all_targets, dtype=torch.long)
        else:
            self.targets = torch.tensor([], dtype=torch.long)
            
        self.weights = torch.tensor(all_weights, dtype=torch.float)
        self.coreset_size = len(all_selected_indices)
        
        # Final memory cleanup
        clear_gpu_memory()
    
    def __iter__(self):
        """Iterator that yields the entire coreset once"""
        # Move data to device only when yielding
        data = self.data.to(self.device, non_blocking=True) if self.device.type == 'cuda' else self.data
        targets = self.targets.to(self.device, non_blocking=True) if self.device.type == 'cuda' else self.targets
        weights = self.weights.to(self.device, non_blocking=True) if self.device.type == 'cuda' else self.weights
        yield data, targets, weights
    
    def __len__(self):
        return 1
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'data') and self.data is not None:
            del self.data
        if hasattr(self, 'targets') and self.targets is not None:
            del self.targets
        if hasattr(self, 'weights') and self.weights is not None:
            del self.weights
        clear_gpu_memory()


def train_with_mixup(model, coreset_loader, optimizer, alpha=0.1):
    """Improved training with mixup augmentation and better memory management"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for data, target, weights in coreset_loader:
        optimizer.zero_grad()
        
        # Process in smaller chunks to avoid memory issues
        chunk_size = min(64, len(data))  # Smaller chunk size
        
        chunk_losses = []
        chunk_correct = 0
        
        for i in range(0, len(data), chunk_size):
            chunk_data = data[i:i+chunk_size]
            chunk_target = target[i:i+chunk_size]
            chunk_weights = weights[i:i+chunk_size]
            
            # Apply mixup augmentation
            if alpha > 0 and len(chunk_data) > 1:
                lam = np.random.beta(alpha, alpha)
                batch_size = chunk_data.size(0)
                index = torch.randperm(batch_size).to(chunk_data.device)
                
                mixed_data = lam * chunk_data + (1 - lam) * chunk_data[index, :]
                target_a, target_b = chunk_target, chunk_target[index]
                mixed_weights = lam * chunk_weights + (1 - lam) * chunk_weights[index]
                
                output = model(mixed_data)
                loss_a = F.cross_entropy(output, target_a, reduction='none')
                loss_b = F.cross_entropy(output, target_b, reduction='none')
                loss = (lam * loss_a + (1 - lam) * loss_b) * mixed_weights
                loss = loss.mean()
                
                # For accuracy calculation (use original data)
                with torch.no_grad():
                    original_output = model(chunk_data)
                    pred = original_output.argmax(dim=1, keepdim=True)
                    chunk_correct += pred.eq(chunk_target.view_as(pred)).sum().item()
                    del original_output, pred
            else:
                # Standard training without mixup
                output = model(chunk_data)
                individual_losses = F.cross_entropy(output, chunk_target, reduction='none')
                loss = (individual_losses * chunk_weights).mean()
                
                pred = output.argmax(dim=1, keepdim=True)
                chunk_correct += pred.eq(chunk_target.view_as(pred)).sum().item()
                del pred, individual_losses
            
            chunk_losses.append(loss)
            del output, loss
        
        # Combine losses from all chunks
        if chunk_losses:
            total_chunk_loss = sum(chunk_losses) / len(chunk_losses)
            total_chunk_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_chunk_loss.item()
            correct += chunk_correct
            total += len(target)
            
            del total_chunk_loss
        
        # Clear memory after processing each batch
        clear_gpu_memory()

    accuracy = 100. * correct / total if total > 0 else 0
    return total_loss, accuracy


def test(model, device, test_loader):
    """Improved testing function with better memory management"""
    model.eval()
    test_loss, correct = 0, 0
    
    clear_gpu_memory()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            del output, pred
            
            if batch_idx % 10 == 0:  # More frequent cleanup
                clear_gpu_memory()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def run_improved_experiment(coreset_size, momentum_value, epochs=100):
    """Improved experiment with fixed optimizer settings and better memory management"""
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    np.random.seed(42)
    
    clear_gpu_memory()
    
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Lighter data augmentation for memory efficiency
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
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
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=2, pin_memory=True)  # Smaller batch size

    # Model with reduced dropout for smaller coresets
    dropout_rate = 0.05 if coreset_size < 1000 else 0.1
    model = ImprovedResNet18(dropout_rate=dropout_rate).to(device)
    
    # FIXED: Conditional Nesterov momentum based on momentum value
    if momentum_value > 0:
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=momentum_value,  # Reduced learning rate
                             weight_decay=1e-4, nesterov=True)
    else:
        # For momentum=0, don't use Nesterov acceleration
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=momentum_value, 
                             weight_decay=1e-4, nesterov=False)
    
    # Learning rate schedule
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)

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
    
    best_test_acc = 0
    patience = 30
    epochs_without_improvement = 0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        try:
            # Create coreset loader for this epoch
            coreset_loader = CoresetLoader(
                train_dataset, model, device,
                sample_size=min(coreset_size, 1000),  # Reduced sample size
                epsilon=0.05,
                R=1.0, L=10.0, 
                max_coreset_size=coreset_size
            )

            # Training with reduced mixup for memory efficiency
            train_loss, train_acc = train_with_mixup(model, coreset_loader, optimizer, alpha=0.05)
            test_loss, test_acc = test(model, device, test_loader)
            scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['epoch_times'].append(time.time() - epoch_start)
            
            # Early stopping logic
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                epochs_without_improvement = 0
                # Save best model
                torch.save(model.state_dict(), 
                          f'{base_dir}/models/best_resnet18_coreset_{coreset_size}_momentum_{momentum_value}.pt')
            else:
                epochs_without_improvement += 1
            
            print(f"Epoch {epoch}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}, "
                  f"Time: {history['epoch_times'][-1]:.2f}s")
            
            # Early stopping
            if epochs_without_improvement >= patience and epoch > 50:
                print(f"Early stopping at epoch {epoch} (best test acc: {best_test_acc:.2f}%)")
                break
            
            # Clear memory after each epoch
            del coreset_loader
            clear_gpu_memory()
            
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            clear_gpu_memory()
            # Try to continue with next epoch
            continue

    print(f"Best test accuracy achieved: {best_test_acc:.2f}%")
    
    # Clear model from memory
    del model, optimizer, scheduler
    clear_gpu_memory()
    
    return history


def save_results(result, coreset_size, momentum_value):
    """Save experiment results"""
    metrics = {
        'train_acc': result['train_acc'],
        'test_acc': result['test_acc'],
        'train_loss': result['train_loss'],
        'test_loss': result['test_loss'],
        'epoch_times': result['epoch_times']
    }
    for metric_name, values in metrics.items():
        torch.save(values, f'{base_dir}/results/improved_coreset_{coreset_size}_momentum_{momentum_value}_{metric_name}.pt')
    torch.save(result, f'{base_dir}/results/improved_coreset_{coreset_size}_momentum_{momentum_value}_full.pt')


def plot_comprehensive_results(all_results, momentum_values, sizes):
    """Create comprehensive plots for all coreset sizes with fixed dimension matching"""
    
    # Define plot styles for different momentum values
    styles = {
        0.0: {'marker': 'o', 'linestyle': '-', 'color': 'green', 'markersize': 6},
        0.9: {'marker': 's', 'linestyle': '--', 'color': 'darkorange', 'markersize': 6}
    }
    
    # Overall performance vs coreset size
    plt.figure(figsize=(24, 6))
    
    # Training Loss vs Coreset Size
    plt.subplot(1, 4, 1)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            results = all_results[momentum_key]
            
            # Filter out empty results and ensure matching dimensions
            valid_results = [r for r in results if r['train_loss'] and len(r['train_loss']) > 0]
            
            if valid_results:
                sizes_list = [r['coreset_size'] for r in valid_results]
                final_train_loss = [r['train_loss'][-1] for r in valid_results]
                
                # Ensure equal lengths
                if len(sizes_list) == len(final_train_loss) and len(sizes_list) > 0:
                    style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
                    plt.plot(sizes_list, final_train_loss, label=f'Momentum={momentum}', **style)
    
    plt.title('Final Training Loss vs Coreset Size', fontsize=12, fontweight='bold')
    plt.xlabel('Coreset Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Test Loss vs Coreset Size
    plt.subplot(1, 4, 2)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            results = all_results[momentum_key]
            
            # Filter out empty results
            valid_results = [r for r in results if r['test_loss'] and len(r['test_loss']) > 0]
            
            if valid_results:
                sizes_list = [r['coreset_size'] for r in valid_results]
                final_test_loss = [r['test_loss'][-1] for r in valid_results]
                
                if len(sizes_list) == len(final_test_loss) and len(sizes_list) > 0:
                    style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
                    plt.plot(sizes_list, final_test_loss, label=f'Momentum={momentum}', **style)
    
    plt.title('Final Test Loss vs Coreset Size', fontsize=12, fontweight='bold')
    plt.xlabel('Coreset Size')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Training Accuracy vs Coreset Size
    plt.subplot(1, 4, 3)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            results = all_results[momentum_key]
            
            valid_results = [r for r in results if r['train_acc'] and len(r['train_acc']) > 0]
            
            if valid_results:
                sizes_list = [r['coreset_size'] for r in valid_results]
                final_train_acc = [r['train_acc'][-1] for r in valid_results]
                
                if len(sizes_list) == len(final_train_acc) and len(sizes_list) > 0:
                    style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
                    plt.plot(sizes_list, final_train_acc, label=f'Momentum={momentum}', **style)
    
    plt.title('Final Training Accuracy vs Coreset Size', fontsize=12, fontweight='bold')
    plt.xlabel('Coreset Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Test Accuracy vs Coreset Size
    plt.subplot(1, 4, 4)
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            results = all_results[momentum_key]
            
            valid_results = [r for r in results if r['test_acc'] and len(r['test_acc']) > 0]
            
            if valid_results:
                sizes_list = [r['coreset_size'] for r in valid_results]
                final_test_acc = [r['test_acc'][-1] for r in valid_results]
                best_test_acc = [max(r['test_acc']) if r['test_acc'] else 0 for r in valid_results]
                
                if len(sizes_list) == len(final_test_acc) and len(sizes_list) > 0:
                    style = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
                    plt.plot(sizes_list, final_test_acc, label=f'Final - M={momentum}', **style)
                    
                    # Also plot best accuracy achieved
                    if len(sizes_list) == len(best_test_acc):
                        style_best = styles.get(momentum, {'marker': 'o', 'linestyle': '-'})
                        style_best['linestyle'] = ':'
                        style_best['alpha'] = 0.7
                        plt.plot(sizes_list, best_test_acc, label=f'Best - M={momentum}', **style_best)
    
    plt.title('Test Accuracy vs Coreset Size', fontsize=12, fontweight='bold')
    plt.xlabel('Coreset Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{base_dir}/plots/comprehensive_metrics_improved.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_comprehensive_summary(all_results, momentum_values):
    """Print comprehensive performance summary"""
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Momentum':<10} {'Size':<8} {'Final Train':<12} {'Final Test':<12} {'Best Test':<12} {'Train Acc':<12} {'Test Acc':<12} {'Best Acc':<12}")
    print(f"{'Value':<10} {'      ':<8} {'Loss':<12} {'Loss':<12} {'Loss':<12} {'(%)':<12} {'(%)':<12} {'(%)':<12}")
    print("-" * 100)

    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            results = sorted(all_results[momentum_key], key=lambda x: x['coreset_size'])
            for res in results:
                if res['test_loss'] and res['test_acc'] and res['train_loss'] and res['train_acc']:
                    best_test_loss = min(res['test_loss'])
                    best_test_acc = max(res['test_acc'])
                    print(f"{momentum:<10} {res['coreset_size']:<8} {res['train_loss'][-1]:<12.4f} "
                          f"{res['test_loss'][-1]:<12.4f} {best_test_loss:<12.4f} {res['train_acc'][-1]:<12.2f} "
                          f"{res['test_acc'][-1]:<12.2f} {best_test_acc:<12.2f}")
            print("-" * 100)
    
    # Find overall best results
    print("\nOVERALL BEST RESULTS:")
    best_acc = 0
    best_config = None
    
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        if momentum_key in all_results and all_results[momentum_key]:
            for res in all_results[momentum_key]:
                max_acc = max(res['test_acc'])
                if max_acc > best_acc:
                    best_acc = max_acc
                    best_config = (momentum, res['coreset_size'])
    
    if best_config:
        print(f"Best Test Accuracy: {best_acc:.2f}% (Momentum={best_config[0]}, Coreset Size={best_config[1]})")


def run_multiple_coreset_experiments(
    sizes=[250, 500, 1000, 2000, 4000, 8000, 16000], 
    momentum_values=[0.0, 0.9], 
    epochs=200
):
    """Run experiments with multiple coreset sizes"""
    
    all_results = {}
    total_experiments = len(sizes) * len(momentum_values)
    current_experiment = 0
    
    print(f"Starting comprehensive experiments with {total_experiments} configurations...")
    print(f"Coreset sizes: {sizes}")
    print(f"Momentum values: {momentum_values}")
    print(f"Epochs per experiment: {epochs}")
    print("="*80)
    
    for momentum in momentum_values:
        momentum_key = f"momentum_{momentum}"
        all_results[momentum_key] = []
        
        for size in sizes:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] Starting experiment:")
            print(f"  Coreset Size: {size}")
            print(f"  Momentum: {momentum}")
            print(f"  Max Epochs: {epochs}")
            print("-" * 50)
            
            try:
                # Run experiment with error handling
                result = run_improved_experiment(size, momentum, epochs)
                
                if result is not None:
                    all_results[momentum_key].append(result)
                    save_results(result, size, momentum)
                    
                    # Print intermediate results
                    final_acc = result['test_acc'][-1]
                    best_acc = max(result['test_acc'])
                    print(f"  ✓ Completed: Final Acc = {final_acc:.2f}%, Best Acc = {best_acc:.2f}%")
                else:
                    print(f"  ✗ Failed: No results returned")
                    
            except Exception as e:
                print(f"  ✗ Error in experiment: {e}")
                clear_gpu_memory()
                continue
            
            # Clear memory between experiments
            clear_gpu_memory()
            
            # Save intermediate results
            if all_results[momentum_key]:
                torch.save(all_results, f'{base_dir}/results/intermediate_comprehensive_results.pt')
    
    # Save final combined results
    if any(all_results.values()):
        torch.save(all_results, f'{base_dir}/results/comprehensive_results_500epochs.pt')
        
        # Create comprehensive plots
        plot_comprehensive_results(all_results, momentum_values, sizes)
        
        # Print summary
        print_comprehensive_summary(all_results, momentum_values)
        
        print(f"\n{'='*80}")
        print("ALL EXPERIMENTS COMPLETED!")
        print(f"Results saved to: {base_dir}/results/comprehensive_results_500epochs.pt")
        print(f"Plots saved to: {base_dir}/plots/")
        print(f"{'='*80}")
    else:
        print("No successful experiments to analyze")


def main():
    """Main function with comprehensive experiment setup"""
    # Set memory management environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    print("Starting Comprehensive CIFAR-10 Coreset Experiments")
    print("="*60)
    
    # Define experiment parameters
    coreset_sizes = [ 500, 2000, 8000, 20000]
    momentum_values = [0.0, 0.9]
    max_epochs = 200
    
    # Run comprehensive experiments
    run_multiple_coreset_experiments(
        sizes=coreset_sizes,
        momentum_values=momentum_values,
        epochs=max_epochs
    )


if __name__ == '__main__':
    main()