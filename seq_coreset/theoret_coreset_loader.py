import torch
import torch.nn.functional as F
import numpy as np
import math
import time
from torch.utils.data import DataLoader


class CoresetLoader:
    """
    A class that builds and loads a coreset based on Algorithm 1, optimized for speed
    while maintaining theoretical correctness
    """
    def __init__(self, dataset, model, device, epsilon=0.1, R=1.0, L=10.0, 
                max_coreset_size=None, batch_size=512, sampling_ratio=0.1):
        """
        Initialize the fast coreset loader following theoretical Algorithm 1
        
        Args:
            dataset: Original dataset
            model: Model used to compute losses
            device: Device to perform computations
            epsilon: Coreset approximation parameter
            R: Region radius as specified in the algorithm
            L: Lipschitz constant as described in Assumption 1
            max_coreset_size: Optional constraint on maximum coreset size
            batch_size: Batch size for faster processing
            sampling_ratio: Ratio of dataset to sample for H estimation (between 0 and 1)
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.max_coreset_size = max_coreset_size
        self.batch_size = batch_size
        self.sampling_ratio = sampling_ratio
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm: N = ⌈log n⌉
        self.N = math.ceil(math.log2(self.n))
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _fast_compute_H(self):
        """
        Efficiently compute an estimate of F(β_anc) using batched processing and sampling
        """
        print(f"Computing H (F(β_anc)) with sampling ratio {self.sampling_ratio}...")
        
        # Determine sample size based on sampling ratio
        sample_size = max(int(self.n * self.sampling_ratio), 1000)
        sample_size = min(sample_size, self.n)  # Don't sample more than available
        
        # Sample indices for H estimation
        indices = np.random.choice(self.n, sample_size, replace=False)
        sampled_dataset = torch.utils.data.Subset(self.dataset, indices)
        
        # Create data loader for batch processing
        dataloader = DataLoader(
            sampled_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=4
        )
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_targets in dataloader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_data)
                batch_loss = F.nll_loss(outputs, batch_targets, reduction='sum').item()
                
                total_loss += batch_loss
                total_samples += len(batch_data)
        
        H = total_loss / total_samples
        print(f"H (F(β_anc)) = {H}")
        return H
    
    def _batch_compute_losses(self, dataloader, H):
        """
        Efficiently compute losses and assign data points to layers using batched processing
        """
        self.model.eval()
        layers = [[] for _ in range(self.N + 1)]
        
        batch_idx_offset = 0
        with torch.no_grad():
            for batch_data, batch_targets in dataloader:
                batch_data = batch_data.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                # Compute losses for the entire batch
                outputs = self.model(batch_data)
                # Get per-example losses
                batch_losses = F.nll_loss(outputs, batch_targets, reduction='none')
                
                # Assign each example to its appropriate layer
                for i, loss in enumerate(batch_losses):
                    global_idx = batch_idx_offset + i
                    loss_val = loss.item()
                    
                    # Layer assignment exactly as per equations (6) and (7)
                    if loss_val <= H:
                        layers[0].append(global_idx)  # Layer P₀
                    else:
                        # Find the appropriate layer P_j for j ≥ 1
                        assigned = False
                        for j in range(1, self.N + 1):
                            if (2**(j-1) * H) < loss_val <= (2**j * H):
                                layers[j].append(global_idx)
                                assigned = True
                                break
                        
                        # If not assigned to any layer 1,...,N, assign to layer N
                        if not assigned:
                            layers[self.N].append(global_idx)
                
                batch_idx_offset += len(batch_data)
        
        return layers
    
    def _calculate_sample_size(self, layer_size, epsilon, R, L):
        """
        Calculate optimized sample size for a layer based on theory and practical constraints
        """
        if layer_size == 0:
            return 0
            
        # This balances theoretical requirements with practical efficiency
        c = 0.5  # Confidence parameter (reduced for practical applications)
        denominator = self.epsilon ** 2
        numerator = c * math.log(self.n) * (self.R ** 2) * (self.L ** 2)
        
        # Calculate theoretical sample size
        theoretical_size = int(numerator / denominator)
        
        # Cap at layer size and ensure at least 1 sample if layer not empty
        sample_size = min(theoretical_size, layer_size)
        sample_size = max(sample_size, 1)
        
        return sample_size
        
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1, optimized for speed
        """
        print("Building fast theoretical coreset...")
        start_time = time.time()
        
        # Step 1: Initialize as per Algorithm 1, using fast estimation
        H = self._fast_compute_H()
        
        # Step 2: Create a DataLoader for batch processing
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=4
        )
        
        # Partition dataset into layers using batch processing
        print(f"Partitioning dataset into {self.N + 1} layers based on loss values...")
        layers = self._batch_compute_losses(dataloader, H)
        
        # Print layer distribution for debugging
        layer_counts = [len(layer) for layer in layers]
        print(f"Layer distribution: {layer_counts}")
        
        # Step 3: For each layer, sample and assign weights
        all_selected_indices = []
        all_weights = []
        
        # First pass: calculate theoretical sizes
        layer_sample_sizes = []
        total_theoretical_size = 0
        
        for j in range(self.N + 1):
            layer_size = len(layers[j])
            if layer_size > 0:
                # Calculate sample size based on theory
                sample_size = self._calculate_sample_size(layer_size, self.epsilon, self.R, self.L)
                layer_sample_sizes.append(sample_size)
                total_theoretical_size += sample_size
            else:
                layer_sample_sizes.append(0)
        
        # Apply maximum coreset size constraint if specified
        scaling_factor = 1.0
        if self.max_coreset_size is not None and total_theoretical_size > self.max_coreset_size:
            scaling_factor = self.max_coreset_size / total_theoretical_size
            print(f"Scaling sample sizes by {scaling_factor:.4f} to meet max_coreset_size constraint")
            
            # Rescale layer sample sizes
            layer_sample_sizes = [max(1, int(size * scaling_factor)) if size > 0 else 0 
                                 for size in layer_sample_sizes]
        
        # Second pass: sample from each layer and assign weights
        for j in range(self.N + 1):
            layer_size = len(layers[j])
            if layer_size > 0:
                sample_size = layer_sample_sizes[j]
                
                # Ensure we don't sample more than available
                sample_size = min(sample_size, layer_size)
                
                # Sample from this layer
                selected = np.random.choice(layers[j], sample_size, replace=False)
                
                # Assign weights according to Algorithm 1, Step 3(b)
                weight = layer_size / sample_size  # This is |P_j|/|Q_j|
                
                all_selected_indices.extend(selected)
                all_weights.extend([weight] * sample_size)
        
        elapsed_time = time.time() - start_time
        print(f"Created fast theoretical coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        print(f"Fast coreset construction time: {elapsed_time:.2f} seconds")
        
        # Load selected data points efficiently
        # Create a small dataset and dataloader for the selected indices
        selected_dataset = torch.utils.data.Subset(self.dataset, all_selected_indices)
        selected_loader = DataLoader(selected_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_data = []
        all_targets = []
        
        # Load data in batches
        for batch_data, batch_targets in selected_loader:
            all_data.append(batch_data)
            all_targets.append(batch_targets)
        
        # Concatenate batches
        if all_data:
            self.data = torch.cat(all_data).to(self.device)
            self.targets = torch.cat(all_targets).to(self.device)
        else:
            self.data = torch.tensor([]).to(self.device)
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