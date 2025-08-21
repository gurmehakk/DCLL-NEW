class CoresetLoader:
    """
    A combined class that builds and loads a coreset according to Algorithm 1
    """
    def __init__(self, dataset, model,  device, max_coreset_size, epsilon=0.1, R=1.0, L=10.0, delta=0.01, lambda_val=0.01):
        """
        Initialize the coreset loader
        
        Args:
            dataset: Original dataset
            model: Model used to compute gradients
            device: Device to perform computations
            epsilon: Coreset approximation parameter (ε)
            R: Region radius
            L: Lipschitz constant
            delta: Confidence parameter for Hoeffding's inequality
            lambda_val: Probability bound parameter (λ)
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.max_coreset_size = max_coreset_size
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.delta = delta
        self.lambda_val = lambda_val
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm
        self.N = math.ceil(math.log(self.n))
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norm(self, data, target):
        """
        Compute gradient norm for a single data point
        This corresponds to ||∇f(βanc, xi, yi)|| in the algorithm
        """
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
        """
        Compute the overall loss at the anchor point (F(βanc))
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for i in range(self.n):
                data, target = self.dataset[i]
                data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
                output = self.model(data.unsqueeze(0))
                loss = F.nll_loss(output, target.unsqueeze(0))
                total_loss += loss.item()
        
        return total_loss / self.n
    
    def _calculate_sample_size(self, j, H, M):
        """
        Calculate the sample size |Qj| according to equation (9) in Lemma 1
        
        Args:
            j: Layer index
            H: F(βanc) value
            M: Maximum gradient norm
        """
        # Using equation (9): |Qj| = O((2^(j-1)H + MR + LR^2)^2 * δ^(-2) * log(1/λ))
        if j == 0:
            # For j = 0, we use H instead of 2^(j-1)H
            numerator = (H + self.R * M + 0.5 * self.L * self.R**2)**2
        else:
            numerator = ((2**(j-1)) * H + self.R * M + 0.5 * self.L * self.R**2)**2
        
        # Calculate sample size with a constant factor (you might need to adjust this)
        constant = 0.5  # This is a tunable parameter
        sample_size = int(constant * numerator * (1/self.delta**2) * math.log(1/self.lambda_val))
        
        # Ensure minimum sample size of 1
        return max(1, sample_size)
    
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1
        """
        print("Building coreset according to Algorithm 1...")
        start_time = time.time()
        
        # Step 1: Initialize as per Algorithm 1
        H = self._compute_model_loss()  # This is F(βanc) in Algorithm 1
        
        # Initialize weight vector W = [0, 0, ..., 0]
        W = np.zeros(self.n)
        
        # Compute gradient norms for all points
        print("Computing gradient norms for all data points...")
        grad_norms = []
        self.model.eval()
        
        for i in range(self.n):
            data, target = self.dataset[i]
            data, target = data.to(self.device), torch.tensor(target, dtype=torch.long).to(self.device)
            norm = self._compute_grad_norm(data, target)
            grad_norms.append((i, norm))
        
        # Calculate M = max||∇f(βanc, xi, yi)||
        M = max([norm for _, norm in grad_norms])
        
        # Step 2: Partition into layers based on gradient norms
        layers = [[] for _ in range(self.N + 1)]
        
        # Partition according to equations (6) and (7)
        for idx, norm in grad_norms:
            if norm <= H:  # P0 = {(xi, yi) | ||∇f(βanc, xi, yi)|| ≤ H}
                layers[0].append(idx)
            else:
                for j in range(1, self.N + 1):
                    if j == self.N or norm <= (2**j) * H:  # Pj = {(xi, yi) | 2^(j-1)H < ||∇f(βanc, xi, yi)|| ≤ 2^j H}
                        layers[j].append(idx)
                        break
        
        # Step 3: For each layer, sample and assign weights
        all_selected_indices = []
        
        for j in range(self.N + 1):
            if len(layers[j]) > 0:
                # Calculate sample size for this layer using equation (9)
                size = self._calculate_sample_size(j, H, M)
                
                # Ensure we don't sample more than available
                size = min(size, len(layers[j]))
                
                # Sample from this layer
                if size < len(layers[j]):
                    selected = np.random.choice(layers[j], size, replace=False)
                else:
                    selected = layers[j]  # Take all if sample size ≥ layer size
                
                # Assign weights according to step 3(b) in Algorithm 1
                weight = len(layers[j]) / size  # |Pj|/|Qj|
                
                # Update weight vector W
                for idx in selected:
                    W[idx] = weight
                
                all_selected_indices.extend(selected)
        
        elapsed_time = time.time() - start_time
        print(f"Created coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        print(f"Coreset construction time: {elapsed_time:.2f} seconds")
        
        # Load all selected data into memory
        all_data = []
        all_targets = []
        all_weights = []
        
        for idx in all_selected_indices:
            data, target = self.dataset[idx]
            all_data.append(data)
            all_targets.append(target)
            all_weights.append(W[idx])
        
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
        
        # Store original weight vector and coreset size for reference
        self.W = W
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

#faster + max coreset util
class OptimizedCoresetLoader:
    """
    A faster implementation of CoresetLoader that respects max_coreset_size
    """
    def __init__(self, dataset, model, device, max_coreset_size, epsilon=0.1, R=1.0, L=10.0, delta=0.01, lambda_val=0.01, batch_size=64):
        """
        Initialize the coreset loader
        
        Args:
            dataset: Original dataset
            model: Model used to compute gradients
            device: Device to perform computations
            max_coreset_size: Maximum size of the coreset (hard limit)
            epsilon: Coreset approximation parameter (ε)
            R: Region radius
            L: Lipschitz constant
            delta: Confidence parameter for Hoeffding's inequality
            lambda_val: Probability bound parameter (λ)
            batch_size: Batch size for gradient computation
        """
        self.dataset = dataset
        self.model = model
        self.device = device
        self.max_coreset_size = max_coreset_size
        self.epsilon = epsilon
        self.R = R
        self.L = L
        self.delta = delta
        self.lambda_val = lambda_val
        self.batch_size = batch_size
        
        # Store original dataset size
        self.n = len(dataset)
        # Number of layers as per algorithm
        self.N = math.ceil(math.log(self.n))
        
        # Build the coreset when initialized
        self._build_coreset()
    
    def _compute_grad_norms_batched(self):
        """
        Compute gradient norms for all data points in batches
        """
        grad_norms = []
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        
        print(f"Computing gradient norms in batches of {self.batch_size}...")
        
        for batch_idx, (data_batch, target_batch) in enumerate(dataloader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(dataloader)}")
                
            data_batch = data_batch.to(self.device)
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(self.device)
            
            batch_norms = []
            # Process each point in the batch individually
            for i in range(len(data_batch)):
                data = data_batch[i]
                target = target_batch[i]
                
                self.model.zero_grad()
                output = self.model(data.unsqueeze(0))
                loss = F.nll_loss(output, target.unsqueeze(0))
                loss.backward()
                
                # Collect and flatten gradients
                grad_norm = 0
                for param in self.model.parameters():
                    if param.grad is not None:
                        grad_norm += torch.sum(param.grad ** 2)
                
                batch_norms.append((batch_idx * self.batch_size + i, torch.sqrt(grad_norm).item()))
            
            grad_norms.extend(batch_norms)
        
        return grad_norms
    
    def _compute_model_loss_batched(self):
        """
        Compute the overall loss at the anchor point (F(βanc)) using batches
        """
        self.model.eval()
        total_loss = 0.0
        
        dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )
        
        with torch.no_grad():
            for data_batch, target_batch in dataloader:
                data_batch = data_batch.to(self.device)
                target_batch = torch.tensor(target_batch, dtype=torch.long).to(self.device)
                
                output = self.model(data_batch)
                loss = F.nll_loss(output, target_batch)
                total_loss += loss.item() * len(data_batch)
        
        return total_loss / self.n
    
    def _calculate_sample_size(self, j, H, M):
        """
        Calculate the sample size |Qj| according to equation (9) in Lemma 1
        
        Args:
            j: Layer index
            H: F(βanc) value
            M: Maximum gradient norm
        """
        # Using equation (9): |Qj| = O((2^(j-1)H + MR + LR^2)^2 * δ^(-2) * log(1/λ))
        if j == 0:
            # For j = 0, we use H instead of 2^(j-1)H
            numerator = (H + self.R * M + 0.5 * self.L * self.R**2)**2
        else:
            numerator = ((2**(j-1)) * H + self.R * M + 0.5 * self.L * self.R**2)**2
        
        # Calculate sample size with a constant factor (you might need to adjust this)
        constant = 0.5  # This is a tunable parameter
        sample_size = int(constant * numerator * (1/self.delta**2) * math.log(1/self.lambda_val))
        
        # Ensure minimum sample size of 1
        return max(1, sample_size)
    
    def _build_coreset(self):
        """
        Build the coreset according to Algorithm 1 with optimizations
        """
        print("Building optimized coreset...")
        start_time = time.time()
        
        # Step 1: Initialize as per Algorithm 1
        H = self._compute_model_loss_batched()  # This is F(βanc) in Algorithm 1
        
        # Initialize weight vector W = [0, 0, ..., 0]
        W = np.zeros(self.n)
        
        # Compute gradient norms for all points in batches
        grad_norms = self._compute_grad_norms_batched()
        
        # Calculate M = max||∇f(βanc, xi, yi)||
        M = max([norm for _, norm in grad_norms])
        
        # Step 2: Partition into layers based on gradient norms
        layers = [[] for _ in range(self.N + 1)]
        
        # Partition according to equations (6) and (7)
        for idx, norm in grad_norms:
            if norm <= H:  # P0 = {(xi, yi) | ||∇f(βanc, xi, yi)|| ≤ H}
                layers[0].append(idx)
            else:
                for j in range(1, self.N + 1):
                    if j == self.N or norm <= (2**j) * H:  # Pj = {(xi, yi) | 2^(j-1)H < ||∇f(βanc, xi, yi)|| ≤ 2^j H}
                        layers[j].append(idx)
                        break
        
        # Step 3: Calculate sample sizes for each layer while respecting max_coreset_size
        layer_sample_sizes = []
        total_samples = 0
        
        # First pass: calculate desired sample sizes
        for j in range(self.N + 1):
            if len(layers[j]) > 0:
                # Calculate theoretical sample size
                size = self._calculate_sample_size(j, H, M)
                size = min(size, len(layers[j]))
                layer_sample_sizes.append((j, size))
                total_samples += size
        
        # Second pass: scale down if needed to respect max_coreset_size
        if total_samples > self.max_coreset_size:
            scaling_factor = self.max_coreset_size / total_samples
            adjusted_sizes = []
            
            for j, size in layer_sample_sizes:
                # Scale down and ensure at least 1 sample per non-empty layer
                adjusted_size = max(1, int(size * scaling_factor))
                adjusted_sizes.append((j, adjusted_size))
            
            # Ensure we don't exceed max_coreset_size due to rounding
            while sum(size for _, size in adjusted_sizes) > self.max_coreset_size:
                # Find layer with largest adjusted size and decrement
                largest_idx = max(range(len(adjusted_sizes)), key=lambda i: adjusted_sizes[i][1])
                j, size = adjusted_sizes[largest_idx]
                if size > 1:  # Ensure we keep at least 1 sample
                    adjusted_sizes[largest_idx] = (j, size - 1)
            
            layer_sample_sizes = adjusted_sizes
        
        # Step 4: Sample from each layer and assign weights
        all_selected_indices = []
        
        for j, size in layer_sample_sizes:
            if size > 0:
                # Sample from this layer
                if size < len(layers[j]):
                    selected = np.random.choice(layers[j], size, replace=False)
                else:
                    selected = layers[j]  # Take all if sample size ≥ layer size
                
                # Assign weights according to step 3(b) in Algorithm 1
                weight = len(layers[j]) / len(selected)  # |Pj|/|Qj|
                
                # Update weight vector W
                for idx in selected:
                    W[idx] = weight
                
                all_selected_indices.extend(selected)
        
        elapsed_time = time.time() - start_time
        print(f"Created coreset with {len(all_selected_indices)} samples from {self.n} original samples")
        print(f"Coreset construction time: {elapsed_time:.2f} seconds")
        
        # Load all selected data into memory
        all_data = []
        all_targets = []
        all_weights = []
        
        for idx in all_selected_indices:
            data, target = self.dataset[idx]
            all_data.append(data)
            all_targets.append(target)
            all_weights.append(W[idx])
        
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
        
        # Store original weight vector and coreset size for reference
        self.W = W
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