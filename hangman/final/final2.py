# # Trexquant Interview Project (The Hangman Game)
# 
# * Copyright Trexquant Investment LP. All Rights Reserved. 
# * Redistribution of this question without written consent from Trexquant is prohibited

# ## Instruction:
# For this coding test, your mission is to write an algorithm that plays the game of Hangman through our API server. 
# 
# When a user plays Hangman, the server first selects a secret word at random from a list. The server then returns a row of underscores (space separated)—one for each letter in 
# the secret word—and asks the user to guess a letter. If the user guesses a letter that is in the word, the word is redisplayed with all instances of that letter shown in the 
# correct positions, along with any letters correctly guessed on previous turns. If the letter does not appear in the word, the user is charged with an incorrect guess. 
# The user keeps guessing letters until either (1) the user has correctly guessed all the letters in the word
# or (2) the user has made six incorrect guesses.
# 
# You are required to write a "guess" function that takes current word (with underscores) as input and returns a guess letter. You will use the API codes below to play 1,000 Hangman games.
# You have the opportunity to practice before you want to start recording your game results.
# 
# Your algorithm is permitted to use a training set of approximately 250,000 dictionary words. Your algorithm will be tested on an entirely disjoint set of 250,000 dictionary words. 
# Please note that this means the words that you will ultimately be tested on do NOT appear in the dictionary that you are given. 
# You are not permitted to use any dictionary other than the training dictionary we provided. This requirement will be strictly enforced by code review.
# 
# You are provided with a basic, working algorithm. This algorithm will match the provided masked string (e.g. a _ _ l e) to all possible words in the dictionary, 
# tabulate the frequency of letters appearing in these possible words, and then guess the letter with the highest frequency of appearence that has not already been guessed. 
# If there are no remaining words that match then it will default back to the character frequency distribution of the entire dictionary.
# 
# This benchmark strategy is successful approximately 18% of the time. Your task is to design an algorithm that significantly outperforms this benchmark.

import collections
import json
import math
import os
import random
import re
import string
import time

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class WordSequenceDataset(Dataset):
    """Custom dataset for training character sequence prediction models"""
    
    def __init__(self, vocabulary_list, obfuscation_rate=0.25, max_seq_len=15):
        self.word_corpus = [w for w in vocabulary_list if w.isalpha() and 0 < len(w) <= max_seq_len]
        self.obfuscation_prob = obfuscation_rate
        self.sequence_limit = max_seq_len
        
        # Create vocabulary mapping with special tokens
        self.token_vocab = ['<NULL>', '<HIDDEN>'] + list(string.ascii_lowercase)
        self.token_to_idx = {token: i for i, token in enumerate(self.token_vocab)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}
        
    def __len__(self):
        return len(self.word_corpus)

    def __getitem__(self, idx):
        target_word = self.word_corpus[idx]
        char_sequence = list(target_word.lower())
        
        # Pad sequence to fixed length
        if len(char_sequence) < self.sequence_limit:
            char_sequence.extend(['<NULL>'] * (self.sequence_limit - len(char_sequence)))
        
        encoder_seq = []
        decoder_seq = []
        
        for character in char_sequence:
            if character == '<NULL>':
                encoder_seq.append(self.token_to_idx['<NULL>'])
                decoder_seq.append(-100)  # Ignore index for loss calculation
            else:
                if random.random() < self.obfuscation_prob:
                    encoder_seq.append(self.token_to_idx['<HIDDEN>'])
                    decoder_seq.append(self.token_to_idx[character])
                else:
                    encoder_seq.append(self.token_to_idx[character])
                    decoder_seq.append(-100)  # No loss for revealed characters
        
        return torch.tensor(encoder_seq, dtype=torch.long), torch.tensor(decoder_seq, dtype=torch.long)


class CustomPositionalEmbedding(nn.Module):
    """Custom positional encoding with learnable components and sinusoidal patterns"""
    
    def __init__(self, d_model, max_seq_length=100):
        super(CustomPositionalEmbedding, self).__init__()
        self.d_model = d_model
        
        # Create base positional encoding matrix
        encoding_matrix = torch.zeros(max_seq_length, d_model)
        position_indices = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute frequency divisors for sinusoidal encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # Apply sinusoidal transformations
        encoding_matrix[:, 0::2] = torch.sin(position_indices * div_term)
        if d_model % 2 == 1:
            encoding_matrix[:, 1::2] = torch.cos(position_indices * div_term[:-1])
        else:
            encoding_matrix[:, 1::2] = torch.cos(position_indices * div_term)
        
        # Register as buffer (not trainable parameter)
        self.register_buffer('encoding_matrix', encoding_matrix.unsqueeze(0))
        
        # Add learnable bias term for adaptation
        self.adaptive_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, embeddings):
        seq_len = embeddings.size(1)
        # Add positional information with adaptive component
        pos_encoding = self.encoding_matrix[:, :seq_len, :] + self.adaptive_bias
        return embeddings + pos_encoding


class EnhancedTransformerPredictor(nn.Module):
    """Enhanced transformer architecture for character prediction tasks"""
    
    def __init__(self, vocabulary_size=28, embed_dimensions=128, attention_heads=4, 
                 transformer_layers=6, ff_dimensions=512, sequence_max=100, dropout_rate=0.1):
        super(EnhancedTransformerPredictor, self).__init__()
        
        # Store parameters for model saving/loading
        self.vocabulary_size = vocabulary_size
        self.embed_dimensions = embed_dimensions
        self.attention_heads = attention_heads
        self.transformer_layers = transformer_layers
        self.ff_dimensions = ff_dimensions
        self.sequence_max = sequence_max
        self.dropout_rate = dropout_rate
        
        # Embedding layer with scaled initialization
        self.char_embeddings = nn.Embedding(vocabulary_size, embed_dimensions)
        nn.init.normal_(self.char_embeddings.weight, mean=0, std=0.1)
        
        # Custom positional encoding
        self.position_encoder = CustomPositionalEmbedding(embed_dimensions, sequence_max)
        
        # Input preprocessing
        self.input_dropout = nn.Dropout(dropout_rate)
        self.input_layernorm = nn.LayerNorm(embed_dimensions)
        
        # Core transformer architecture
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dimensions,
            nhead=attention_heads,
            dim_feedforward=ff_dimensions,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'  # Different activation function
        )
        
        self.sequence_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=transformer_layers,
            norm=nn.LayerNorm(embed_dimensions)
        )
        
        # Output processing layers
        self.pre_output_norm = nn.LayerNorm(embed_dimensions)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dimensions, embed_dimensions // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(embed_dimensions // 2, vocabulary_size)
        )
        
        # Initialize output projection
        for layer in self.output_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, input_tokens, attention_mask=None):
        # Token embedding with scaling
        embedded_tokens = self.char_embeddings(input_tokens)
        scaled_embeddings = embedded_tokens * math.sqrt(self.char_embeddings.embedding_dim)
        
        # Add positional information
        position_enhanced = self.position_encoder(scaled_embeddings)
        
        # Apply input normalization and dropout
        processed_input = self.input_layernorm(position_enhanced)
        processed_input = self.input_dropout(processed_input)
        
        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = self._generate_padding_mask(input_tokens)
        
        # Pass through transformer encoder
        encoded_sequence = self.sequence_encoder(
            processed_input, 
            src_key_padding_mask=attention_mask
        )
        
        # Final processing and projection
        normalized_output = self.pre_output_norm(encoded_sequence)
        prediction_logits = self.output_projection(normalized_output)
        
        return prediction_logits
    
    def _generate_padding_mask(self, input_tokens):
        """Generate attention mask to ignore padding tokens"""
        # Assuming token 0 is the padding token
        return (input_tokens == 0)


def save_model_with_config(model, filepath):
    """Save model with its configuration for proper loading"""
    model_data = {
        'state_dict': model.state_dict(),
        'config': {
            'vocabulary_size': model.vocabulary_size,
            'embed_dimensions': model.embed_dimensions,
            'attention_heads': model.attention_heads,
            'transformer_layers': model.transformer_layers,
            'ff_dimensions': model.ff_dimensions,
            'sequence_max': model.sequence_max,
            'dropout_rate': model.dropout_rate
        }
    }
    torch.save(model_data, filepath)


def load_model_with_config(filepath, device='cpu'):
    """Load model with proper configuration"""
    model_data = torch.load(filepath, map_location=device, weights_only=False)
    
    # Create model with saved configuration
    model = EnhancedTransformerPredictor(**model_data['config'])
    model.load_state_dict(model_data['state_dict'])
    model.to(device)
    model.eval()
    
    return model


def build_specialized_model(word_collection, obfuscation_level, output_path, 
                          training_epochs=100, mini_batch_size=128, optimizer_lr=0.0001, 
                          seq_length=15, compute_device='cpu', patience=10, min_delta=0.001,
                          validation_split=0.2):
    """Build and train a specialized character prediction model with early stopping"""
    
    print(f"Building specialized model: {output_path}")
    
    # Prepare dataset
    full_dataset = WordSequenceDataset(word_collection, obfuscation_rate=obfuscation_level, 
                                     max_seq_len=seq_length)
    print(f"Dataset prepared with {len(full_dataset)} samples")
    
    # Create train/validation split
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    training_data, validation_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Data loaders
    train_loader = DataLoader(training_data, batch_size=mini_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(validation_data, batch_size=mini_batch_size, shuffle=False, drop_last=False)
    
    print(f"Training samples: {len(training_data)}, Validation samples: {len(validation_data)}")
    
    # Initialize model
    neural_model = EnhancedTransformerPredictor(
        vocabulary_size=len(full_dataset.token_vocab),
        embed_dimensions=128,
        attention_heads=4,
        transformer_layers=6,
        ff_dimensions=512,
        sequence_max=25,  # Slightly larger buffer
        dropout_rate=0.15
    ).to(compute_device)
    
    # Optimization setup
    model_optimizer = optim.AdamW(neural_model.parameters(), lr=optimizer_lr, 
                                 weight_decay=1e-4, betas=(0.9, 0.999))
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        model_optimizer, T_0=20, T_mult=2, eta_min=optimizer_lr * 0.01
    )
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    print("Commencing training process...")
    
    for epoch_num in range(training_epochs):
        # Training phase
        neural_model.train()
        epoch_train_losses = []
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(compute_device)
            target_batch = target_batch.to(compute_device)
            
            model_optimizer.zero_grad()
            
            # Forward pass
            prediction_logits = neural_model(input_batch)
            
            # Compute loss
            batch_loss = loss_function(
                prediction_logits.view(-1, prediction_logits.size(-1)), 
                target_batch.view(-1)
            )
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(neural_model.parameters(), max_norm=1.0)
            
            model_optimizer.step()
            epoch_train_losses.append(batch_loss.item())
        
        # Learning rate scheduling
        scheduler.step()
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        
        # Validation phase
        neural_model.eval()
        epoch_val_losses = []
        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch = input_batch.to(compute_device)
                target_batch = target_batch.to(compute_device)
                
                prediction_logits = neural_model(input_batch)
                val_loss = loss_function(
                    prediction_logits.view(-1, prediction_logits.size(-1)), 
                    target_batch.view(-1)
                )
                epoch_val_losses.append(val_loss.item())
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model_state = neural_model.state_dict().copy()
            print(f"Epoch {epoch_num+1}: New best validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Periodic logging
        if (epoch_num + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch_num+1}/{training_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}, No improvement: {epochs_without_improvement}")
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch_num+1}")
            print(f"Best validation loss achieved: {best_val_loss:.4f}")
            break
    
    # Restore best model
    if best_model_state is not None:
        neural_model.load_state_dict(best_model_state)
        print("Restored model to best validation state")
    
    # Save model with configuration
    save_model_with_config(neural_model, output_path)
    print(f"Model saved to: {output_path}")
    
    return neural_model, best_val_loss


def initialize_model_suite():
    """Initialize the complete suite of neural models with different configurations"""
    
    try:
        with open("words_250000_train.txt", "r") as file_handle:
            word_collection = file_handle.read().strip().split()
        word_collection = [word.lower() for word in word_collection if word.isalpha()]
        print(f"Loaded vocabulary: {len(word_collection)} words")
    except FileNotFoundError:
        print("Training vocabulary file 'words_250000_train.txt' not found")
        return False
    
    processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {processing_device}")
    
    # Model configuration specifications
    model_specifications = {
        "intensive_model.pt": {
            "rate": 0.6, 
            "desc": "High obfuscation model",
            "epochs": 200,  
            "patience": 15, 
            "min_delta": 0.001,
            "seq_length": 15
        },
        "balanced_model.pt": {
            "rate": 0.35, 
            "desc": "Balanced obfuscation model",
            "epochs": 150,
            "patience": 12,
            "min_delta": 0.001,
            "seq_length": 15
        },
        "compact_model.pt": {
            "rate": 0.25, 
            "desc": "Compact word model",
            "epochs": 200,  
            "patience": 10,
            "min_delta": 0.0005, 
            "seq_length": 7
        }
    }
    
    # Check if models already exist
    models_ready = all(os.path.exists(path) for path in model_specifications.keys())
    
    if not models_ready:
        print("Training neural model suite...")
        training_results = {}
        
        for model_path, config in model_specifications.items():
            if not os.path.exists(model_path):
                print(f"\n{'='*60}")
                print(f"Training: {config['desc']}")
                print(f"Configuration: {config}")
                print(f"{'='*60}")
                
                # Handle compact model with filtered vocabulary
                if "compact" in model_path:
                    filtered_words = [w for w in word_collection if len(w) <= 7]
                    print(f"Using filtered vocabulary: {len(filtered_words)} words")
                    
                    trained_model, best_val_loss = build_specialized_model(
                        filtered_words,
                        obfuscation_level=config['rate'],
                        output_path=model_path,
                        training_epochs=config['epochs'],
                        mini_batch_size=128,
                        optimizer_lr=0.0001,
                        seq_length=config['seq_length'],
                        compute_device=processing_device,
                        patience=config['patience'],
                        min_delta=config['min_delta'],
                        validation_split=0.2
                    )
                else:
                    print(f"Using full vocabulary: {len(word_collection)} words")
                    
                    trained_model, best_val_loss = build_specialized_model(
                        word_collection,
                        obfuscation_level=config['rate'],
                        output_path=model_path,
                        training_epochs=config['epochs'],
                        mini_batch_size=128,
                        optimizer_lr=0.0001,
                        seq_length=config['seq_length'],
                        compute_device=processing_device,
                        patience=config['patience'],
                        min_delta=config['min_delta'],
                        validation_split=0.2
                    )
                
                training_results[model_path] = {
                    'best_val_loss': best_val_loss,
                    'config': config
                }
                
                print(f"✓ {config['desc']} training completed")
                print(f"  Final validation loss: {best_val_loss:.4f}")
                
            else:
                print(f"✓ {model_path} already exists")
        
        # Training summary
        if training_results:
            print(f"\n{'='*60}")
            print("TRAINING RESULTS SUMMARY")
            print(f"{'='*60}")
            for model_path, results in training_results.items():
                print(f"{results['config']['desc']}: {results['best_val_loss']:.4f}")
            print(f"{'='*60}")
            
    else:
        print("All models already exist - skipping training")
    
    # Final verification
    missing_models = [path for path in model_specifications.keys() if not os.path.exists(path)]
    if missing_models:
        print(f"\n⚠ Warning: Missing models: {missing_models}")
        return False
    
    print(f"\n✓ Model suite initialization complete!")
    print(f"Available models: {list(model_specifications.keys())}")
    return True


class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)


class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.attempted_chars = set()  
        
        # Load and process dictionary
        full_dictionary_location = "words_250000_train.txt"
        self.word_database = self.build_dictionary(full_dictionary_location)        
        self.active_vocabulary = self.word_database.copy()  
        
        self.frequency_analysis = self._analyze_letter_distribution()
        self.pattern_database = self._construct_pattern_library()
        
        self.processing_unit = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.processing_unit}")
        
        self.char_universe = ['<NULL>', '<HIDDEN>'] + list(string.ascii_lowercase)
        self.char_to_num = {char: idx for idx, char in enumerate(self.char_universe)}
        self.num_to_char = {idx: char for char, idx in self.char_to_num.items()}
        
        # Initialize neural model ensemble
        self._setup_neural_ensemble()
        
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com']

        data = {link: 0 for link in links}

        for link in links:
            requests.get(link)
            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def build_dictionary(self, dictionary_file_location):
        """Load and preprocess word dictionary"""
        try:
            with open(dictionary_file_location, "r") as f:
                words = f.read().strip().split()
            return [word.lower() for word in words if word.isalpha()]
        except FileNotFoundError:
            print(f"Dictionary file {dictionary_file_location} not found")
            return []

    def _analyze_letter_distribution(self):
        """Analyze character frequency across dictionary"""
        combined_text = ''.join(self.word_database)
        return collections.Counter(combined_text).most_common()

    def _construct_pattern_library(self):
        """Build pattern matching library for substrings"""
        max_length = max(len(word) for word in self.word_database) if self.word_database else 30
        pattern_lib = {size: [] for size in range(3, min(max_length, 30) + 1)}
        
        for size in range(3, min(max_length, 30) + 1):
            for word in self.word_database:
                if len(word) >= size:
                    for start_pos in range(len(word) - size + 1):
                        pattern_lib[size].append(word[start_pos:start_pos + size])
        
        return pattern_lib

    def _setup_neural_ensemble(self):
        """Initialize ensemble of neural prediction models"""
        model_registry = {
            'intensive_predictor': 'intensive_model.pt',
            'balanced_predictor': 'balanced_model.pt', 
            'compact_predictor': 'compact_model.pt'
        }
        
        for predictor_name, model_file in model_registry.items():
            try:
                # Use the new loading function
                predictor = load_model_with_config(model_file, self.processing_unit)
                setattr(self, predictor_name, predictor)
                print(f"✓ {model_file} loaded successfully")
                
            except FileNotFoundError:
                print(f"⚠ Warning: {model_file} not found. {predictor_name} disabled.")
                setattr(self, predictor_name, None)
            except Exception as e:
                print(f"⚠ Warning: Could not load {model_file}. {predictor_name} disabled. Error: {e}")
                setattr(self, predictor_name, None)

    def _extract_char_statistics(self, word_subset):
        """Extract character statistics from word subset"""
        char_stats = collections.Counter()
        for word in word_subset:
            unique_letters = set(word)
            for letter in unique_letters:
                char_stats[letter] += 1
        return char_stats

    def _apply_pattern_filter(self, target_pattern):
        """Filter vocabulary using regex pattern matching"""
        pattern_regex = "^" + target_pattern + "$"
        matching_words = []
        
        for word in self.active_vocabulary:
            if len(word) == len(target_pattern) and re.fullmatch(pattern_regex, word):
                matching_words.append(word)
        
        return matching_words

    def _find_pattern_matches(self, search_pattern):
        """Find character frequencies using pattern matching"""
        pattern_len = len(search_pattern)
        if pattern_len in self.pattern_database:
            relevant_patterns = []
            for substring in self.pattern_database[pattern_len]:
                if re.fullmatch("^" + search_pattern + "$", substring):
                    relevant_patterns.append(substring)
            return self._extract_char_statistics(relevant_patterns)
        return collections.Counter()

    def _query_neural_predictor(self, predictor, word_state):
        """Query neural model for character predictions"""
        if predictor is None:
            return {char: 1.0/26 for char in string.ascii_lowercase}
        
        # Transform pattern for neural input
        neural_input = []
        for symbol in word_state:
            if symbol == '.':
                neural_input.append('<HIDDEN>')
            elif symbol in self.char_universe:
                neural_input.append(symbol)
            else:
                neural_input.append('<NULL>')
        
        # Convert to tensor format
        token_sequence = [self.char_to_num.get(char, self.char_to_num['<NULL>']) 
                         for char in neural_input]
        
        # Handle sequence length limits
        max_model_len = predictor.position_encoder.encoding_matrix.size(1)
        if len(token_sequence) > max_model_len:
            token_sequence = token_sequence[:max_model_len]
        
        input_tensor = torch.tensor([token_sequence], dtype=torch.long, 
                                  device=self.processing_unit)
        
        with torch.no_grad():
            prediction_logits = predictor(input_tensor)
            
            # Identify positions needing prediction
            unknown_positions = [i for i, char in enumerate(neural_input) if char == '<HIDDEN>']
            
            # Calculate character probabilities
            char_probabilities = collections.Counter()
            alphabet_indices = range(2, 28)  
            
            if unknown_positions:
                for pos in unknown_positions:
                    if pos < prediction_logits.size(1):
                        pos_logits = prediction_logits[0, pos]
                        pos_probs = torch.softmax(pos_logits, dim=0)
                        
                        for char_idx in alphabet_indices:
                            if char_idx < len(self.num_to_char):
                                character = self.num_to_char[char_idx]
                                char_probabilities[character] += pos_probs[char_idx].item()
                
                # Normalize across positions
                for character in char_probabilities:
                    char_probabilities[character] /= len(unknown_positions)
            else:
                # Fallback to uniform distribution
                for character in string.ascii_lowercase:
                    char_probabilities[character] = 1.0 / 26
        
        return dict(char_probabilities)

    def _compute_vowel_density(self, known_chars):
        """Calculate vowel density in known characters"""
        if not known_chars:
            return 0.0
        vowel_set = set('aeiou')
        vowel_count = sum(1 for char in known_chars if char in vowel_set)
        return vowel_count / len(known_chars)

    def _determine_strategy(self, word_length, unknown_ratio):
        """Determine prediction strategy based on word characteristics"""
        
        strategy_map = {
            'short_intensive': {
                'condition': lambda l, r: l <= 7 and r > 0.7,
                'predictors': ['compact_predictor', 'intensive_predictor'],
                'weights': {'baseline': 0.3, 'compact_predictor': 0.4, 'intensive_predictor': 0.3}
            },
            'short_balanced': {
                'condition': lambda l, r: l <= 7 and 0.4 < r <= 0.7,
                'predictors': ['compact_predictor', 'balanced_predictor'],
                'weights': {'baseline': 0.4, 'compact_predictor': 0.3, 'balanced_predictor': 0.3}
            },
            'short_conservative': {
                'condition': lambda l, r: l <= 7 and r <= 0.4,
                'predictors': ['compact_predictor'],
                'weights': {'baseline': 0.6, 'compact_predictor': 0.4}
            },
            'long_intensive': {
                'condition': lambda l, r: l > 7 and r > 0.7,
                'predictors': ['intensive_predictor', 'balanced_predictor'],
                'weights': {'baseline': 0.3, 'intensive_predictor': 0.4, 'balanced_predictor': 0.3}
            },
            'long_balanced': {
                'condition': lambda l, r: l > 7 and 0.4 < r <= 0.7,
                'predictors': ['balanced_predictor'],
                'weights': {'baseline': 0.5, 'balanced_predictor': 0.5}
            },
            'long_conservative': {
                'condition': lambda l, r: l > 7 and r <= 0.4,
                'predictors': ['balanced_predictor'],
                'weights': {'baseline': 0.7, 'balanced_predictor': 0.3}
            }
        }
        
        # Find matching strategy
        for strategy_name, config in strategy_map.items():
            if config['condition'](word_length, unknown_ratio):
                return {
                    'predictors': config['predictors'],
                    'weights': config['weights']
                }
        
        # Default fallback strategy
        return {
            'predictors': ['balanced_predictor'],
            'weights': {'baseline': 0.6, 'balanced_predictor': 0.4}
        }
        
    def guess(self, word):
        """Primary guess method using ensemble approach"""
        
        # Parse and analyze word state
        clean_word = word[::2].replace("_", ".")
        word_len = len(clean_word)
        mystery_count = clean_word.count('.')
        mystery_ratio = mystery_count / word_len if word_len > 0 else 0.0
        
        # Update active vocabulary
        self.active_vocabulary = self._apply_pattern_filter(clean_word)
        
        # Compute heuristic scores
        if self.active_vocabulary:
            heuristic_stats = self._extract_char_statistics(self.active_vocabulary)
        else:
            heuristic_stats = self._find_pattern_matches(clean_word)
            if not heuristic_stats:
                heuristic_stats = collections.Counter(dict(self.frequency_analysis))
        
        # Normalize heuristic scores
        total_heuristic = sum(heuristic_stats.values())
        baseline_scores = {}
        if total_heuristic > 0:
            for char, frequency in heuristic_stats.items():
                baseline_scores[char] = frequency / total_heuristic
        
        # Neural ensemble predictions
        neural_outputs = {}
        
        # Strategy mapping based on word characteristics
        strategy_config = self._determine_strategy(word_len, mystery_ratio)
        
        # Collect neural predictions based on strategy
        for predictor_type in strategy_config['predictors']:
            if hasattr(self, predictor_type) and getattr(self, predictor_type) is not None:
                neural_outputs[predictor_type] = self._query_neural_predictor(
                    getattr(self, predictor_type), clean_word)
        
        # Apply weighting scheme
        weight_distribution = strategy_config['weights']
        
        # Normalize weights
        total_weights = sum(weight_distribution.values())
        if total_weights > 0:
            for component in weight_distribution:
                weight_distribution[component] /= total_weights
        
        # Ensemble combination
        final_scores = {}
        for character in string.ascii_lowercase:
            score = weight_distribution.get('baseline', 0.0) * baseline_scores.get(character, 0.0)
            
            for predictor_name, predictions in neural_outputs.items():
                if predictor_name in weight_distribution:
                    score += weight_distribution[predictor_name] * predictions.get(character, 0.0)
            
            final_scores[character] = score
        
        # Apply linguistic heuristics
        revealed_chars = [c for c in clean_word if c != '.']
        vowel_density = self._compute_vowel_density(revealed_chars)
        
        # Rank characters by final scores
        ranked_chars = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Selection with vowel bias consideration
        if vowel_density > 0.6 and mystery_ratio < 0.5:
            consonant_set = set(string.ascii_lowercase) - set('aeiou')
            for character, score in ranked_chars:
                if character not in self.attempted_chars and character in consonant_set:
                    return character
        
        # Standard selection
        for character, score in ranked_chars:
            if character not in self.attempted_chars:
                return character

        return 'a'

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
                
    def start_game(self, practice=True, verbose=True):
        # reset attempted characters to empty set and current plausible dictionary to the full dictionary
        self.attempted_chars = set()  # Changed from guessed_letters
        self.active_vocabulary = self.word_database.copy()  # Changed from current_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to attempted characters field in hangman object
                self.attempted_chars.add(guess_letter)  # Changed from append to add
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result



def main():
    """Main function to train models and run the hangman game"""
    
    # Train models if needed
    if not initialize_model_suite():
        print("Failed to initialize models. Exiting.")
        return
    
    print("\nInitializing Hangman API...")
    api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)
    
    api.start_game(practice=1,verbose=True)
    [total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
    practice_success_rate = total_practice_successes / total_practice_runs
    print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))


    # ## Playing recorded games:
    # Please finalize your code prior to running the cell below. Once this code executes once successfully your submission will be finalized. Our system will not allow you to rerun any additional games.
    # 
    # Please note that it is expected that after you successfully run this block of code that subsequent runs will result in the error message "Your account has been deactivated".
    # 
    # Once you've run this section of the code your submission is complete. Please send us your source code via email.

    # In[ ]:


    # for i in range(1000):
    #     print('Playing ', i, ' th game')
    #     # Uncomment the following line to execute your final runs. Do not do this until you are satisfied with your submission
    #     api.start_game(practice=0,verbose=False)
        
    #     # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
    #     time.sleep(0.5)

    # ## To check your game statistics
    # 1. Simply use "my_status" method.
    # 2. Returns your total number of games, and number of wins.


    [total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
    success_rate = total_recorded_successes/total_recorded_runs
    print('overall success rate = %.3f' % success_rate)



if __name__ == "__main__":
    main()