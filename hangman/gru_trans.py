# -*- coding: utf-8 -*-
"""
Optimized PyTorch Neural Network Hangman Solver
Target: 70%+ accuracy using GRU+Transformer architecture with hyperparameter optimization
"""

import collections
import json
import pickle
import random
import re
import secrets
import string
import time
import warnings
from collections import Counter
from typing import Dict, List, Optional, Tuple
import itertools

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset

warnings.filterwarnings('ignore')

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class PositionalEncoding(nn.Module):
    """Enhanced Positional encoding for transformer"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class OptimizedGRUTransformerModel(nn.Module):
    """Optimized GRU + Transformer model for Hangman"""
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int, 
                 gru_hidden: int = 256, num_heads: int = 8, num_layers: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        
        # Enhanced embedding with dropout
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=27)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(embedding_dim, gru_hidden, 
                         batch_first=True, bidirectional=True, 
                         dropout=dropout if num_layers > 1 else 0,
                         num_layers=2)
        
        # Project GRU output to transformer dimension
        gru_output_dim = gru_hidden * 2  # bidirectional
        self.gru_projection = nn.Linear(gru_output_dim, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',  # GELU activation performs better
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embedding_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embedding_dim, 26)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
        
    def forward(self, x):
        # Create padding mask
        padding_mask = (x == 27)  # 27 is padding token
        
        # Embedding with dropout
        embedded = self.embedding(x)
        embedded = self.embedding_dropout(embedded)
        
        # GRU processing
        gru_out, _ = self.gru(embedded)
        gru_out = self.gru_projection(gru_out)
        
        # Residual connection: combine embedding and GRU output
        combined = embedded + gru_out
        
        # Positional encoding
        combined = self.pos_encoding(combined.transpose(0, 1)).transpose(0, 1)
        
        # Transformer processing
        transformer_out = self.transformer(combined, src_key_padding_mask=padding_mask)
        
        # Classification
        transformer_out = transformer_out.transpose(1, 2)  # (batch, features, seq_len)
        output = self.classifier(transformer_out)
        
        return F.softmax(output, dim=1)

class OptimizedHangmanSolver:
    """Optimized Hangman solver with improved training data generation"""
    
    def __init__(self, max_word_length=30, embedding_dim=128):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.alphabet)}
        self.char_to_idx['_'] = 0  # Unknown character
        self.char_to_idx['<PAD>'] = 27  # Padding
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        self.max_word_length = max_word_length
        self.embedding_dim = embedding_dim
        self.vocab_size = len(self.char_to_idx)
        
        self.model = None
        self.device = device
        
        # Letter frequency in English (optimized order)
        self.freq_order = "etaoinshrdlcumwfgypbvkjxqz"
        
        # Common patterns and n-grams
        self.load_patterns()
        
    def load_patterns(self):
        """Load common English patterns"""
        self.common_bigrams = [
            'th', 'he', 'in', 'er', 'an', 're', 'nd', 'on', 'en', 'at',
            'ou', 'ed', 'ha', 'to', 'or', 'it', 'is', 'hi', 'es', 'ng'
        ]
        
        self.common_trigrams = [
            'the', 'and', 'ing', 'ion', 'tio', 'ent', 'ati', 'for', 'her', 'ter'
        ]
        
        self.common_endings = [
            'ing', 'ion', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less'
        ]
        
        self.common_beginnings = [
            'pre', 'pro', 'anti', 'dis', 'mis', 'over', 'under', 'out', 'up', 're'
        ]
    
    def build_enhanced_dictionary(self, dictionary_file_location):
        """Build enhanced dictionary with pattern analysis"""
        print("Building enhanced dictionary...")
        
        with open(dictionary_file_location, "r") as text_file:
            words = text_file.read().splitlines()
        
        # Filter and enhance words
        enhanced_words = []
        word_patterns = {}
        
        for word in words:
            word = word.lower().strip()
            if 2 <= len(word) <= 25 and word.isalpha():
                enhanced_words.append(word)
                
                # Analyze patterns
                word_patterns[word] = {
                    'length': len(word),
                    'unique_chars': len(set(word)),
                    'char_freq': Counter(word),
                    'bigrams': [word[i:i+2] for i in range(len(word)-1)],
                    'trigrams': [word[i:i+3] for i in range(len(word)-2)],
                    'vowel_positions': [i for i, c in enumerate(word) if c in 'aeiou'],
                    'consonant_clusters': self._find_consonant_clusters(word)
                }
        
        self.word_patterns = word_patterns
        print(f"Enhanced dictionary with {len(enhanced_words)} words")
        return enhanced_words
    
    def _find_consonant_clusters(self, word):
        """Find consonant clusters in word"""
        clusters = []
        vowels = set('aeiou')
        current_cluster = []
        
        for char in word:
            if char not in vowels:
                current_cluster.append(char)
            else:
                if len(current_cluster) >= 2:
                    clusters.append(''.join(current_cluster))
                current_cluster = []
        
        if len(current_cluster) >= 2:
            clusters.append(''.join(current_cluster))
        
        return clusters
    
    def encode_word(self, word):
        """Enhanced word encoding"""
        word = word.replace(' ', '').lower()
        encoded = []
        
        for char in word:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['_'])
        
        # Pad or truncate
        if len(encoded) < self.max_word_length:
            encoded.extend([self.char_to_idx['<PAD>']] * (self.max_word_length - len(encoded)))
        else:
            encoded = encoded[:self.max_word_length]
            
        return np.array(encoded)
    
    def create_smart_training_data(self, dictionary_words, samples_per_word=5):
        """Create smarter training data with strategic masking"""
        print("Creating enhanced training data...")
        
        X_train = []
        y_train = []
        
        for word in dictionary_words:
            word = word.lower().strip()
            if len(word) < 2 or len(word) > 25:
                continue
            
            word_chars = list(set(word))
            
            # Generate multiple strategic scenarios per word
            for scenario in range(samples_per_word):
                
                # Scenario 1: Early game (high frequency letters revealed)
                if scenario == 0:
                    revealed = [c for c in word_chars if c in 'etaoin'[:3]]
                    
                # Scenario 2: Mid game (some vowels and consonants)
                elif scenario == 1:
                    revealed = random.sample(word_chars, min(len(word_chars)//2, 4))
                    
                # Scenario 3: Pattern-based (reveal common patterns)
                elif scenario == 2:
                    revealed = []
                    for bigram in self.common_bigrams[:5]:
                        if bigram in word:
                            revealed.extend(list(bigram))
                    revealed = list(set(revealed))[:3]
                    
                # Scenario 4: End game (most letters revealed)
                elif scenario == 3:
                    num_revealed = max(1, len(word_chars) - 2)
                    revealed = random.sample(word_chars, num_revealed)
                    
                # Scenario 5: Random strategic
                else:
                    num_revealed = random.randint(1, max(1, len(word_chars)-1))
                    revealed = random.sample(word_chars, num_revealed)
                
                # Create masked word
                masked_word = ''.join(['_' if c not in revealed else c for c in word])
                
                # Find best next letter
                remaining = [c for c in word_chars if c not in revealed]
                if remaining:
                    # Choose based on frequency in remaining positions
                    target_letter = self._choose_best_target(word, masked_word, remaining)
                    
                    # Create target vector
                    target_vector = np.zeros(26)
                    target_idx = ord(target_letter) - ord('a')
                    target_vector[target_idx] = 1
                    
                    X_train.append(self.encode_word(masked_word))
                    y_train.append(target_vector)
        
        print(f"Created {len(X_train)} training examples")
        return np.array(X_train), np.array(y_train)
    
    def _choose_best_target(self, original_word, masked_word, remaining_letters):
        """Choose the best target letter based on multiple criteria"""
        scores = {}
        
        for letter in remaining_letters:
            score = 0
            
            # Frequency in word
            score += original_word.count(letter) * 2
            
            # English frequency
            if letter in self.freq_order:
                score += (26 - self.freq_order.index(letter)) * 0.1
            
            # Pattern completion bonus
            for i, char in enumerate(masked_word):
                if char == '_':
                    # Check if this letter completes common patterns
                    if i > 0 and masked_word[i-1] != '_':
                        bigram = masked_word[i-1] + letter
                        if bigram in self.common_bigrams:
                            score += 1
                    
                    if i < len(masked_word) - 1 and masked_word[i+1] != '_':
                        bigram = letter + masked_word[i+1]
                        if bigram in self.common_bigrams:
                            score += 1
            
            scores[letter] = score
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def build_model(self, **kwargs):
        """Build optimized model with hyperparameters"""
        return OptimizedGRUTransformerModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_word_length,
            **kwargs
        ).to(self.device)
    
    def train_model_with_params(self, model, train_loader, val_loader, 
                               lr=0.001, epochs=200, patience=15):
        """Train model with specific hyperparameters"""
        
        # Enhanced optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        
        # Label smoothing loss
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                _, targets = torch.max(batch_y.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, targets = torch.max(batch_y.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')
            
            # Early stopping with patience
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch}, best val acc: {best_val_acc:.4f}')
                    break
        
        return best_val_acc
    
    def grid_search_hyperparameters(self, dictionary_words, sample_size=5000):
        """Perform grid search on a subset of data"""
        print("Starting hyperparameter grid search...")
        
        # Create subset for faster grid search
        sample_words = random.sample(dictionary_words, min(sample_size, len(dictionary_words)))
        X, y = self.create_smart_training_data(sample_words, samples_per_word=2)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        # Define parameter grid
        param_grid = {
            # 'embedding_dim': [128, 256],
            'gru_hidden': [128, 256],
            'num_heads': [4, 8],
            'num_layers': [2, 3],
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.001, 0.002, 0.0005],
            'batch_size': [64, 128]
        }
        
        best_params = None
        best_score = 0
        
        # Grid search with early stopping for efficiency
        grid = list(ParameterGrid(param_grid))
        random.shuffle(grid)  # Randomize order
        
        for i, params in enumerate(grid[:20]):  # Test top 20 combinations
            print(f"\nTesting combination {i+1}/20: {params}")
            
            try:
                # Create data loaders
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
                
                # Build and train model
                model_params = {k: v for k, v in params.items() if k not in ['lr', 'batch_size']}
                model = self.build_model(**model_params)
                
                score = self.train_model_with_params(
                    model, train_loader, val_loader, 
                    lr=params['lr'], epochs=50, patience=8
                )
                
                print(f"Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"New best score: {best_score:.4f}")
                
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
        
        print(f"\nBest parameters: {best_params}")
        print(f"Best validation accuracy: {best_score:.4f}")
        
        return best_params
    
    def train_optimized_model(self, dictionary_words, best_params=None):
        """Train the final optimized model"""
        print("Training optimized model...")
        
        # Use best params or defaults
        if best_params is None:
            best_params = {
                # 'embedding_dim': 256,
                'gru_hidden': 256, 
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.2,
                'lr': 0.001,
                'batch_size': 128
            }
        
        # Create full training data
        X, y = self.create_smart_training_data(dictionary_words, samples_per_word=6)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.15, random_state=42
        )
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
        
        # Build model with best params
        model_params = {k: v for k, v in best_params.items() if k not in ['lr', 'batch_size']}
        self.model = self.build_model(**model_params)
        
        # Train model
        final_accuracy = self.train_model_with_params(
            self.model, train_loader, val_loader,
            lr=best_params['lr'], epochs=300, patience=20
        )
        
        print(f"Final model validation accuracy: {final_accuracy:.4f}")
        return final_accuracy
    
    def predict_letter(self, word, guessed_letters):
        """Predict next letter using trained model"""
        if self.model is None:
            return self.frequency_fallback(word, guessed_letters)
        
        # Encode current word state
        encoded_word = torch.tensor(self.encode_word(word), dtype=torch.long).unsqueeze(0).to(self.device)
        
        try:
            self.model.eval()
            with torch.no_grad():
                pred = self.model(encoded_word).cpu().numpy()[0]
                
                # Zero out guessed letters
                for letter in guessed_letters:
                    if letter in self.alphabet:
                        idx = ord(letter) - ord('a')
                        pred[idx] = 0
                
                # Get prediction
                if pred.sum() > 0:
                    # Apply strategic filtering
                    pred = self._apply_strategic_filtering(word, pred, guessed_letters)
                    
                    best_idx = np.argmax(pred)
                    predicted_letter = chr(best_idx + ord('a'))
                    
                    if predicted_letter not in guessed_letters:
                        return predicted_letter
        except Exception as e:
            print(f"Prediction error: {e}")
        
        return self.frequency_fallback(word, guessed_letters)
    
    def _apply_strategic_filtering(self, word, predictions, guessed_letters):
        """Apply strategic filtering to predictions"""
        # Boost predictions for letters that complete common patterns
        for i, char in enumerate(word):
            if char == '_':
                for letter_idx, pred_score in enumerate(predictions):
                    letter = chr(letter_idx + ord('a'))
                    
                    if letter in guessed_letters:
                        continue
                    
                    # Check pattern completion
                    if i > 0 and word[i-1] != '_':
                        bigram = word[i-1] + letter
                        if bigram in self.common_bigrams:
                            predictions[letter_idx] *= 1.5
                    
                    if i < len(word) - 1 and word[i+1] != '_':
                        bigram = letter + word[i+1]
                        if bigram in self.common_bigrams:
                            predictions[letter_idx] *= 1.5
        
        return predictions
    
    def frequency_fallback(self, word, guessed_letters):
        """Enhanced frequency-based fallback"""
        # Start with frequency order
        for letter in self.freq_order:
            if letter not in guessed_letters:
                # Additional pattern-based checks
                if self._letter_fits_pattern(word, letter):
                    return letter
        
        # Final fallback
        for letter in self.alphabet:
            if letter not in guessed_letters:
                return letter
        
        return 'a'
    
    def _letter_fits_pattern(self, word, letter):
        """Check if letter fits common patterns"""
        # Simple pattern matching
        for i, char in enumerate(word):
            if char == '_':
                # Check adjacent characters for pattern matching
                if i > 0 and word[i-1] != '_':
                    if word[i-1] + letter in self.common_bigrams:
                        return True
                if i < len(word) - 1 and word[i+1] != '_':
                    if letter + word[i+1] in self.common_bigrams:
                        return True
        return True  # Default to True if no specific pattern found
    
    def save_model(self, filepath):
        """Save trained model and metadata"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'vocab_size': self.vocab_size,
                    'embedding_dim': self.embedding_dim,
                    'max_seq_len': self.max_word_length
                },
                'metadata': {
                    'char_to_idx': self.char_to_idx,
                    'max_word_length': self.max_word_length,
                    'embedding_dim': self.embedding_dim
                }
            }, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Load metadata
            self.char_to_idx = checkpoint['metadata']['char_to_idx']
            self.max_word_length = checkpoint['metadata']['max_word_length']
            self.embedding_dim = checkpoint['metadata']['embedding_dim']
            
            # Build and load model
            config = checkpoint['model_config']
            self.model = OptimizedGRUTransformerModel(**config).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print("Model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False



def build_dictionary(dictionary_file_location):
    """Build dictionary from file"""
    with open(dictionary_file_location, "r") as text_file:
        full_dictionary = text_file.read().splitlines()
    return full_dictionary


# Initialize solver
neural_solver = OptimizedHangmanSolver(embedding_dim=256)
# Training section (uncomment to train new models)

print("Loading dictionary...")
words = build_dictionary("words_250000_train.txt")
enhanced_words = neural_solver.build_enhanced_dictionary("words_250000_train.txt")

# Try to load existing model first
model_path = "optimized_hangman_model.pth"
if not neural_solver.load_model(model_path):
    print("No pre-trained model found. Starting training process...")
    
    # Step 1: Hyperparameter optimization (optional - comment out for faster training)
    print("Performing hyperparameter search...")
    best_params = neural_solver.grid_search_hyperparameters(enhanced_words, sample_size=3000)
    
    # Step 2: Train final model with best parameters
    print("Training final optimized model...")
    final_accuracy = neural_solver.train_optimized_model(enhanced_words, best_params)
    
    # Step 3: Save the trained model
    neural_solver.save_model(model_path)
    print(f"Model training completed with accuracy: {final_accuracy:.4f}")
else:
    print("Pre-trained model loaded successfully!")

# Alternative faster training (skip hyperparameter search)
# Uncomment this block and comment the above if you want faster training:
"""
if not neural_solver.load_model(model_path):
    print("Training model with default parameters...")
    final_accuracy = neural_solver.train_optimized_model(enhanced_words)
    neural_solver.save_model(model_path)
    print(f"Model training completed with accuracy: {final_accuracy:.4f}")
"""
class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        # Initialize neural solver
        self.solver = neural_solver

    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']
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

    def guess(self, word, verbose=False):
        """Make a guess using optimized neural network"""
        guess_letter = self.solver.predict_letter(word, self.guessed_letters)
        
        if verbose:
            print(f"Neural prediction: {guess_letter}")
        
        return guess_letter

    def build_dictionary(self, dictionary_file_location):
        return self.solver.build_enhanced_dictionary(dictionary_file_location)

    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        
        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(
                    game_id, tries_remains, word))
            
            while tries_remains > 0:
                guess_letter = self.guess(word, verbose)
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))

                try:
                    res = self.request("/guess_letter", {
                        "request": "guess_letter", 
                        "game_id": game_id, 
                        "letter": guess_letter
                    })
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e

                if verbose:
                    print("Server response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                
                if status == "success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status == "failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status == "ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return False

    def my_status(self):
        return self.request("/my_status", {})

    def request(self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        if self.access_token:
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
            print("STATUS:", response.status_code)
            print("CONTENT-TYPE:", response.headers.get("Content-Type"))
            print("BODY:", response.text)
            raise HangmanAPIError(f'Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result

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

# Main execution
if __name__ == "__main__":
    api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)

    # Practice runs
    print("Starting practice runs...")
    for i in range(100):
        print(f"Practice game {i+1}/100")
        api.start_game(practice=1, verbose=True)

    [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    practice_success_rate = total_practice_successes / total_practice_runs
    print('Run %d practice games out of an allotted 100,000. Practice success rate so far = %.3f' % 
          (total_practice_runs, practice_success_rate))

    # Official runs
    print("Starting official runs...")
    # for i in range(1000):
    #     print('Playing', i+1, 'th game')
    #     api.start_game(practice=0, verbose=False)
    #     time.sleep(0.5)

    [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    success_rate = total_recorded_successes / total_recorded_runs
    print('Overall success rate = %.3f' % success_rate)