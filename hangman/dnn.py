# -*- coding: utf-8 -*-
"""
Advanced Hangman AI using Deep Learning with Attention Mechanism
Completely different approach from the original GP-based solution
"""
import os
import json
import requests
import random
import string
import secrets
import time
import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Deep Learning Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import math

class HangmanDataset(Dataset):
    """Custom dataset for Hangman word patterns"""
    
    def __init__(self, patterns, targets, vocab_size=29, max_length=50):
        self.patterns = patterns
        self.targets = targets
        self.vocab_size = vocab_size  # 26 letters + underscore + padding + unknown
        self.max_length = max_length
        self.char_to_idx = {chr(i + ord('a')): i + 1 for i in range(26)}
        self.char_to_idx['_'] = 27
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 28
        
    def __len__(self):
        return len(self.patterns)
    
    def encode_pattern(self, pattern):
        """Encode pattern string to numerical sequence"""
        encoded = []
        for char in pattern.lower():
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['<UNK>'])
        
        # Pad to max_length
        if len(encoded) < self.max_length:
            encoded.extend([0] * (self.max_length - len(encoded)))
        else:
            encoded = encoded[:self.max_length]
            
        return torch.tensor(encoded, dtype=torch.long)
    
    def __getitem__(self, idx):
        pattern = self.patterns[idx]
        target = self.targets[idx]
        
        encoded_pattern = self.encode_pattern(pattern)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return encoded_pattern, target_tensor

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like architecture"""
    
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(attention_output)
        return output

class HangmanTransformer(nn.Module):
    """Advanced Hangman AI using Transformer architecture"""
    
    def __init__(self, vocab_size=29, d_model=256, n_heads=8, n_layers=6, 
                 max_length=50, dropout=0.1):
        super(HangmanTransformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=n_heads, 
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Additional attention layers
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        
        # Classification layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Multiple prediction heads for better performance
        self.classifier_1 = nn.Linear(d_model, 512)
        self.classifier_2 = nn.Linear(512, 256)
        self.classifier_3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 26)  # 26 letters
        
        # Additional context layers
        self.context_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def create_padding_mask(self, x):
        """Create mask for padding tokens"""
        return (x != 0).float().unsqueeze(-1)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        
        # Create padding mask
        padding_mask = self.create_padding_mask(x)
        
        # Embedding + Positional encoding
        embedded = self.embedding(x) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded.transpose(0, 1)).transpose(0, 1)
        
        # Apply transformer layers
        hidden = embedded
        for transformer_layer in self.transformer_layers:
            hidden = transformer_layer(hidden, src_key_padding_mask=(x == 0))
        
        # Self attention
        attended = self.self_attention(hidden, hidden, hidden)
        
        # Combine attended and original hidden states
        combined = hidden + attended
        combined = self.layer_norm(combined)
        
        # Apply masking to ignore padding tokens
        masked_hidden = combined * padding_mask
        
        # Global context via convolution
        conv_input = masked_hidden.transpose(1, 2)
        conv_output = F.relu(self.context_conv(conv_input))
        
        # Global average pooling
        pooled = self.global_pool(conv_output).squeeze(-1)
        
        # Also use max pooling for different perspective
        max_pooled, _ = torch.max(masked_hidden, dim=1)
        
        # Combine both pooling methods
        final_repr = pooled + max_pooled
        
        # Classification layers with residual connections
        x1 = F.relu(self.classifier_1(final_repr))
        x1 = self.dropout(x1)
        
        x2 = F.relu(self.classifier_2(x1))
        x2 = self.dropout(x2)
        
        x3 = F.relu(self.classifier_3(x2))
        x3 = self.dropout(x3)
        
        # Final output
        output = self.output_layer(x3)
        
        return output

class HangmanAI:
    """Main Hangman AI class using deep learning"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.char_to_idx = {chr(i + ord('a')): i for i in range(26)}
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Model parameters
        self.model = HangmanTransformer(
            vocab_size=29,
            d_model=256,
            n_heads=8,
            n_layers=6,
            max_length=50,
            dropout=0.1
        ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print("Model loaded successfully!")
        
        # Load dictionary
        self.dictionary = self.load_dictionary("words_250000_train.txt")
        
    def load_dictionary(self, filepath):
        """Load word dictionary"""
        try:
            with open(filepath, 'r') as f:
                words = [line.strip().lower() for line in f.readlines()]
            print(f"Loaded {len(words)} words from dictionary")
            return words
        except FileNotFoundError:
            print(f"Dictionary file {filepath} not found. Using sample words.")
            return ['python', 'machine', 'learning', 'artificial', 'intelligence']
    
    def generate_training_data(self, num_samples=1000000):
        """Generate comprehensive training data from dictionary with multiple strategies"""
        print("Generating extensive training data...")
        
        patterns = []
        targets = []
        word_lengths = {}
        
        # Categorize words by length for balanced sampling
        for word in self.dictionary:
            if len(word) <= 50:
                length = len(word)
                if length not in word_lengths:
                    word_lengths[length] = []
                word_lengths[length].append(word)
        
        print(f"Dictionary contains words of lengths: {sorted(word_lengths.keys())}")
        
        # Strategy 1: Random revelation (40% of samples)
        print("Generating random revelation samples...")
        for _ in range(int(num_samples * 0.4)):
            word = random.choice(self.dictionary)
            if len(word) > 50:
                continue
                
            # Varying revelation strategies
            num_revealed = random.randint(1, max(1, min(3, len(word) - 1)))
            revealed_positions = random.sample(range(len(word)), num_revealed)
            
            pattern = list('_' * len(word))
            for pos in revealed_positions:
                pattern[pos] = word[pos]
            
            pattern_str = ''.join(pattern)
            hidden_letters = [word[i] for i in range(len(word)) if i not in revealed_positions]
            
            if hidden_letters:
                # Choose target letter with weighted probability
                letter_counts = Counter(hidden_letters)
                total_count = sum(letter_counts.values())
                
                # Weighted selection - favor more frequent letters
                if random.random() < 0.7:  # 70% chance for most frequent
                    target_letter = letter_counts.most_common(1)[0][0]
                else:  # 30% chance for random selection
                    target_letter = random.choice(hidden_letters)
                
                target_idx = self.char_to_idx[target_letter]
                patterns.append(pattern_str)
                targets.append(target_idx)
        
        # Strategy 2: Progressive revelation (30% of samples)
        print("Generating progressive revelation samples...")
        for _ in range(int(num_samples * 0.3)):
            word = random.choice(self.dictionary)
            if len(word) > 50:
                continue
            
            # Simulate game progression - start with few letters, gradually reveal more
            max_revealed = random.randint(1, max(1, len(word) - 1))
            for num_revealed in range(1, max_revealed + 1):
                if num_revealed >= len(word):
                    break
                    
                # Favor revealing vowels and common consonants first
                vowels = set('aeiou')
                common_consonants = set('nrtlshdcf')
                
                revealed_positions = []
                available_positions = list(range(len(word)))
                
                # First reveal vowels
                for pos in available_positions[:]:
                    if len(revealed_positions) >= num_revealed:
                        break
                    if word[pos] in vowels:
                        revealed_positions.append(pos)
                        available_positions.remove(pos)
                
                # Then common consonants
                for pos in available_positions[:]:
                    if len(revealed_positions) >= num_revealed:
                        break
                    if word[pos] in common_consonants:
                        revealed_positions.append(pos)
                        available_positions.remove(pos)
                
                # Fill remaining with random
                while len(revealed_positions) < num_revealed and available_positions:
                    pos = random.choice(available_positions)
                    revealed_positions.append(pos)
                    available_positions.remove(pos)
                
                pattern = list('_' * len(word))
                for pos in revealed_positions:
                    pattern[pos] = word[pos]
                
                pattern_str = ''.join(pattern)
                hidden_letters = [word[i] for i in range(len(word)) if i not in revealed_positions]
                
                if hidden_letters:
                    target_letter = Counter(hidden_letters).most_common(1)[0][0]
                    target_idx = self.char_to_idx[target_letter]
                    patterns.append(pattern_str)
                    targets.append(target_idx)
        
        # Strategy 3: Difficult cases (20% of samples)
        print("Generating challenging samples...")
        for _ in range(int(num_samples * 0.2)):
            word = random.choice(self.dictionary)
            if len(word) > 50 or len(word) < 2:
                continue
            
            # Create challenging patterns - reveal very few letters
            num_revealed = random.randint(1, max(1, min(3, len(word) - 1)))
            
            # Favor revealing letters that appear multiple times
            letter_positions = {}
            for i, char in enumerate(word):
                if char not in letter_positions:
                    letter_positions[char] = []
                letter_positions[char].append(i)
            
            # Prefer letters that appear multiple times
            multi_occurrence_letters = [char for char, positions in letter_positions.items() 
                                      if len(positions) > 1]
            
            revealed_positions = []
            if multi_occurrence_letters and random.random() < 0.6:
                chosen_letter = random.choice(multi_occurrence_letters)
                revealed_positions.extend(letter_positions[chosen_letter])
            
            # Fill remaining reveals randomly
            available_positions = [i for i in range(len(word)) if i not in revealed_positions]
            while len(revealed_positions) < num_revealed and available_positions:
                pos = random.choice(available_positions)
                revealed_positions.append(pos)
                available_positions.remove(pos)
            
            pattern = list('_' * len(word))
            for pos in revealed_positions:
                pattern[pos] = word[pos]
            
            pattern_str = ''.join(pattern)
            hidden_letters = [word[i] for i in range(len(word)) if i not in revealed_positions]
            
            if hidden_letters:
                target_letter = Counter(hidden_letters).most_common(1)[0][0]
                target_idx = self.char_to_idx[target_letter]
                patterns.append(pattern_str)
                targets.append(target_idx)
        
        # Strategy 4: Length-balanced sampling (10% of samples)
        print("Generating length-balanced samples...")
        for _ in range(int(num_samples * 0.1)):
            # Sample words proportionally by length
            length_weights = {length: 1.0/len(words) for length, words in word_lengths.items()}
            chosen_length = random.choices(list(length_weights.keys()), 
                                         weights=list(length_weights.values()))[0]
            word = random.choice(word_lengths[chosen_length])
            
            # Adaptive revelation based on word length
            if len(word) <= 4:
                num_revealed = 1
            elif len(word) <= 8:
                num_revealed = random.randint(1, max(1, min(3, len(word) - 1)))
            else:
                num_revealed = random.randint(2, max(2, len(word) // 3))
            
            revealed_positions = random.sample(range(len(word)), num_revealed)
            
            pattern = list('_' * len(word))
            for pos in revealed_positions:
                pattern[pos] = word[pos]
            
            pattern_str = ''.join(pattern)
            hidden_letters = [word[i] for i in range(len(word)) if i not in revealed_positions]
            
            if hidden_letters:
                target_letter = Counter(hidden_letters).most_common(1)[0][0]
                target_idx = self.char_to_idx[target_letter]
                patterns.append(pattern_str)
                targets.append(target_idx)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        unique_targets = []
        
        for pattern, target in zip(patterns, targets):
            pattern_target_pair = (pattern, target)
            if pattern_target_pair not in seen:
                seen.add(pattern_target_pair)
                unique_patterns.append(pattern)
                unique_targets.append(target)
        
        print(f"Generated {len(unique_patterns)} unique training samples")
        print(f"Sample distribution by target letter:")
        target_dist = Counter(unique_targets)
        for idx in sorted(target_dist.keys()):
            letter = self.idx_to_char[idx]
            count = target_dist[idx]
            print(f"  {letter}: {count} samples ({100*count/len(unique_targets):.1f}%)")
        
        return unique_patterns, unique_targets
    
    def train_model(self, epochs=1000, batch_size=128, learning_rate=0.001):
        """Train the deep learning model with extensive data"""
        print("Starting model training with extensive dataset...")
        
        # Generate massive training data - 1M samples
        patterns, targets = self.generate_training_data(1000000)
        
        # Split data
        train_patterns, val_patterns, train_targets, val_targets = train_test_split(
            patterns, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets with larger batch size for efficiency
        train_dataset = HangmanDataset(train_patterns, train_targets)
        val_dataset = HangmanDataset(val_patterns, val_targets)
        
        # Larger batch size for faster training with more data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
        
        # Setup training with advanced optimizations
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, 
                               weight_decay=0.01, betas=(0.9, 0.999))
        
        # More sophisticated learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=learning_rate * 10,  # Peak learning rate
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # Warm up for 10% of training
            anneal_strategy='cos'
        )
        
        # Additional training optimizations
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        best_val_acc = 0
        patience = 50  # Early stopping patience
        patience_counter = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_patterns, batch_targets in train_loader:
                batch_patterns = batch_patterns.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision training for speed
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_patterns)
                        loss = criterion(outputs, batch_targets)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_patterns)
                    loss = criterion(outputs, batch_targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                total_loss += loss.item()
                num_batches += 1
                
                # Progress tracking for large datasets
                if num_batches % 1000 == 0:
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Validation phase - more frequent for large datasets
            if epoch % 5 == 0:
                self.model.eval()
                correct = 0
                total = 0
                val_loss = 0
                
                with torch.no_grad():
                    for batch_patterns, batch_targets in val_loader:
                        batch_patterns = batch_patterns.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        
                        if scaler is not None:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(batch_patterns)
                                loss = criterion(outputs, batch_targets)
                        else:
                            outputs = self.model(batch_patterns)
                            loss = criterion(outputs, batch_targets)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_targets.size(0)
                        correct += (predicted == batch_targets).sum().item()
                
                val_acc = 100 * correct / total
                avg_val_loss = val_loss / len(val_loader)
                val_accuracies.append(val_acc)
                
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                
                # Save best model and early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model('best_hangman_model.pth')
                    patience_counter = 0
                    print(f"New best validation accuracy: {best_val_acc:.2f}%")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return train_losses, val_accuracies
    
    def predict_letter(self, pattern, guessed_letters=None):
        """Predict the next best letter"""
        if guessed_letters is None:
            guessed_letters = []
        
        self.model.eval()
        
        # Create dataset for single prediction
        dataset = HangmanDataset([pattern], [0])  # Dummy target
        pattern_tensor = dataset.encode_pattern(pattern).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(pattern_tensor)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # Sort letters by probability
        letter_probs = [(self.idx_to_char[i], probabilities[i]) for i in range(26)]
        letter_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Return first letter that hasn't been guessed
        for letter, prob in letter_probs:
            if letter not in guessed_letters:
                return letter
        
        # Fallback
        return 'e'
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']

# Hangman API Integration (same as original but simplified)
class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        # Initialize our AI model
        self.ai = HangmanAI('best_hangman_model.pth')

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
        """Make a guess using our AI model"""
        guess_letter = self.ai.predict_letter(word.replace(' ', ''), self.guessed_letters)
        if verbose:
            print(f"AI predicts: {guess_letter}")
        return guess_letter

    def build_dictionary(self, dictionary_file_location):
        with open(dictionary_file_location, "r") as text_file:
            full_dictionary = text_file.read().splitlines()
        return full_dictionary

    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        
        response = self.request("/new_game", {"practice": practice})
        if response.get('status') == "approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            
            if verbose:
                print(f"New game started! ID: {game_id}, Tries: {tries_remains}, Word: {word}")
            
            while tries_remains > 0:
                guess_letter = self.guess(word, verbose)
                self.guessed_letters.append(guess_letter)
                
                if verbose:
                    print(f"Guessing: {guess_letter}")

                try:
                    res = self.request("/guess_letter", {
                        "request": "guess_letter",
                        "game_id": game_id,
                        "letter": guess_letter
                    })
                except Exception as e:
                    print(f'Exception: {e}')
                    continue

                if verbose:
                    print(f"Response: {res}")
                
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                
                if status == "success":
                    if verbose:
                        print(f"Game won: {game_id}")
                    return True
                elif status == "failed":
                    reason = res.get('reason', 'Tries exceeded!')
                    if verbose:
                        print(f"Game failed: {game_id}. Reason: {reason}")
                    return False
                elif status == "ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start new game")
        
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
            raise HangmanAPIError('Maintype was not text, or querystring')

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

    # Training and execution
if __name__ == "__main__":
    print("Training Advanced Hangman AI with 1M samples...")
    
    # Create and train model
    ai = HangmanAI()
    if not os.path.exists('best_hangman_model.pth'):
        print("No saved model found. Training new model...")
        train_losses, val_accuracies = ai.train_model(epochs=1000, batch_size=128)
    else:
        print("Using existing trained model.")
    
    print("Training completed! Starting API tests...")
    
    # Test with API
    api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)
    
    # Practice runs
    practice_wins = 0
    total_practice = 100
    
    for i in range(total_practice):
        print(f"Practice game {i+1}/{total_practice}")
        if api.start_game(practice=1, verbose=True):
            practice_wins += 1
    
    print(f"Practice Results: {practice_wins}/{total_practice} = {100*practice_wins/total_practice:.1f}%")
    
    [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    practice_success_rate = total_practice_successes / total_practice_runs if total_practice_runs > 0 else 0
    print(f'Total practice games: {total_practice_runs}, Success rate: {practice_success_rate:.3f}')
    
    # Official runs
    official_wins = 0
    total_official = 1000
    
    # for i in range(total_official):
    #     if (i + 1) % 100 == 0:
    #         print(f'Playing official game {i+1}/{total_official}')
        
    #     if api.start_game(practice=0, verbose=False):
    #         official_wins += 1
    #     time.sleep(0.5)
    
    # print(f"Official Results: {official_wins}/{total_official} = {100*official_wins/total_official:.1f}%")
    
    [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    success_rate = total_recorded_successes / total_recorded_runs if total_recorded_runs > 0 else 0
    print(f'Final overall success rate: {success_rate:.3f} ({100*success_rate:.1f}%)')