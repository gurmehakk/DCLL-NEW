import json
import requests
import random
import string
import secrets
import time
import re
import collections
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pickle
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Improved character mapping with better encoding
def get_enhanced_char_mapping():
    char_mapping = {'_': 0, '<PAD>': 1}  # Use 0 for underscore, 1 for padding
    for i, char in enumerate(string.ascii_lowercase):
        char_mapping[char] = i + 2  # Start from 2
    return char_mapping

class AdvancedHangmanDataset:
    def __init__(self, dictionary_path="words_250000_train.txt"):
        self.dictionary = self.load_dictionary(dictionary_path)
        self.char_mapping = get_enhanced_char_mapping()
        self.reverse_mapping = {v: k for k, v in self.char_mapping.items()}
        
        # Enhanced features
        self.word_stats = self.compute_word_statistics()
        self.position_stats = self.compute_position_statistics()
        self.length_stats = self.compute_length_statistics()
        
    def load_dictionary(self, path):
        try:
            with open(path, "r") as f:
                words = [line.strip().lower() for line in f if line.strip()]
            return [word for word in words if word.isalpha() and 3 <= len(word) <= 35]
        except FileNotFoundError:
            print(f"Dictionary file {path} not found. Using minimal backup.")
            return ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had']
    
    def compute_word_statistics(self):
        stats = {}
        for word in self.dictionary:
            length = len(word)
            if length not in stats:
                stats[length] = {
                    'vowel_counts': [],
                    'consonant_counts': [],
                    'unique_counts': [],
                    'first_letters': Counter(),
                    'last_letters': Counter(),
                    'letter_frequencies': Counter()
                }
            
            vowels = sum(1 for c in word if c in 'aeiou')
            consonants = len(word) - vowels
            unique_chars = len(set(word))
            
            stats[length]['vowel_counts'].append(vowels)
            stats[length]['consonant_counts'].append(consonants)
            stats[length]['unique_counts'].append(unique_chars)
            stats[length]['first_letters'][word[0]] += 1
            stats[length]['last_letters'][word[-1]] += 1
            
            for char in word:
                stats[length]['letter_frequencies'][char] += 1
        
        return stats
    
    def compute_position_statistics(self):
        position_stats = defaultdict(lambda: defaultdict(Counter))
        for word in self.dictionary:
            for pos, char in enumerate(word):
                position_stats[len(word)][pos][char] += 1
        return position_stats
    
    def compute_length_statistics(self):
        length_counter = Counter(len(word) for word in self.dictionary)
        return length_counter
    
    def create_enhanced_permutations(self, word, max_masks=None):
        """Create more intelligent permutations based on word patterns"""
        unique_letters = list(set(word))
        permutations = set()
        
        # Strategic masking based on letter frequency and position
        vowels = [c for c in unique_letters if c in 'aeiou']
        consonants = [c for c in unique_letters if c not in 'aeiou']
        
        # Always include the original word
        permutations.add(word)
        
        # Mask vowels progressively (more realistic game scenarios)
        for num_vowels in range(1, min(len(vowels) + 1, 4)):
            for vowel_combo in self.combinations(vowels, num_vowels):
                masked = word
                for vowel in vowel_combo:
                    masked = masked.replace(vowel, '_')
                permutations.add(masked)
        
        # Mask consonants progressively
        for num_consonants in range(1, min(len(consonants) + 1, 3)):
            for consonant_combo in self.combinations(consonants, num_consonants):
                masked = word
                for consonant in consonant_combo:
                    masked = masked.replace(consonant, '_')
                permutations.add(masked)
        
        # Mixed masking (more realistic)
        for num_total in range(2, min(len(unique_letters), 5)):
            for letter_combo in self.combinations(unique_letters, num_total):
                masked = word
                for letter in letter_combo:
                    masked = masked.replace(letter, '_')
                permutations.add(masked)
        
        # Position-based masking (mask beginning, middle, end)
        if len(word) >= 6:
            # Mask beginning
            masked = '_' * 2 + word[2:]
            permutations.add(masked)
            
            # Mask end
            masked = word[:-2] + '_' * 2
            permutations.add(masked)
            
            # Mask middle
            mid_start = len(word) // 3
            mid_end = 2 * len(word) // 3
            masked = word[:mid_start] + '_' * (mid_end - mid_start) + word[mid_end:]
            permutations.add(masked)
        
        result = list(permutations)
        if max_masks and len(result) > max_masks:
            # Keep most diverse permutations
            result = random.sample(result, max_masks)
        
        return result
    
    def combinations(self, items, r):
        """Generate combinations of items taken r at a time"""
        from itertools import combinations
        return list(combinations(items, r))
    
    def encode_input_enhanced(self, word, max_length=35):
        """Enhanced input encoding with positional and contextual features"""
        # Basic character encoding
        char_vector = [self.char_mapping['<PAD>']] * max_length
        
        word_len = min(len(word), max_length)
        start_pos = max_length - word_len
        
        for i, char in enumerate(word):
            if i < max_length:
                char_vector[start_pos + i] = self.char_mapping.get(char, self.char_mapping['_'])
        
        # Additional features
        features = []
        
        # Length feature
        features.append(word_len)
        
        # Revealed vs hidden ratio
        revealed_count = sum(1 for c in word if c != '_')
        hidden_count = word_len - revealed_count
        features.append(revealed_count / word_len if word_len > 0 else 0)
        features.append(hidden_count / word_len if word_len > 0 else 0)
        
        # Vowel/consonant features
        revealed_letters = [c for c in word if c != '_']
        vowel_count = sum(1 for c in revealed_letters if c in 'aeiou')
        consonant_count = len(revealed_letters) - vowel_count
        
        features.append(vowel_count / len(revealed_letters) if revealed_letters else 0)
        features.append(consonant_count / len(revealed_letters) if revealed_letters else 0)
        
        # Pattern features
        features.append(1 if word.startswith('_') else 0)
        features.append(1 if word.endswith('_') else 0)
        
        # Consecutive underscore patterns
        consecutive_underscores = 0
        max_consecutive = 0
        for char in word:
            if char == '_':
                consecutive_underscores += 1
                max_consecutive = max(max_consecutive, consecutive_underscores)
            else:
                consecutive_underscores = 0
        features.append(max_consecutive / word_len if word_len > 0 else 0)
        
        return char_vector + features
    
    def encode_output_enhanced(self, word):
        """Enhanced output encoding with letter priorities"""
        output_vector = [0] * 26
        letter_priorities = [0] * 26
        
        unique_letters = set(word)
        letter_counts = Counter(word)
        
        for letter in unique_letters:
            if letter in string.ascii_lowercase:
                idx = ord(letter) - ord('a')
                output_vector[idx] = 1
                # Add priority based on frequency in word
                letter_priorities[idx] = letter_counts[letter] / len(word)
        
        return output_vector, letter_priorities

class ImprovedNeuralNetwork(nn.Module):
    def __init__(self, vocab_size=28, embed_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        # Enhanced embedding with positional encoding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.pos_encoding = nn.Parameter(torch.randn(35, embed_dim) * 0.1)
        
        # Bidirectional LSTM with residual connections
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=8, dropout=dropout, batch_first=True
        )
        
        # Additional feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(8, 32),  # 8 additional features
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 26)
        )
        
        # Letter priority head
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 26),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Split input into character sequence and additional features
        char_seq = x[:, :35].long()  # Character sequence
        add_features = x[:, 35:].float()  # Additional features
        
        # Character embedding with positional encoding
        embedded = self.embedding(char_seq)
        embedded = embedded + self.pos_encoding[:embedded.size(1), :].unsqueeze(0)
        embedded = self.dropout(embedded)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Attention mechanism
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM output and attention
        combined = lstm_out + attended
        
        # Global average pooling and max pooling
        avg_pooled = torch.mean(combined, dim=1)
        max_pooled, _ = torch.max(combined, dim=1)
        
        # Combine pooled representations
        sequence_repr = avg_pooled + max_pooled
        
        # Process additional features
        feature_repr = self.feature_fc(add_features)
        
        # Combine all representations
        final_repr = torch.cat([sequence_repr, feature_repr], dim=1)
        
        # Classification
        letter_logits = self.classifier(final_repr)
        letter_priorities = self.priority_head(final_repr)
        
        return letter_logits, letter_priorities

class EnhancedHangmanSolver:
    def __init__(self, model_path=None, dictionary_path="words_250000_train.txt"):
        self.dataset = AdvancedHangmanDataset(dictionary_path)
        self.model = ImprovedNeuralNetwork()
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
        
        # Vowel priors by length
        self.vowel_priors = self.compute_vowel_priors()
        
        # Common letter frequencies
        self.letter_frequencies = Counter()
        for word in self.dataset.dictionary:
            self.letter_frequencies.update(word)
        
        # Normalize frequencies
        total_letters = sum(self.letter_frequencies.values())
        self.letter_frequencies = {
            k: v / total_letters for k, v in self.letter_frequencies.items()
        }
        
    def compute_vowel_priors(self):
        priors = {}
        for length in range(3, 36):
            length_words = [w for w in self.dataset.dictionary if len(w) == length]
            if not length_words:
                continue
                
            vowel_probs = {}
            for vowel in 'aeiou':
                count = sum(1 for word in length_words if vowel in word)
                vowel_probs[vowel] = count / len(length_words)
            
            # Sort by probability
            sorted_vowels = sorted(vowel_probs.items(), key=lambda x: x[1], reverse=True)
            priors[length] = sorted_vowels
            
        return priors
    
    def predict_next_letter(self, pattern, guessed_letters, tries_remaining):
        """Enhanced prediction combining multiple strategies"""
        available_letters = [c for c in string.ascii_lowercase if c not in guessed_letters]
        if not available_letters:
            return 'e'  # Fallback
        
        word_length = len(pattern.replace(' ', ''))
        
        # Strategy 1: Vowel priority for early guesses
        if tries_remaining > 4 and len(guessed_letters) <= 2:
            if word_length in self.vowel_priors:
                for vowel, prob in self.vowel_priors[word_length]:
                    if vowel in available_letters and prob > 0.3:
                        return vowel
        
        # Strategy 2: Neural network prediction
        if hasattr(self, 'model'):
            try:
                # Prepare input
                clean_pattern = pattern.replace(' ', '')
                input_features = self.dataset.encode_input_enhanced(clean_pattern)
                input_tensor = torch.tensor([input_features], dtype=torch.float32)
                
                with torch.no_grad():
                    letter_logits, letter_priorities = self.model(input_tensor)
                    
                # Combine logits and priorities
                combined_scores = F.softmax(letter_logits[0], dim=0) + letter_priorities[0]
                
                # Score available letters
                letter_scores = []
                for letter in available_letters:
                    idx = ord(letter) - ord('a')
                    score = combined_scores[idx].item()
                    
                    # Boost score based on word patterns
                    score += self.get_pattern_boost(letter, clean_pattern, word_length)
                    
                    letter_scores.append((letter, score))
                
                # Sort by score and return best
                letter_scores.sort(key=lambda x: x[1], reverse=True)
                return letter_scores[0][0]
                
            except Exception as e:
                print(f"Neural network prediction failed: {e}")
        
        # Strategy 3: Frequency-based fallback with pattern matching
        possible_words = self.get_matching_words(pattern)
        if possible_words:
            letter_counts = Counter()
            for word in possible_words:
                letter_counts.update(set(word))  # Count unique letters per word
            
            # Score available letters
            letter_scores = []
            for letter in available_letters:
                score = letter_counts.get(letter, 0) / len(possible_words)
                score += self.letter_frequencies.get(letter, 0) * 0.1  # Small frequency boost
                letter_scores.append((letter, score))
            
            letter_scores.sort(key=lambda x: x[1], reverse=True)
            return letter_scores[0][0]
        
        # Strategy 4: Pure frequency fallback
        frequency_order = 'etaoinshrdlcumwfgypbvkjxqz'
        for letter in frequency_order:
            if letter in available_letters:
                return letter
        
        return available_letters[0]  # Ultimate fallback
    
    def get_pattern_boost(self, letter, pattern, word_length):
        """Calculate pattern-based score boost"""
        boost = 0.0
        
        # Position-based scoring
        if word_length in self.dataset.position_stats:
            for pos, char in enumerate(pattern):
                if char == '_' and pos in self.dataset.position_stats[word_length]:
                    pos_freq = self.dataset.position_stats[word_length][pos].get(letter, 0)
                    total_pos = sum(self.dataset.position_stats[word_length][pos].values())
                    if total_pos > 0:
                        boost += (pos_freq / total_pos) * 0.1
        
        # Common pattern boost
        if letter in 'etaoin':  # Most common letters
            boost += 0.05
        
        # Vowel/consonant balance
        revealed = [c for c in pattern if c != '_' and c != ' ']
        if revealed:
            vowel_ratio = sum(1 for c in revealed if c in 'aeiou') / len(revealed)
            if letter in 'aeiou' and vowel_ratio < 0.4:
                boost += 0.03
            elif letter not in 'aeiou' and vowel_ratio > 0.6:
                boost += 0.03
        
        return boost
    
    def get_matching_words(self, pattern):
        """Find words matching the current pattern"""
        clean_pattern = pattern.replace(' ', '')
        word_length = len(clean_pattern)
        
        regex_pattern = clean_pattern.replace('_', '.')
        matching_words = []
        
        for word in self.dataset.dictionary:
            if len(word) == word_length and re.match(f'^{regex_pattern}$', word):
                matching_words.append(word)
        
        return matching_words

# Training function with improvements
def train_enhanced_model(dataset, epochs=15, batch_size=256, lr=0.001):
    """Train the enhanced model with better techniques"""
    
    # Generate training data
    print("Generating enhanced training data...")
    input_data, output_data, priority_data = [], [], []
    
    for i, word in enumerate(dataset.dictionary):
        if i % 10000 == 0:
            print(f"Processing word {i}/{len(dataset.dictionary)}")
        
        # Create diverse permutations
        permutations = dataset.create_enhanced_permutations(word, max_masks=20)
        
        for masked_word in permutations:
            if masked_word != word and '_' in masked_word:  # Skip original and non-masked
                try:
                    input_features = dataset.encode_input_enhanced(masked_word)
                    output_vector, priority_vector = dataset.encode_output_enhanced(word)
                    
                    input_data.append(input_features)
                    output_data.append(output_vector)
                    priority_data.append(priority_vector)
                except Exception as e:
                    continue
    
    print(f"Generated {len(input_data)} training samples")
    
    # Convert to tensors
    X = torch.tensor(input_data, dtype=torch.float32)
    y = torch.tensor(output_data, dtype=torch.float32)  
    priorities = torch.tensor(priority_data, dtype=torch.float32)
    
    # Create data loader
    dataset_train = torch.utils.data.TensorDataset(X, y, priorities)
    dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = ImprovedNeuralNetwork()
    
    # Loss functions
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_priority = nn.MSELoss()
    
    # Optimizer with learning rate scheduling
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets, target_priorities) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            logits, predicted_priorities = model(inputs)
            
            # Calculate losses
            class_loss = criterion_class(logits, targets)
            priority_loss = criterion_priority(predicted_priorities, target_priorities)
            
            # Combined loss
            total_batch_loss = class_loss + 0.1 * priority_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            num_batches += 1
            
            if batch_idx % 1000 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {total_batch_loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), "enhanced_hangman_model.pt")
    print("Model saved as enhanced_hangman_model.pt")
    
    return model

# Usage example
if __name__ == "__main__":
    # Initialize dataset
    dataset = AdvancedHangmanDataset("words_250000_train.txt")
    
    # Train the model (uncomment to train)
    # model = train_enhanced_model(dataset, epochs=20, batch_size=256, lr=0.001)
    
    # Initialize solver with trained model
    solver = EnhancedHangmanSolver("enhanced_hangman_model.pt", "words_250000_train.txt")
    
    # Test the solver
    test_patterns = [
        "_ _ _ _ _",
        "t _ e _ _", 
        "h _ _ _ o",
        "_ o _ _ _ r"
    ]
    
    for pattern in test_patterns:
        guessed = ['t', 'e'] if 't' in pattern or 'e' in pattern else []
        prediction = solver.predict_next_letter(pattern, guessed, 5)
        print(f"Pattern: {pattern}, Guessed: {guessed}, Prediction: {prediction}")