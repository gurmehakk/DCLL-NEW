# -*- coding: utf-8 -*-
"""
GRU-based Hangman Solver with Interval-1 Updates
Designed to achieve 70%+ accuracy - FIXED VERSION
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
import pickle
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class HangmanGRUDataset(Dataset):
    """Dataset class for Hangman GRU training"""
    def __init__(self, sequences, targets, max_length=20):
        self.sequences = sequences
        self.targets = targets
        self.max_length = max_length
        self.char_to_idx = {chr(i): i-96 for i in range(97, 123)}  # a=1, b=2, ..., z=26
        self.char_to_idx['_'] = 0  # unknown character
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        # Convert sequence to indices
        seq_indices = []
        for char in sequence:
            seq_indices.append(self.char_to_idx.get(char, 0))
        
        # Pad or truncate to max_length
        if len(seq_indices) < self.max_length:
            seq_indices.extend([0] * (self.max_length - len(seq_indices)))
        else:
            seq_indices = seq_indices[:self.max_length]
        
        # Convert target letter to index (a=0, b=1, ..., z=25)
        target_idx = ord(target) - ord('a')
        
        return torch.tensor(seq_indices, dtype=torch.long), torch.tensor(target_idx, dtype=torch.long)

class HangmanGRU(nn.Module):
    """GRU-based model for Hangman letter prediction"""
    def __init__(self, vocab_size=27, embedding_dim=64, hidden_dim=128, num_layers=2, dropout=0.3):
        super(HangmanGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # GRU layers
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 26)  # 26 letters
        
    def forward(self, x, lengths=None):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # GRU
        gru_out, hidden = self.gru(embedded)  # (batch_size, seq_len, hidden_dim)
        
        # Self-attention
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        # Use the last output (or mean pooling)
        if lengths is not None:
            # Use actual lengths for variable length sequences
            output = torch.stack([attn_out[i, lengths[i]-1] for i in range(len(lengths))])
        else:
            # Use mean pooling
            output = torch.mean(attn_out, dim=1)
        
        # Final layers
        output = self.dropout(output)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return F.softmax(output, dim=1)

class AdvancedHangmanGRU:
    """Advanced GRU-based Hangman solver with interval-1 updates"""
    def __init__(self, model_path=None):
        self.device = device
        self.model = HangmanGRU().to(self.device)
        self.char_to_idx = {chr(i): i-96 for i in range(97, 123)}
        self.char_to_idx['_'] = 0
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Letter frequency in English (for fallback)
        self.english_freq = 'etaoinshrdlcumwfgypbvkjxqz'
        
        # Pattern analysis cache
        self.pattern_cache = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            print(f"Loaded pre-trained model from {model_path}")
    
    def prepare_training_data(self, dictionary_path, max_samples=100000):
        """Prepare training data from dictionary"""
        words = self.build_dictionary(dictionary_path)
        
        sequences = []
        targets = []
        
        print(f"Preparing training data from {len(words)} words...")
        
        for word in words[:max_samples]:
            if len(word) < 3 or len(word) > 20:  # Skip very short/long words
                continue
                
            # Create multiple training examples per word
            for num_revealed in range(1, min(len(word), 8)):
                # Randomly mask letters
                indices = list(range(len(word)))
                revealed_indices = random.sample(indices, num_revealed)
                
                # Create pattern
                pattern = ['_'] * len(word)
                for idx in revealed_indices:
                    pattern[idx] = word[idx]
                
                # Choose a target letter from unrevealed positions
                unrevealed = [i for i in indices if i not in revealed_indices]
                if unrevealed:
                    target_idx = random.choice(unrevealed)
                    target_letter = word[target_idx]
                    
                    sequences.append(''.join(pattern))
                    targets.append(target_letter)
        
        print(f"Generated {len(sequences)} training examples")
        return sequences, targets
    
    def train_model(self, dictionary_path, epochs=50, batch_size=64, learning_rate=0.001):
        """Train the GRU model with interval-1 updates"""
        sequences, targets = self.prepare_training_data(dictionary_path)
        
        # Split data
        train_seq, val_seq, train_targets, val_targets = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = HangmanGRUDataset(train_seq, train_targets)
        val_dataset = HangmanGRUDataset(val_seq, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer and loss
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0
        patience_counter = 0
        
        print("Starting training...")
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences, targets = sequences.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
                # Interval-1 update: Update probabilities every batch (per epoch granularity)
                if batch_idx % 100 == 0:
                    self._update_probabilities()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences, targets = sequences.to(self.device), targets.to(self.device)
                    outputs = self.model(sequences)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            scheduler.step(val_loss)
            
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
                  f'Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best_gru_hangman_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered")
                    break
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    def _update_probabilities(self):
        """Interval-1 update: Update internal probability distributions"""
        # This method can be customized to update internal probability caches
        # For now, we'll clear the pattern cache to force re-computation
        self.pattern_cache.clear()
    
    def predict_letter(self, pattern, guessed_letters, words_list=None):
        """Predict the next letter using GRU model and fallback strategies"""
        pattern = pattern.replace(' ', '')
        
        # Strategy 1: GRU Model Prediction
        gru_prediction = self._gru_predict(pattern, guessed_letters)
        
        # Strategy 2: Pattern Analysis
        pattern_prediction = self._pattern_analysis(pattern, guessed_letters, words_list)
        
        # Strategy 3: Frequency Analysis
        freq_prediction = self._frequency_analysis(pattern, guessed_letters)
        
        # Combine predictions with weights
        final_prediction = self._combine_predictions(
            gru_prediction, pattern_prediction, freq_prediction, guessed_letters
        )
        
        return final_prediction
    
    def _gru_predict(self, pattern, guessed_letters):
        """Use GRU model to predict next letter"""
        try:
            self.model.eval()
            
            # Convert pattern to tensor
            seq_indices = [self.char_to_idx.get(char, 0) for char in pattern]
            
            # Pad to max length
            max_len = 20
            if len(seq_indices) < max_len:
                seq_indices.extend([0] * (max_len - len(seq_indices)))
            else:
                seq_indices = seq_indices[:max_len]
            
            input_tensor = torch.tensor([seq_indices], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                probabilities = self.model(input_tensor)[0]  # Get first (and only) prediction
            
            # Convert to numpy and find best unguessed letter
            probs = probabilities.cpu().numpy()
            
            # Mask already guessed letters
            for letter in guessed_letters:
                if 'a' <= letter <= 'z':
                    idx = ord(letter) - ord('a')
                    probs[idx] = 0
            
            if np.sum(probs) > 0:
                probs = probs / np.sum(probs)  # Renormalize
                best_idx = np.argmax(probs)
                return chr(best_idx + ord('a')), probs[best_idx]
            
        except Exception as e:
            print(f"GRU prediction error: {e}")
        
        return None, 0
    
    def _pattern_analysis(self, pattern, guessed_letters, words_list):
        """Analyze patterns in dictionary words"""
        if words_list is None:
            return None, 0
        
        # Create regex pattern
        regex_pattern = pattern.replace('_', '.')
        
        matching_words = []
        for word in words_list:
            if len(word) == len(pattern) and re.match(regex_pattern, word):
                # Check if word contains any guessed letters in wrong positions
                valid = True
                for i, char in enumerate(word):
                    if pattern[i] != '_' and pattern[i] != char:
                        valid = False
                        break
                    if pattern[i] == '_' and char in [c for j, c in enumerate(pattern) if j != i and c != '_']:
                        # This would create a duplicate letter in revealed positions
                        continue
                
                if valid:
                    matching_words.append(word)
        
        if not matching_words:
            return None, 0
        
        # Count letters in blank positions
        letter_counts = Counter()
        total_blanks = 0
        
        for word in matching_words:
            for i, char in enumerate(word):
                if pattern[i] == '_' and char not in guessed_letters:
                    letter_counts[char] += 1
                    total_blanks += 1
        
        if total_blanks == 0:
            return None, 0
        
        best_letter = letter_counts.most_common(1)[0]
        return best_letter[0], best_letter[1] / total_blanks
    
    def _frequency_analysis(self, pattern, guessed_letters):
        """Use English letter frequency as fallback"""
        for letter in self.english_freq:
            if letter not in guessed_letters:
                return letter, 0.1  # Low confidence score
        return 'a', 0.01  # Ultimate fallback
    
    def _combine_predictions(self, gru_pred, pattern_pred, freq_pred, guessed_letters):
        """Combine different prediction strategies"""
        candidates = []
        
        if gru_pred[0] and gru_pred[0] not in guessed_letters:
            candidates.append((gru_pred[0], gru_pred[1] * 0.6))  # 60% weight for GRU
        
        if pattern_pred[0] and pattern_pred[0] not in guessed_letters:
            candidates.append((pattern_pred[0], pattern_pred[1] * 0.3))  # 30% weight for pattern
        
        if freq_pred[0] and freq_pred[0] not in guessed_letters:
            candidates.append((freq_pred[0], freq_pred[1] * 0.1))  # 10% weight for frequency
        
        if not candidates:
            return freq_pred[0]  # Ultimate fallback
        
        # Return the candidate with highest weighted score
        best_candidate = max(candidates, key=lambda x: x[1])
        return best_candidate[0]
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = checkpoint['idx_to_char']
        print(f"Model loaded from {path}")
    
    @staticmethod
    def build_dictionary(dictionary_file_location):
        """Build dictionary from file"""
        with open(dictionary_file_location, "r") as text_file:
            full_dictionary = text_file.read().splitlines()
        return full_dictionary

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

# Updated HangmanAPI class to use GRU model
class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self.determine_hangman_url()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        # Initialize GRU model
        self.gru_model = AdvancedHangmanGRU('best_gru_hangman_model.pth')
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = []

    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']
        data = {link: 0 for link in links}

        for link in links:
            try:
                requests.get(link, timeout=5)
                for i in range(3):  # Reduced from 10 to 3 for faster startup
                    s = time.time()
                    requests.get(link, timeout=5)
                    data[link] += time.time() - s
            except:
                data[link] = float('inf')  # Mark failed connections

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word, verbose=False):
        """Use GRU model to make guess"""
        guess_letter = self.gru_model.predict_letter(
            word, 
            self.guessed_letters, 
            self.full_dictionary
        )
        
        if verbose:
            print(f"GRU prediction: {guess_letter}")
        
        return guess_letter

    def build_dictionary(self, dictionary_file_location):
        try:
            with open(dictionary_file_location, "r") as text_file:
                full_dictionary = text_file.read().splitlines()
            return full_dictionary
        except FileNotFoundError:
            print(f"Warning: Dictionary file {dictionary_file_location} not found. Using basic frequency approach.")
            return []

    def start_game(self, practice=True, verbose=True):
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
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
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
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
    
    def request(self, path, args=None, post_args=None, method=None):
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
        content_type = headers.get('content-type', '').lower()
        
        # Fixed content type handling
        try:
            if 'json' in content_type:
                result = response.json()
            elif 'text' in content_type or 'html' in content_type:
                # Try to parse as JSON first, then check for query string
                try:
                    result = response.json()
                except (ValueError, json.JSONDecodeError):
                    # If JSON parsing fails, check for query string format
                    if "access_token" in response.text:
                        query_str = parse_qs(response.text)
                        if "access_token" in query_str:
                            result = {"access_token": query_str["access_token"][0]}
                            if "expires" in query_str:
                                result["expires"] = query_str["expires"][0]
                        else:
                            raise HangmanAPIError("Failed to parse response")
                    else:
                        # Try to parse as JSON one more time
                        try:
                            result = json.loads(response.text)
                        except:
                            raise HangmanAPIError(f'Failed to parse response: {response.text[:100]}')
            else:
                # For other content types, try JSON parsing
                try:
                    result = response.json()
                except:
                    raise HangmanAPIError(f'Unsupported content type: {content_type}')
                    
        except Exception as e:
            print(f"Response parsing error: {e}")
            print(f"Content-Type: {content_type}")
            print(f"Response text: {response.text[:200]}")
            raise HangmanAPIError(f'Response parsing failed: {str(e)}')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result

# Training and usage example
if __name__ == "__main__":
    # Initialize and train the GRU model
    print("Initializing GRU Hangman model...")
    gru_hangman = AdvancedHangmanGRU()
    
    # Train the model (uncomment to train)
    # print("Training model...")
    # gru_hangman.train_model("words_250000_train.txt", epochs=50)
    
    # Initialize API with trained model
    api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)
    
    # Practice games
    print("Starting practice games...")
    for i in range(5):  # Start with fewer games for testing
        print(f"Game {i+1}")
        api.start_game(practice=1, verbose=True)
    
    [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    practice_success_rate = total_practice_successes / total_practice_runs if total_practice_runs > 0 else 0
    print('Run %d practice games out of an allotted 100,000. Practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))
    
    # Actual games (uncomment when ready)
    # for i in range(1000):
    #     print('Playing', i, 'th game')
    #     api.start_game(practice=0, verbose=False)
    #     time.sleep(0.5)
    
    # [total_practice_runs, total_recorded_runs, total_recorded_successes, total_practice_successes] = api.my_status()
    # success_rate = total_recorded_successes / total_recorded_runs
    # print('Overall success rate = %.3f' % success_rate)