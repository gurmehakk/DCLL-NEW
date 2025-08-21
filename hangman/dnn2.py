#!/usr/bin/env python
# coding: utf-8

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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import json
import requests
import random
import string
import time
import re
import collections
import os

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class HangmanDataset(Dataset):
    """Dataset for training the transformer model on masked words"""
    
    def __init__(self, word_list, mask_probability=0.25, sequence_length=15):
        self.words = [word for word in word_list if word.isalpha() and 0 < len(word) <= sequence_length]
        self.mask_prob = mask_probability
        self.seq_len = sequence_length
        
        # Create vocabulary: padding, mask token, then all lowercase letters
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        word = self.words[index]
        characters = list(word.lower())
        
        # Pad sequence if necessary
        if len(characters) < self.seq_len:
            characters += ['<PAD>'] * (self.seq_len - len(characters))
        
        input_sequence = []
        target_sequence = []
        
        for char in characters:
            if char == '<PAD>':
                input_sequence.append(self.char_to_id['<PAD>'])
                target_sequence.append(-100)  # Ignore padding in loss
            else:
                if random.random() < self.mask_prob:
                    input_sequence.append(self.char_to_id['<MASK>'])
                    target_sequence.append(self.char_to_id[char])
                else:
                    input_sequence.append(self.char_to_id[char])
                    target_sequence.append(-100)  # Don't predict unmasked tokens
        
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, embedding_dim, max_sequence_length=100):
        super(PositionalEncoder, self).__init__()
        
        position_encoding = torch.zeros(max_sequence_length, embedding_dim)
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                           -(math.log(10000.0) / embedding_dim))
        
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        position_encoding = position_encoding.unsqueeze(0)
        
        self.register_buffer('position_encoding', position_encoding)

    def forward(self, x):
        sequence_length = x.size(1)
        return x + self.position_encoding[:, :sequence_length, :]


class HangmanTransformer(nn.Module):
    """Transformer model for predicting masked characters in words"""
    
    def __init__(self, vocab_size=28, embedding_dim=128, num_heads=4, 
                 num_layers=6, feedforward_dim=512, max_length=100):
        super(HangmanTransformer, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoder = PositionalEncoder(embedding_dim, max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, token_ids):
        # Embed tokens and add positional encoding
        embeddings = self.token_embedding(token_ids)
        encoded_input = self.positional_encoder(embeddings)
        
        # Pass through transformer
        transformer_output = self.transformer_encoder(encoded_input)
        
        # Project to vocabulary size
        logits = self.output_projection(transformer_output)
        return logits


def train_transformer_model(word_dictionary, mask_prob, model_filename, 
                          epochs=100, batch_size=128, learning_rate=0.0001, 
                          max_length=15, device='cpu'):
    """Train a transformer model for the given masking probability"""
    
    print(f"Training model: {model_filename}")
    
    # Create dataset and dataloader
    dataset = HangmanDataset(word_dictionary, mask_probability=mask_prob, sequence_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = HangmanTransformer(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=128,
        num_heads=4,
        num_layers=6,
        feedforward_dim=512,
        max_length=20
    ).to(device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        for input_ids, targets in dataloader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    return model


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
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        # Calculate letter frequencies
        self.letter_frequencies = self._calculate_letter_frequencies()
        
        # Initialize working dictionary and substring patterns
        self.current_dictionary = []
        self.substring_dictionary = self._build_substring_patterns()
        
        # Setup device and vocabulary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Initialize and load transformer models
        self._initialize_models()
        
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

    def _calculate_letter_frequencies(self):
        """Calculate letter frequency distribution from dictionary"""
        all_letters = ''.join(self.full_dictionary)
        return collections.Counter(all_letters).most_common()

    def _build_substring_patterns(self):
        """Build dictionary of substring patterns for pattern matching"""
        max_word_length = max(len(word) for word in self.full_dictionary)
        substring_dict = {length: [] for length in range(3, min(max_word_length, 30) + 1)}
        
        for length in range(3, min(max_word_length, 30) + 1):
            for word in self.full_dictionary:
                if len(word) >= length:
                    for i in range(len(word) - length + 1):
                        substring_dict[length].append(word[i:i + length])
        
        return substring_dict

    def _initialize_models(self):
        """Initialize and load all transformer models"""
        model_configs = [
            ('hangman_model_high_mask.pt', 'high_mask_model'),
            ('hangman_model_medium_mask.pt', 'medium_mask_model'),
            ('hangman_model_low_mask.pt', 'low_mask_model'),
            ('hangman_model_short_words.pt', 'short_word_model')
        ]
        
        for filename, attr_name in model_configs:
            model = HangmanTransformer(
                vocab_size=len(self.vocabulary),
                embedding_dim=128,
                num_heads=4,
                num_layers=6,
                feedforward_dim=512,
                max_length=20 
            ).to(self.device)
            
            try:
                model.load_state_dict(torch.load(filename, map_location=self.device, weights_only=True))
                model.eval()
                setattr(self, attr_name, model)
                print(f"Loaded {filename} successfully")
            except FileNotFoundError:
                print(f"Warning: {filename} not found. Setting {attr_name} to None.")
                setattr(self, attr_name, None)

    def _get_character_frequencies(self, word_list):
        """Calculate character frequency from filtered word list"""
        char_counter = collections.Counter()
        for word in word_list:
            unique_chars = set(word)
            for char in unique_chars:
                char_counter[char] += 1
        return char_counter

    def _filter_by_pattern(self, pattern):
        """Filter dictionary by regex pattern"""
        regex_pattern = "^" + pattern + "$"
        filtered_words = []
        
        for word in self.current_dictionary:
            if len(word) == len(pattern) and re.fullmatch(regex_pattern, word):
                filtered_words.append(word)
        
        return filtered_words

    def _get_substring_matches(self, pattern):
        """Get character frequencies from substring pattern matching"""
        pattern_length = len(pattern)
        if pattern_length in self.substring_dictionary:
            matching_substrings = []
            for substring in self.substring_dictionary[pattern_length]:
                if re.fullmatch("^" + pattern + "$", substring):
                    matching_substrings.append(substring)
            return self._get_character_frequencies(matching_substrings)
        return collections.Counter()

    def _get_model_predictions(self, model, word_pattern):
        """Get character predictions from transformer model"""
        if model is None:
            # Return uniform distribution if model not available
            return {char: 1.0/26 for char in string.ascii_lowercase}
        
        # Convert pattern to model input
        model_input = []
        for char in word_pattern:
            if char == '.':
                model_input.append('<MASK>')
            elif char in self.vocabulary:
                model_input.append(char)
            else:
                model_input.append('<PAD>')
        
        # Convert to tensor
        input_ids = [self.char_to_id.get(char, self.char_to_id['<PAD>']) for char in model_input]
        
        # Truncate if too long
        model_max_length = model.positional_encoder.position_encoding.size(1)
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            
            # Find mask positions
            mask_positions = [i for i, char in enumerate(model_input) if char == '<MASK>']
            
            # Calculate probabilities for each letter
            letter_probabilities = collections.Counter()
            letter_indices = range(2, 28)  # Skip <PAD> and <MASK>
            
            if mask_positions:
                for position in mask_positions:
                    if position < logits.size(1):
                        position_logits = logits[0, position]
                        position_probs = torch.softmax(position_logits, dim=0)
                        
                        for letter_idx in letter_indices:
                            if letter_idx < len(self.id_to_char):
                                letter = self.id_to_char[letter_idx]
                                letter_probabilities[letter] += position_probs[letter_idx].item()
                
                # Average over all mask positions
                for letter in letter_probabilities:
                    letter_probabilities[letter] /= len(mask_positions)
            else:
                # No masks, return uniform distribution
                for letter in string.ascii_lowercase:
                    letter_probabilities[letter] = 1.0 / 26
        
        return dict(letter_probabilities)

    def _calculate_vowel_ratio(self, known_letters):
        """Calculate ratio of vowels in known letters"""
        if not known_letters:
            return 0.0
        vowels = set('aeiou')
        vowel_count = sum(1 for char in known_letters if char in vowels)
        return vowel_count / len(known_letters)

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Enhanced "guess" function with transformer models #
        ###############################################
        
        # Parse word state
        clean_pattern = word[::2].replace("_", ".")
        word_length = len(clean_pattern)
        unknown_count = clean_pattern.count('.')
        unknown_ratio = unknown_count / word_length if word_length > 0 else 0.0
        
        # Update current dictionary based on pattern
        self.current_dictionary = self._filter_by_pattern(clean_pattern)
        
        # Get heuristic letter frequencies
        if self.current_dictionary:
            heuristic_frequencies = self._get_character_frequencies(self.current_dictionary)
        else:
            # Fallback to substring matching
            heuristic_frequencies = self._get_substring_matches(clean_pattern)
            if not heuristic_frequencies:
                # Final fallback to overall frequencies
                heuristic_frequencies = collections.Counter(dict(self.letter_frequencies))
        
        # Normalize heuristic scores
        total_heuristic = sum(heuristic_frequencies.values())
        heuristic_scores = {}
        if total_heuristic > 0:
            for letter, count in heuristic_frequencies.items():
                heuristic_scores[letter] = count / total_heuristic
        
        # Get model predictions and set weights
        model_predictions = {}
        weights = {}
        
        if word_length <= 7:
            # Use short word model for short words
            if hasattr(self, 'short_word_model') and self.short_word_model is not None:
                model_predictions['short'] = self._get_model_predictions(self.short_word_model, clean_pattern)
            if hasattr(self, 'high_mask_model') and self.high_mask_model is not None:
                model_predictions['high'] = self._get_model_predictions(self.high_mask_model, clean_pattern)
            if hasattr(self, 'medium_mask_model') and self.medium_mask_model is not None:
                model_predictions['medium'] = self._get_model_predictions(self.medium_mask_model, clean_pattern)
            if hasattr(self, 'low_mask_model') and self.low_mask_model is not None:
                model_predictions['low'] = self._get_model_predictions(self.low_mask_model, clean_pattern)
            
            # Weight models based on unknown ratio and available models
            if unknown_ratio > 0.7:
                weights = {'heuristic': 0.35}
                if 'short' in model_predictions:
                    weights['short'] = 0.15
                if 'high' in model_predictions:
                    weights['high'] = 0.25
                if 'medium' in model_predictions:
                    weights['medium'] = 0.15
                if 'low' in model_predictions:
                    weights['low'] = 0.1
            elif unknown_ratio > 0.4:
                weights = {'heuristic': 0.35}
                if 'short' in model_predictions:
                    weights['short'] = 0.25
                if 'high' in model_predictions:
                    weights['high'] = 0.1
                if 'medium' in model_predictions:
                    weights['medium'] = 0.2
                if 'low' in model_predictions:
                    weights['low'] = 0.1
            else:
                weights = {'heuristic': 0.4}
                if 'short' in model_predictions:
                    weights['short'] = 0.3
                if 'high' in model_predictions:
                    weights['high'] = 0.05
                if 'medium' in model_predictions:
                    weights['medium'] = 0.15
                if 'low' in model_predictions:
                    weights['low'] = 0.1
            
        else:
            # Use regular models for longer words
            if hasattr(self, 'high_mask_model') and self.high_mask_model is not None:
                model_predictions['high'] = self._get_model_predictions(self.high_mask_model, clean_pattern)
            if hasattr(self, 'medium_mask_model') and self.medium_mask_model is not None:
                model_predictions['medium'] = self._get_model_predictions(self.medium_mask_model, clean_pattern)
            if hasattr(self, 'low_mask_model') and self.low_mask_model is not None:
                model_predictions['low'] = self._get_model_predictions(self.low_mask_model, clean_pattern)
            
            # Weight models based on unknown ratio and available models
            if unknown_ratio > 0.7:
                weights = {'heuristic': 0.3}
                if 'high' in model_predictions:
                    weights['high'] = 0.3
                if 'medium' in model_predictions:
                    weights['medium'] = 0.25
                if 'low' in model_predictions:
                    weights['low'] = 0.15
            elif unknown_ratio > 0.4:
                weights = {'heuristic': 0.35}
                if 'high' in model_predictions:
                    weights['high'] = 0.2
                if 'medium' in model_predictions:
                    weights['medium'] = 0.3
                if 'low' in model_predictions:
                    weights['low'] = 0.15
            else:
                weights = {'heuristic': 0.45}
                if 'high' in model_predictions:
                    weights['high'] = 0.1
                if 'medium' in model_predictions:
                    weights['medium'] = 0.2
                if 'low' in model_predictions:
                    weights['low'] = 0.25
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] /= total_weight
        
        # Combine all scores
        combined_scores = {}
        for letter in string.ascii_lowercase:
            score = weights.get('heuristic', 0.0) * heuristic_scores.get(letter, 0.0)
            
            for model_name, predictions in model_predictions.items():
                if model_name in weights:
                    score += weights[model_name] * predictions.get(letter, 0.0)
            
            combined_scores[letter] = score
        
        # Apply vowel bias if needed
        known_letters = [c for c in clean_pattern if c != '.']
        vowel_ratio = self._calculate_vowel_ratio(known_letters)
        
        # Sort letters by score
        sorted_letters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Choose letter with vowel bias consideration
        if vowel_ratio > 0.6 and unknown_ratio < 0.5:
            # Too many vowels, prefer consonants
            consonants = set(string.ascii_lowercase) - set('aeiou')
            for letter, score in sorted_letters:
                if letter not in self.guessed_letters and letter in consonants:
                    return letter
        
        # Default selection
        for letter, score in sorted_letters:
            if letter not in self.guessed_letters:
                return letter
        
        # Final fallback
        return 'e'

    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
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


def train_models_if_needed():
    """Train transformer models if they don't exist"""
    
    # Load dictionary
    try:
        with open("words_250000_train.txt", "r") as f:
            dictionary = f.read().strip().split()
        dictionary = [word.lower() for word in dictionary if word.isalpha()]
        print(f"Loaded {len(dictionary)} words from dictionary")
    except FileNotFoundError:
        print("Dictionary file not found. Please ensure 'words_250000_train.txt' exists.")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if models already exist
    model_files = [
        "hangman_model_high_mask.pt",
        "hangman_model_medium_mask.pt", 
        "hangman_model_low_mask.pt",
        "hangman_model_short_words.pt"
    ]
    
    all_models_exist = all(os.path.exists(f) for f in model_files)
    
    if not all_models_exist:
        # Train models with different masking probabilities
        model_configs = [
            (0.6, "hangman_model_high_mask.pt", "High masking"),
            (0.35, "hangman_model_medium_mask.pt", "Medium masking"), 
            (0.15, "hangman_model_low_mask.pt", "Low masking")
        ]
        
        print("Training transformer models...")
        for mask_prob, filename, description in model_configs:
            if not os.path.exists(filename):
                print(f"\n{description} (mask_prob={mask_prob})")
                train_transformer_model(
                    dictionary, 
                    mask_prob=mask_prob,
                    model_filename=filename,
                    epochs=100,
                    batch_size=128,
                    learning_rate=0.0001,
                    max_length=15,
                    device=device
                )
            else:
                print(f"{filename} already exists, skipping training.")
        
        # Train short word model with different parameters
        if not os.path.exists("hangman_model_short_words.pt"):
            print(f"\nTraining short word model...")
            short_word_dict = [word for word in dictionary if len(word) <= 7]
            train_transformer_model(
                short_word_dict,
                mask_prob=0.25,
                model_filename="hangman_model_short_words.pt", 
                epochs=150,  # More epochs for short words
                batch_size=128,
                learning_rate=0.0001,
                max_length=7,
                device=device
            )
        else:
            print("hangman_model_short_words.pt already exists, skipping training.")
    else:
        print("All models already exist, skipping training.")
    
    print("\nAll models are ready!")
    return True


def main():
    """Main function to train models and run the hangman game"""
    
    # Train models if needed
    if not train_models_if_needed():
        print("Failed to initialize models. Exiting.")
        return
    
    # Initialize AI player
    print("\nInitializing Hangman AI...")
    api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)
    
    # Get current status
    try:
        status = api.my_status()
        print(f"Current status: {status}")
    except Exception as e:
        print(f"Error getting status: {e}")
    
    # Practice games
    print("\nRunning practice games...")
    practice_wins = 0
    practice_games = 50
    
    for i in range(practice_games):
        try:
            print(f"Practice game {i+1}/{practice_games}")
            result = api.start_game(practice=True, verbose=False)
            if result:
                practice_wins += 1
            
            if (i + 1) % 10 == 0:
                win_rate = practice_wins / (i + 1)
                print(f"Practice progress: {i+1}/{practice_games}, Win rate: {win_rate:.3f}")
                
        except Exception as e:
            print(f"Error in practice game {i+1}: {e}")
            continue
    
    final_practice_rate = practice_wins / practice_games
    print(f"Practice complete: {practice_wins}/{practice_games} wins ({final_practice_rate:.3f})")
    
    # Ask user before starting recorded games
    if final_practice_rate < 0.5:
        print(f"Warning: Practice win rate is {final_practice_rate:.3f}, which is below 50%")
        response = input("Do you want to continue with recorded games? (y/n): ")
        if response.lower() != 'y':
            print("Exiting before recorded games.")
            return
    
    # Final recorded games
    print("\nStarting recorded games...")
    input("Press Enter to start 1000 recorded games (this will count towards your final score)...")
    
    recorded_wins = 0
    for i in range(1000):
        try:
            print(f'Playing recorded game {i+1}/1000')
            result = api.start_game(practice=False, verbose=False)
            if result:
                recorded_wins += 1
            
            # Show progress every 50 games
            if (i + 1) % 50 == 0:
                current_rate = recorded_wins / (i + 1)
                print(f'Progress: {i+1}/1000, Wins: {recorded_wins}, Success rate: {current_rate:.3f}')
                
                # Get updated status from server
                try:
                    status = api.my_status()
                    print(f'Server status: {status}')
                except Exception as e:
                    print(f'Could not get server status: {e}')
            
            # Rate limiting to avoid overwhelming the server
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Error in recorded game {i+1}: {e}")
            continue
    
    # Final results
    final_rate = recorded_wins / 1000
    print(f"\nFinal Results:")
    print(f"Recorded games: 1000, Wins: {recorded_wins}")
    print(f"Final success rate: {final_rate:.3f}")
    
    # Get final server status
    try:
        final_status = api.my_status()
        print(f"Final server status: {final_status}")
    except Exception as e:
        print(f"Could not get final server status: {e}")


if __name__ == "__main__":
    main()