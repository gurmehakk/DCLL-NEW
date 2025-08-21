import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import random
import string
import json
import requests
import time
import re
import collections
from urllib.parse import parse_qs
from requests.packages.urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class AdaptiveHangmanDataset(Dataset):
    """Enhanced dataset with multiple masking strategies"""
    
    def __init__(self, word_list, sequence_length=15):
        self.words = [word for word in word_list if word.isalpha() and 0 < len(word) <= sequence_length]
        self.seq_len = sequence_length
        
        # Create vocabulary: padding, mask token, then all lowercase letters
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Define 5 different masking strategies
        self.masking_strategies = {
            'aggressive': {'base_prob': 0.7, 'vowel_boost': 0.2, 'consonant_boost': 0.1},
            'balanced': {'base_prob': 0.4, 'vowel_boost': 0.15, 'consonant_boost': 0.1},
            'conservative': {'base_prob': 0.15, 'vowel_boost': 0.1, 'consonant_boost': 0.05},
            'vowel_focused': {'base_prob': 0.3, 'vowel_boost': 0.4, 'consonant_boost': 0.0},
            'pattern_aware': {'base_prob': 0.25, 'vowel_boost': 0.15, 'consonant_boost': 0.2}
        }
        
    def __len__(self):
        return len(self.words) * len(self.masking_strategies)

    def _apply_masking_strategy(self, word, strategy_name):
        """Apply specific masking strategy to a word"""
        strategy = self.masking_strategies[strategy_name]
        characters = list(word.lower())
        vowels = set('aeiou')
        
        # Pad sequence if necessary
        if len(characters) < self.seq_len:
            characters += ['<PAD>'] * (self.seq_len - len(characters))
        
        input_sequence = []
        target_sequence = []
        
        for i, char in enumerate(characters):
            if char == '<PAD>':
                input_sequence.append(self.char_to_id['<PAD>'])
                target_sequence.append(-100)  # Ignore padding in loss
            else:
                # Calculate masking probability based on strategy
                mask_prob = strategy['base_prob']
                
                if char in vowels:
                    mask_prob += strategy['vowel_boost']
                else:
                    mask_prob += strategy['consonant_boost']
                
                # Pattern-aware strategy: mask more letters near word boundaries
                if strategy_name == 'pattern_aware':
                    if i == 0 or i == len([c for c in characters if c != '<PAD>']) - 1:
                        mask_prob += 0.2  # Boost first/last letters
                
                # Apply masking
                if random.random() < min(mask_prob, 0.95):  # Cap at 95%
                    input_sequence.append(self.char_to_id['<MASK>'])
                    target_sequence.append(self.char_to_id[char])
                else:
                    input_sequence.append(self.char_to_id[char])
                    target_sequence.append(-100)  # Don't predict unmasked tokens
        
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)

    def __getitem__(self, index):
        word_idx = index // len(self.masking_strategies)
        strategy_idx = index % len(self.masking_strategies)
        
        word = self.words[word_idx]
        strategy_name = list(self.masking_strategies.keys())[strategy_idx]
        
        return self._apply_masking_strategy(word, strategy_name)


class MultiHeadAttentionWithContext(nn.Module):
    """Enhanced attention mechanism with context awareness"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        output = self.w_o(context)
        return self.layer_norm(output + x)


class EnhancedHangmanTransformer(nn.Module):
    """Enhanced transformer with multiple prediction heads"""
    
    def __init__(self, vocab_size=28, embedding_dim=256, num_heads=8, 
                 num_layers=8, feedforward_dim=1024, max_length=100, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoder = PositionalEncoder(embedding_dim, max_length)
        
        # Enhanced transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttentionWithContext(embedding_dim, num_heads, dropout),
                'feedforward': nn.Sequential(
                    nn.Linear(embedding_dim, feedforward_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feedforward_dim, embedding_dim),
                    nn.Dropout(dropout)
                ),
                'norm': nn.LayerNorm(embedding_dim)
            }) for _ in range(num_layers)
        ])
        
        # Multiple prediction heads for different strategies
        self.prediction_heads = nn.ModuleDict({
            'primary': nn.Linear(embedding_dim, vocab_size),
            'vowel_focused': nn.Linear(embedding_dim, vocab_size),
            'consonant_focused': nn.Linear(embedding_dim, vocab_size),
            'pattern_aware': nn.Linear(embedding_dim, vocab_size),
            'frequency_based': nn.Linear(embedding_dim, vocab_size)
        })
        
        # Context aggregation layer
        self.context_aggregator = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, token_ids, prediction_mode='ensemble'):
        batch_size, seq_len = token_ids.shape
        
        # Embed tokens and add positional encoding
        embeddings = self.token_embedding(token_ids)
        x = self.positional_encoder(embeddings)
        x = self.dropout(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            # Self-attention
            attended = layer['attention'](x)
            
            # Feedforward
            ff_output = layer['feedforward'](attended)
            x = layer['norm'](ff_output + attended)
        
        # Aggregate context (mean and max pooling)
        mean_context = x.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        max_context, _ = x.max(dim=1, keepdim=True)
        max_context = max_context.expand(-1, seq_len, -1)
        
        # Combine contexts
        combined_context = torch.cat([x, mean_context], dim=-1)
        contextualized = self.context_aggregator(combined_context)
        
        # Generate predictions from different heads
        predictions = {}
        for head_name, head in self.prediction_heads.items():
            predictions[head_name] = head(contextualized)
        
        if prediction_mode == 'ensemble':
            # Ensemble prediction with learned weights
            ensemble_weights = torch.softmax(torch.tensor([0.3, 0.2, 0.2, 0.15, 0.15]), dim=0)
            ensemble_logits = sum(weight * pred for weight, pred in 
                                zip(ensemble_weights, predictions.values()))
            return ensemble_logits
        elif prediction_mode in predictions:
            return predictions[prediction_mode]
        else:
            return predictions['primary']


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


def train_enhanced_transformer(word_dictionary, model_filename, 
                             epochs=150, batch_size=64, learning_rate=0.0001, 
                             max_length=15, device='cpu'):
    """Train the enhanced transformer model"""
    
    print(f"Training enhanced model: {model_filename}")
    
    # Create dataset and dataloader
    dataset = AdaptiveHangmanDataset(word_dictionary, sequence_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = EnhancedHangmanTransformer(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=256,
        num_heads=8,
        num_layers=8,
        feedforward_dim=1024,
        max_length=20,
        dropout=0.1
    ).to(device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        for input_ids, targets in dataloader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Train with ensemble mode
            logits = model(input_ids, prediction_mode='ensemble')
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 15 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model
    torch.save(model.state_dict(), model_filename)
    print(f"Enhanced model saved as {model_filename}")
    return model


class HangmanAPIError(Exception):
    """Custom exception for Hangman API errors"""
    
    def __init__(self, result):
        self.result = result
        self.code = None
        
        try:
            self.error_type = result["error_code"]
        except (KeyError, TypeError):
            self.error_type = ""
        
        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.error_type:
                    self.error_type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = str(result)
        
        super().__init__(self.message)


class EnhancedHangmanAI:
    """Enhanced AI player using single model with multiple prediction strategies"""
    
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self._determine_best_server()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        # Load dictionary
        self.word_dictionary = self._load_dictionary("words_250000_train.txt")
        self.letter_frequencies = self._calculate_letter_frequencies()
        
        # Initialize working dictionary and substring patterns
        self.current_dictionary = self.word_dictionary[:]
        self.substring_dictionary = self._build_substring_patterns()
        
        # Setup device and vocabulary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Initialize the enhanced model
        self._initialize_enhanced_model()

    def _determine_best_server(self):
        """Find the fastest server endpoint"""
        server_urls = ['https://trexsim.com', 'https://sg.trexsim.com']
        response_times = {}
        
        for url in server_urls:
            try:
                start_time = time.time()
                requests.get(url, timeout=5)
                response_times[url] = time.time() - start_time
            except:
                response_times[url] = float('inf')
        
        best_server = min(response_times.items(), key=lambda x: x[1])[0]
        return best_server + '/trexsim/hangman'

    def _load_dictionary(self, filename):
        """Load and preprocess word dictionary"""
        try:
            with open(filename, "r") as file:
                words = file.read().strip().split()
            return [word.lower() for word in words if word.isalpha()]
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using fallback dictionary.")
            return ['apple', 'banana', 'cherry', 'dog', 'elephant']  # Fallback

    def _calculate_letter_frequencies(self):
        """Calculate letter frequency distribution from dictionary"""
        all_letters = ''.join(self.word_dictionary)
        return collections.Counter(all_letters).most_common()

    def _build_substring_patterns(self):
        """Build dictionary of substring patterns for pattern matching"""
        max_word_length = max(len(word) for word in self.word_dictionary)
        substring_dict = {length: [] for length in range(3, min(max_word_length, 30) + 1)}
        
        for length in range(3, min(max_word_length, 30) + 1):
            for word in self.word_dictionary:
                if len(word) >= length:
                    for i in range(len(word) - length + 1):
                        substring_dict[length].append(word[i:i + length])
        
        return substring_dict

    def _initialize_enhanced_model(self):
        """Initialize and load the enhanced transformer model"""
        self.enhanced_model = EnhancedHangmanTransformer(
            vocab_size=len(self.vocabulary),
            embedding_dim=256,
            num_heads=8,
            num_layers=8,
            feedforward_dim=1024,
            max_length=20,
            dropout=0.1
        ).to(self.device)
        
        try:
            self.enhanced_model.load_state_dict(torch.load('hangman_enhanced_model.pt', 
                                                         map_location=self.device, weights_only=True))
            self.enhanced_model.eval()
            print("Loaded enhanced model: hangman_enhanced_model.pt")
        except FileNotFoundError:
            print("Warning: hangman_enhanced_model.pt not found. Model will be trained.")
            self.enhanced_model = None

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

    def _get_enhanced_model_predictions(self, word_pattern):
        """Get predictions from enhanced model using different strategies"""
        if self.enhanced_model is None:
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
        model_max_length = self.enhanced_model.positional_encoder.position_encoding.size(1)
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length]
        
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Get predictions from different heads
            predictions = {}
            prediction_modes = ['ensemble', 'vowel_focused', 'consonant_focused', 'pattern_aware', 'frequency_based']
            
            for mode in prediction_modes:
                logits = self.enhanced_model(input_tensor, prediction_mode=mode)
                
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
                
                predictions[mode] = dict(letter_probabilities)
        
        return predictions

    def make_guess(self, word_state):
        """Make an intelligent guess using the enhanced model"""
        # Parse word state
        clean_pattern = word_state[::2].replace("_", ".")
        word_length = len(clean_pattern)
        unknown_count = clean_pattern.count('.')
        unknown_ratio = unknown_count / word_length if word_length > 0 else 0.0
        
        # Update current dictionary based on pattern
        self.current_dictionary = self._filter_by_pattern(clean_pattern)
        
        # Get heuristic letter frequencies
        if self.current_dictionary:
            heuristic_frequencies = self._get_character_frequencies(self.current_dictionary)
        else:
            # Fallback to overall frequencies
            heuristic_frequencies = collections.Counter(dict(self.letter_frequencies))
        
        # Normalize heuristic scores
        total_heuristic = sum(heuristic_frequencies.values())
        heuristic_scores = {}
        if total_heuristic > 0:
            for letter, count in heuristic_frequencies.items():
                heuristic_scores[letter] = count / total_heuristic
        
        # Get enhanced model predictions
        model_predictions = self._get_enhanced_model_predictions(clean_pattern)
        
        # Adaptive weighting based on game state
        if unknown_ratio > 0.8:
            # Very early game - rely more on frequency-based and pattern-aware
            weights = {
                'heuristic': 0.25,
                'ensemble': 0.20,
                'frequency_based': 0.25,
                'pattern_aware': 0.15,
                'vowel_focused': 0.10,
                'consonant_focused': 0.05
            }
        elif unknown_ratio > 0.5:
            # Mid game - balanced approach
            weights = {
                'heuristic': 0.35,
                'ensemble': 0.25,
                'frequency_based': 0.15,
                'pattern_aware': 0.15,
                'vowel_focused': 0.05,
                'consonant_focused': 0.05
            }
        else:
            # Late game - rely more on heuristics and ensemble
            weights = {
                'heuristic': 0.45,
                'ensemble': 0.30,
                'frequency_based': 0.10,
                'pattern_aware': 0.10,
                'vowel_focused': 0.03,
                'consonant_focused': 0.02
            }
        
        # Combine all scores
        combined_scores = {}
        for letter in string.ascii_lowercase:
            score = weights['heuristic'] * heuristic_scores.get(letter, 0.0)
            
            for pred_type, predictions in model_predictions.items():
                if pred_type in weights:
                    score += weights[pred_type] * predictions.get(letter, 0.0)
            
            combined_scores[letter] = score
        
        # Sort letters by score and select best unguessed letter
        sorted_letters = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        for letter, score in sorted_letters:
            if letter not in self.guessed_letters:
                return letter
        
        # Fallback
        return 'e'

    def start_game(self, practice=True, verbose=True):
        """Start and play a complete Hangman game"""
        self.guessed_letters = []
        self.current_dictionary = self.word_dictionary.copy()
        
        try:
            response = self._make_request("/new_game", {"practice": practice})
            
            if response.get('status') == "approved":
                game_id = response.get('game_id')
                word_state = response.get('word')
                tries_remaining = response.get('tries_remains')
                
                if verbose:
                    print(f"Game {game_id} started. Tries: {tries_remaining}, Word: {word_state}")
                
                while tries_remaining > 0:
                    guess_letter = self.make_guess(word_state)
                    self.guessed_letters.append(guess_letter)
                    
                    if verbose:
                        print(f"Guessing: {guess_letter}")
                    
                    try:
                        response = self._make_request("/guess_letter", {
                            "request": "guess_letter",
                            "game_id": game_id,
                            "letter": guess_letter
                        })
                        
                        if verbose:
                            print(f"Server response: {response}")
                        
                        status = response.get('status')
                        tries_remaining = response.get('tries_remains')
                        
                        if status == "success":
                            if verbose:
                                print(f"Game {game_id} won!")
                            return True
                        elif status == "failed":
                            reason = response.get('reason', 'No tries left')
                            if verbose:
                                print(f"Game {game_id} lost. Reason: {reason}")
                            return False
                        elif status == "ongoing":
                            word_state = response.get('word')
                        
                    except HangmanAPIError as e:
                        print(f"API Error: {e}")
                        continue
                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        raise
            else:
                if verbose:
                    print("Failed to start game")
                return False
                
        except Exception as e:
            print(f"Error starting game: {e}")
            return False

    def get_status(self):
        """Get current game statistics"""
        return self._make_request("/my_status", {})

    def _make_request(self, endpoint, parameters=None, post_data=None, method=None):
        """Make HTTP request to Hangman API with retry logic"""
        if parameters is None:
            parameters = {}
        
        if post_data is not None:
            method = "POST"
        
        # Add access token
        if self.access_token:
            if post_data and "access_token" not in post_data:
                post_data["access_token"] = self.access_token
            elif "access_token" not in parameters:
                parameters["access_token"] = self.access_token
        
        # Rate limiting
        time.sleep(0.2)
        
        max_retries = 50
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + endpoint,
                    timeout=self.timeout,
                    params=parameters,
                    data=post_data,
                    verify=False
                )
                response.raise_for_status()
                break
                
            except requests.HTTPError as e:
                try:
                    error_response = e.response.json()
                except ValueError:
                    error_response = {"error_msg": str(e)}
                raise HangmanAPIError(error_response)
                
            except requests.exceptions.SSLError:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(retry_delay)
                
            except requests.exceptions.RequestException:
                if attempt + 1 == max_retries:
                    raise
                time.sleep(retry_delay)
        
        # Parse response
        headers = response.headers
        if 'json' in headers.get('content-type', ''):
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_params = parse_qs(response.text)
            if "access_token" in query_params:
                result = {"access_token": query_params["access_token"][0]}
                if "expires" in query_params:
                    result["expires"] = query_params["expires"][0]
            else:
                try:
                    result = response.json()
                except ValueError:
                    result = {'error_msg': response.text}
                raise HangmanAPIError(result)
        else:
            raise HangmanAPIError('Invalid response format')
        
        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        
        return result
    
def main():
    """Main function to train the enhanced model and run games"""
    
    # Load dictionary
    try:
        with open("words_250000_train.txt", "r") as f:
            dictionary = f.read().strip().split()
        dictionary = [word.lower() for word in dictionary if word.isalpha()]
        print(f"Loaded {len(dictionary)} words from dictionary")
    except FileNotFoundError:
        print("Dictionary file not found. Please ensure 'words_250000_train.txt' exists.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if enhanced model already exists
    model_path = "hangman_enhanced_model.pt"
    
    try:
        # Try to load existing model
        test_model = EnhancedHangmanTransformer(
            vocab_size=28,  # '<PAD>', '<MASK>' + 26 letters
            embedding_dim=256,
            num_heads=8,
            num_layers=8,
            feedforward_dim=1024,
            max_length=20,
            dropout=0.1
        ).to(device)
        
        test_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Enhanced model '{model_path}' already exists and loaded successfully.")
        
    except FileNotFoundError:
        print(f"Enhanced model '{model_path}' not found. Training new model...")
        
        # Train the enhanced transformer model
        train_enhanced_transformer(
            dictionary,
            model_filename=model_path,
            epochs=150,
            batch_size=64,
            learning_rate=0.0001,
            max_length=15,
            device=device
        )
        print("Enhanced model training completed!")
    
    # Initialize Enhanced Hangman AI
    print("\nInitializing Enhanced Hangman AI...")
    api = EnhancedHangmanAI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)
    
    # Get current status
    try:
        status = api.get_status()
        print(f"Current status: {status}")
    except Exception as e:
        print(f"Could not get status: {e}")
    
    # Practice games
    print("\nRunning practice games...")
    practice_wins = 0
    practice_games = 50
    
    for i in range(practice_games):
        try:
            result = api.start_game(practice=True, verbose=True)
            if result:
                practice_wins += 1
            
            if (i + 1) % 10 == 0:
                win_rate = practice_wins / (i + 1)
                print(f"\nPractice progress: {i+1}/{practice_games}, Win rate: {win_rate:.3f}")
            
            # Small delay between games
            time.sleep(0.3)
            
        except Exception as e:
            print(f"Error in practice game {i+1}: {e}")
            continue
    
    final_practice_rate = practice_wins / practice_games
    print(f"\nPractice complete: {practice_wins}/{practice_games} wins ({final_practice_rate:.3f})")
    
    # Ask user if they want to proceed with recorded games
    if final_practice_rate < 0.5:
        print(f"Warning: Practice win rate is {final_practice_rate:.3f}, which is below 50%")
        proceed = input("Do you want to proceed with recorded games? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Stopping before recorded games.")
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
                print(f'Progress: {i+1}/1000 games, Current wins: {recorded_wins}, Win rate: {current_rate:.3f}')
                
                # Get server status if possible
                try:
                    status = api.get_status()
                    print(f'Server status: {status}')
                except Exception as e:
                    print(f'Could not get server status: {e}')
            
            # Rate limiting to avoid overwhelming the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in recorded game {i+1}: {e}")
            # Continue with next game
            continue
    
    # Final results
    print(f"\nFinal Results:")
    print(f"Practice games: {practice_games}, Wins: {practice_wins}, Rate: {final_practice_rate:.3f}")
    print(f"Recorded games: 1000, Wins: {recorded_wins}, Rate: {recorded_wins/1000:.3f}")
    
    # Try to get final server status
    try:
        final_status = api.get_status()
        print(f"Final server status: {final_status}")
    except Exception as e:
        print(f"Could not get final server status: {e}")


if __name__ == "__main__":
    main()
    