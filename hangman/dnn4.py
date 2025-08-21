import torch
import torch.nn as nn
import torch.optim as optim
print(torch.__version__)
print(torch.utils.data)
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


class DynamicHangmanDataset(Dataset):
    """Simplified dataset with dynamic masking"""
    
    def __init__(self, word_list, sequence_length=15):
        self.words = [word for word in word_list if word.isalpha() and 0 < len(word) <= sequence_length]
        self.seq_len = sequence_length
        
        # Simple vocabulary: padding, mask token, then all lowercase letters
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
    def __len__(self):
        return len(self.words) * 3  # 3 different masking intensities
    
    def _mask_word(self, word, mask_ratio):
        """Apply masking with given ratio"""
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
                # Apply masking with given probability
                if random.random() < mask_ratio:
                    input_sequence.append(self.char_to_id['<MASK>'])
                    target_sequence.append(self.char_to_id[char])
                else:
                    input_sequence.append(self.char_to_id[char])
                    target_sequence.append(-100)  # Don't predict unmasked tokens
        
        return torch.tensor(input_sequence, dtype=torch.long), torch.tensor(target_sequence, dtype=torch.long)
    
    def __getitem__(self, index):
        word_idx = index // 3
        mask_type = index % 3
        
        word = self.words[word_idx]
        
        # Three masking intensities
        mask_ratios = [0.3, 0.5, 0.7]  # light, medium, heavy
        mask_ratio = mask_ratios[mask_type]
        
        return self._mask_word(word, mask_ratio)


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, embedding_dim, max_sequence_length=50):
        super().__init__()
        
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


class DynamicHangmanTransformer(nn.Module):
    """Simplified transformer with dynamic prediction modes"""
    
    def __init__(self, vocab_size=28, embedding_dim=128, num_heads=4, 
                 num_layers=4, feedforward_dim=256, max_length=50, dropout=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoder = PositionalEncoder(embedding_dim, max_length)
        
        # Simplified transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Four prediction modes
        self.prediction_heads = nn.ModuleDict({
            'early_game': nn.Linear(embedding_dim, vocab_size),      # Focus on common letters
            'mid_game': nn.Linear(embedding_dim, vocab_size),        # Balanced approach
            'late_game': nn.Linear(embedding_dim, vocab_size),       # Pattern-based
            'adaptive': nn.Linear(embedding_dim, vocab_size)         # Dynamic weighting
        })
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, token_ids, mode='adaptive'):
        # Embed tokens and add positional encoding
        embeddings = self.token_embedding(token_ids)
        x = self.positional_encoder(embeddings)
        x = self.dropout(x)
        
        # Create attention mask (ignore padding)
        padding_mask = (token_ids == 0)  # Assuming <PAD> has id 0
        
        # Pass through transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)
        
        # Get predictions from specified head
        if mode in self.prediction_heads:
            return self.prediction_heads[mode](x)
        else:
            # Default to adaptive mode
            return self.prediction_heads['adaptive'](x)


def train_dynamic_model(word_dictionary, model_filename, 
                       epochs=100, batch_size=32, learning_rate=0.001, 
                       max_length=15, device='cpu'):
    """Train the simplified dynamic model"""
    
    print(f"Training dynamic model: {model_filename}")
    
    # Create dataset and dataloader
    dataset = DynamicHangmanDataset(word_dictionary, sequence_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = DynamicHangmanTransformer(
        vocab_size=len(dataset.vocabulary),
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        feedforward_dim=256,
        max_length=max_length + 5,
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
            
            # Train all prediction heads
            total_loss = 0
            modes = ['early_game', 'mid_game', 'late_game', 'adaptive']
            
            for mode in modes:
                logits = model(input_ids, mode=mode)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                total_loss += loss
            
            # Average loss across all modes
            total_loss = total_loss / len(modes)
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
        
        scheduler.step()
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Save model
    torch.save(model.state_dict(), model_filename)
    print(f"Dynamic model saved as {model_filename}")
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


class SimplifiedHangmanAI:
    """Simplified AI player with dynamic model selection"""
    
    def __init__(self, access_token=None, session=None, timeout=None):
        self.hangman_url = self._determine_best_server()
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        # Load dictionary
        self.word_dictionary = self._load_dictionary("words_250000_train.txt")
        self.letter_frequencies = self._calculate_letter_frequencies()
        
        # Initialize working dictionary
        self.current_dictionary = self.word_dictionary[:]
        
        # Setup device and vocabulary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.vocabulary = ['<PAD>', '<MASK>'] + list(string.ascii_lowercase)
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Initialize the dynamic model
        self._initialize_dynamic_model()

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
            return ['apple', 'banana', 'cherry', 'dog', 'elephant']

    def _calculate_letter_frequencies(self):
        """Calculate letter frequency distribution from dictionary"""
        all_letters = ''.join(self.word_dictionary)
        return collections.Counter(all_letters).most_common()

    def _initialize_dynamic_model(self):
        """Initialize and load the dynamic transformer model"""
        self.dynamic_model = DynamicHangmanTransformer(
            vocab_size=len(self.vocabulary),
            embedding_dim=128,
            num_heads=4,
            num_layers=4,
            feedforward_dim=256,
            max_length=20,
            dropout=0.1
        ).to(self.device)
        
        try:
            self.dynamic_model.load_state_dict(torch.load('hangman_dynamic_model.pt', 
                                                        map_location=self.device, weights_only=True))
            self.dynamic_model.eval()
            print("Loaded dynamic model: hangman_dynamic_model.pt")
        except FileNotFoundError:
            print("Warning: hangman_dynamic_model.pt not found. Model will be trained.")
            self.dynamic_model = None

    def _filter_by_pattern(self, pattern):
        """Filter dictionary by regex pattern"""
        regex_pattern = "^" + pattern + "$"
        filtered_words = []
        
        for word in self.current_dictionary:
            if len(word) == len(pattern) and re.fullmatch(regex_pattern, word):
                filtered_words.append(word)
        
        return filtered_words

    def _get_model_predictions(self, word_pattern, game_stage):
        """Get predictions from dynamic model based on game stage"""
        if self.dynamic_model is None:
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
        
        # Pad to minimum length
        while len(model_input) < 15:
            model_input.append('<PAD>')
        
        # Convert to tensor
        input_ids = [self.char_to_id.get(char, self.char_to_id['<PAD>']) for char in model_input]
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            # Select model mode based on game stage
            if game_stage == 'early':
                logits = self.dynamic_model(input_tensor, mode='early_game')
            elif game_stage == 'mid':
                logits = self.dynamic_model(input_tensor, mode='mid_game')
            elif game_stage == 'late':
                logits = self.dynamic_model(input_tensor, mode='late_game')
            else:  # adaptive
                logits = self.dynamic_model(input_tensor, mode='adaptive')
            
            # Find mask positions and calculate probabilities
            mask_positions = [i for i, char in enumerate(model_input) if char == '<MASK>']
            letter_probabilities = collections.Counter()
            
            if mask_positions:
                for position in mask_positions:
                    if position < logits.size(1):
                        position_logits = logits[0, position]
                        position_probs = torch.softmax(position_logits, dim=0)
                        
                        # Skip <PAD> and <MASK> tokens (indices 0 and 1)
                        for letter_idx in range(2, min(28, len(self.id_to_char))):
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

    def _determine_game_stage(self, word_pattern):
        """Determine what stage of the game we're in"""
        unknown_count = word_pattern.count('.')
        total_length = len(word_pattern)
        unknown_ratio = unknown_count / total_length if total_length > 0 else 0
        
        if unknown_ratio > 0.7:
            return 'early'
        elif unknown_ratio > 0.3:
            return 'mid'
        else:
            return 'late'

    def make_guess(self, word_state):
        """Make an intelligent guess using the dynamic model"""
        # Parse word state
        clean_pattern = word_state[::2].replace("_", ".")
        
        # Determine game stage
        game_stage = self._determine_game_stage(clean_pattern)
        
        # Update current dictionary based on pattern
        self.current_dictionary = self._filter_by_pattern(clean_pattern)
        
        # Get dictionary-based frequencies
        if self.current_dictionary:
            dict_frequencies = collections.Counter()
            for word in self.current_dictionary:
                for letter in set(word):  # Count each letter once per word
                    dict_frequencies[letter] += 1
            
            # Normalize
            total_dict = sum(dict_frequencies.values())
            dict_scores = {letter: count / total_dict for letter, count in dict_frequencies.items()}
        else:
            # Fallback to global frequencies
            dict_scores = {letter: count / sum(count for _, count in self.letter_frequencies) 
                          for letter, count in self.letter_frequencies}
        
        # Get model predictions
        model_scores = self._get_model_predictions(clean_pattern, game_stage)
        
        # Combine scores with adaptive weighting
        if game_stage == 'early':
            # Early game: rely more on model (common patterns)
            dict_weight, model_weight = 0.3, 0.7
        elif game_stage == 'mid':
            # Mid game: balanced approach
            dict_weight, model_weight = 0.5, 0.5
        else:
            # Late game: rely more on dictionary filtering
            dict_weight, model_weight = 0.7, 0.3
        
        # Combine scores
        combined_scores = {}
        for letter in string.ascii_lowercase:
            dict_score = dict_scores.get(letter, 0.0)
            model_score = model_scores.get(letter, 0.0)
            combined_scores[letter] = dict_weight * dict_score + model_weight * model_score
        
        # Sort and select best unguessed letter
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
    """Main function to train the dynamic model and run games"""
    
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
    
    # Check if dynamic model already exists
    model_path = "hangman_dynamic_model.pt"
    
    try:
        # Try to load existing model
        test_model = DynamicHangmanTransformer(
            vocab_size=28,
            embedding_dim=64,
            num_heads=2,
            num_layers=2,
            feedforward_dim=128,
            max_length=20,
            dropout=0.1
        ).to(device)
        
        test_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Dynamic model '{model_path}' already exists and loaded successfully.")
        
    except FileNotFoundError:
        print(f"Dynamic model '{model_path}' not found. Training new model...")
        
        # Train the dynamic transformer model
        train_dynamic_model(
            dictionary,
            model_filename=model_path,
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            max_length=15,
            device=device
        )
        print("Dynamic model training completed!")
    
    # Initialize Simplified Hangman AI
    print("\nInitializing Simplified Hangman AI...")
    api = SimplifiedHangmanAI(access_token="e8706baea67230c5106bf22704c839", timeout=200)
    
    # Get current status
    try:
        status = api.get_status()
        print(f"Current status: {status}")
    except Exception as e:
        print(f"Could not get status: {e}")
    
    # Practice games
    print("\nRunning practice games...")
    practice_wins = 0
    practice_games = 20
    
    for i in range(practice_games):
        try:
            result = api.start_game(practice=True, verbose=True)
            if result:
                practice_wins += 1
            
            print(f"Practice game {i+1}: {'Won' if result else 'Lost'}")
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in practice game {i+1}: {e}")
            continue
    
    final_practice_rate = practice_wins / practice_games
    print(f"\nPractice Results: {practice_wins}/{practice_games} wins ({final_practice_rate:.3f})")
    
    # Ask user if they want to proceed with recorded games
    if final_practice_rate > 0.6:
        print("Good practice performance! Ready for recorded games.")
    else:
        print(f"Practice win rate: {final_practice_rate:.3f}")
        # proceed = input("Do you want to proceed with recorded games? (y/n): ").lower().strip()
        # if proceed != 'y':
        #     print("Stopping before recorded games.")
        #     return
    
    # Final recorded games
    # print("\nStarting recorded games...")
    # input("Press Enter to start 1000 recorded games...")
    
    # recorded_wins = 0
    
    # for i in range(1000):
    #     try:
    #         if (i + 1) % 50 == 0:
    #             print(f'Playing recorded game {i+1}/1000')
            
    #         result = api.start_game(practice=False, verbose=False)
    #         if result:
    #             recorded_wins += 1
            
    #         if (i + 1) % 100 == 0:
    #             current_rate = recorded_wins / (i + 1)
    #             print(f'Progress: {i+1}/1000, Wins: {recorded_wins}, Rate: {current_rate:.3f}')
            
    #         time.sleep(0.3)
            
    #     except Exception as e:
    #         print(f"Error in recorded game {i+1}: {e}")
    #         continue
    
    # Final results
    print(f"\nFinal Results:")
    print(f"Practice: {practice_wins}/{practice_games} ({final_practice_rate:.3f})")
    print(f"Recorded: {recorded_wins}/1000 ({recorded_wins/1000:.3f})")
    
    try:
        final_status = api.get_status()
        print(f"Final status: {final_status}")
    except Exception as e:
        print(f"Could not get final status: {e}")


if __name__ == "__main__":
    main()