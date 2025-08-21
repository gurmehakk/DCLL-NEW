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

# In[ ]:


import json
import requests
import random
import string
import secrets
import time
import re
import collections

try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


# In[ ]:


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
        
        self.current_dictionary = []
        self.initialize_all_algorithms()
        print("All algorithms initialized successfully")
        
        
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
        try:
            with open(dictionary_file_location, "r") as text_file:
                full_dictionary = text_file.read().splitlines()
            return full_dictionary
        except:
            return ["apple", "banana", "cherry", "grape", "orange", "python", "machine", "learning", "algorithm", "neural"]

    def initialize_all_algorithms(self):
        """Initialize data structures for all algorithms"""
        self.initialize_basic_data()
        self.initialize_ml_models()
        self.initialize_neural_networks()
    
    def initialize_basic_data(self):
        """Initialize basic data structures"""
        # Position-based frequency
        self.position_frequency = defaultdict(lambda: defaultdict(dict))
        for word in self.full_dictionary:
            for pos, char in enumerate(word):
                word_len = len(word)
                if pos not in self.position_frequency[word_len]:
                    self.position_frequency[word_len][pos] = defaultdict(int)
                self.position_frequency[word_len][pos][char] += 1
        
        # N-gram analysis
        self.trigrams = defaultdict(int)
        self.bigrams = defaultdict(int)
        for word in self.full_dictionary:
            for i in range(len(word) - 2):
                self.trigrams[word[i:i+3]] += 1
            for i in range(len(word) - 1):
                self.bigrams[word[i:i+2]] += 1
        
        # Word patterns
        self.word_patterns = defaultdict(list)
        for word in self.full_dictionary:
            pattern = self.get_word_pattern(word)
            self.word_patterns[pattern].append(word)
    
    def get_word_pattern(self, word):
        """Convert word to pattern (e.g., 'hello' -> '12334')"""
        char_map = {}
        pattern = []
        next_num = 1
        for char in word:
            if char not in char_map:
                char_map[char] = next_num
                next_num += 1
            pattern.append(str(char_map[char]))
        return ''.join(pattern)
    
    def initialize_ml_models(self):
        """Initialize traditional ML models"""
        # Prepare training data for ML models
        self.prepare_ml_training_data()
        
        # Initialize models
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingClassifier(random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.svm_model = SVC(probability=True, random_state=42)
        self.nb_model = MultinomialNB()
        self.xgb_model = xgb.XGBClassifier(random_state=42)
        self.lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        
        # Train models
        self.train_ml_models()
    
    def prepare_ml_training_data(self):
        """Prepare training data for ML models"""
        X_features = []
        y_labels = []
        
        for word in self.full_dictionary[:10000]:  # Use subset for training
            for i, target_letter in enumerate(word):
                # Create partial word (simulate hangman state)
                revealed_positions = random.sample(range(len(word)), 
                                                 random.randint(0, len(word)//2))
                partial_word = ['_'] * len(word)
                revealed_letters = set()
                
                for pos in revealed_positions:
                    partial_word[pos] = word[pos]
                    revealed_letters.add(word[pos])
                
                # Extract features
                features = self.extract_ml_features(''.join(partial_word), revealed_letters, len(word))
                X_features.append(features)
                y_labels.append(target_letter)
        
        self.X_train = np.array(X_features)
        self.y_train = np.array(y_labels)
        self.label_encoder = LabelEncoder()
        self.y_train_encoded = self.label_encoder.fit_transform(self.y_train)
    
    def extract_ml_features(self, partial_word, guessed_letters, word_length):
        """Extract features for ML models"""
        features = []
        
        # Basic features
        features.append(word_length)
        features.append(len(guessed_letters))
        features.append(partial_word.count('_'))
        
        # Position features
        for i in range(10):  # Max word length consideration
            if i < len(partial_word):
                features.append(1 if partial_word[i] != '_' else 0)
            else:
                features.append(0)
        
        # Letter frequency features
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            features.append(1 if letter in guessed_letters else 0)
        
        # Pattern features
        vowel_count = sum(1 for c in partial_word if c in 'aeiou' and c != '_')
        consonant_count = len([c for c in partial_word if c not in 'aeiou_'])
        features.extend([vowel_count, consonant_count])
        
        return features
    
    def train_ml_models(self):
        """Train all ML models"""
        try:
            self.rf_model.fit(self.X_train, self.y_train_encoded)
            self.gb_model.fit(self.X_train, self.y_train_encoded)
            self.lr_model.fit(self.X_train, self.y_train_encoded)
            # self.svm_model.fit(self.X_train, self.y_train_encoded)  # Commented due to memory
            self.nb_model.fit(self.X_train, self.y_train_encoded)
            self.xgb_model.fit(self.X_train, self.y_train_encoded)
            self.lgb_model.fit(self.X_train, self.y_train_encoded)
            print("ML models trained successfully")
        except Exception as e:
            print(f"Error training ML models: {e}")
    
    def initialize_neural_networks(self):
        """Initialize neural network models"""
        # Prepare data for neural networks
        self.prepare_nn_data()
        
        # Build models
        self.build_lstm_model()
        self.build_gru_model()
        self.build_bilstm_model()
        self.build_rnn_model()
    
    def prepare_nn_data(self):
        """Prepare data for neural networks"""
        # Create character-level tokenizer
        self.tokenizer = Tokenizer(char_level=True)
        self.tokenizer.fit_on_texts(self.full_dictionary)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        # Prepare training sequences
        sequences = []
        targets = []
        
        for word in self.full_dictionary[:5000]:  # Use subset
            seq = self.tokenizer.texts_to_sequences([word])[0]
            for i in range(1, len(seq)):
                sequences.append(seq[:i])
                targets.append(seq[i])
        
        self.max_len = 20
        self.X_seq = pad_sequences(sequences, maxlen=self.max_len)
        self.y_seq = np.array(targets) - 1  # Adjust for 0-indexing
    
    def build_lstm_model(self):
        """Build LSTM model"""
        self.lstm_model = Sequential([
            Embedding(self.vocab_size, 50, input_length=self.max_len),
            LSTM(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.lstm_model.compile(loss='sparse_categorical_crossentropy', 
                               optimizer='adam', metrics=['accuracy'])
    
    def build_gru_model(self):
        """Build GRU model"""
        self.gru_model = Sequential([
            Embedding(self.vocab_size, 50, input_length=self.max_len),
            GRU(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.gru_model.compile(loss='sparse_categorical_crossentropy', 
                              optimizer='adam', metrics=['accuracy'])
    
    def build_bilstm_model(self):
        """Build Bidirectional LSTM model"""
        self.bilstm_model = Sequential([
            Embedding(self.vocab_size, 50, input_length=self.max_len),
            Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.bilstm_model.compile(loss='sparse_categorical_crossentropy', 
                                 optimizer='adam', metrics=['accuracy'])
    
    def build_rnn_model(self):
        """Build simple RNN model"""
        self.rnn_model = Sequential([
            Embedding(self.vocab_size, 50, input_length=self.max_len),
            tf.keras.layers.SimpleRNN(100, dropout=0.2, recurrent_dropout=0.2),
            Dense(self.vocab_size, activation='softmax')
        ])
        self.rnn_model.compile(loss='sparse_categorical_crossentropy', 
                              optimizer='adam', metrics=['accuracy'])

    # ALGORITHM 1: Random Forest Classifier
    def algo1_random_forest(self, word):
        """Random Forest based letter prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            features = self.extract_ml_features(clean_word, set(self.guessed_letters), len(clean_word))
            
            # Get probabilities for all letters
            probs = self.rf_model.predict_proba([features])[0]
            
            # Sort by probability and return best unguessed letter
            letter_probs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                           for i, prob in enumerate(probs)]
            letter_probs.sort(key=lambda x: x[1], reverse=True)
            
            for letter, prob in letter_probs:
                if letter not in self.guessed_letters:
                    return letter
                    
        except Exception as e:
            print(f"RF Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 2: Gradient Boosting
    def algo2_gradient_boosting(self, word):
        """Gradient Boosting based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            features = self.extract_ml_features(clean_word, set(self.guessed_letters), len(clean_word))
            
            probs = self.gb_model.predict_proba([features])[0]
            letter_probs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                           for i, prob in enumerate(probs)]
            letter_probs.sort(key=lambda x: x[1], reverse=True)
            
            for letter, prob in letter_probs:
                if letter not in self.guessed_letters:
                    return letter
                    
        except Exception as e:
            print(f"GB Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 3: Logistic Regression
    def algo3_logistic_regression(self, word):
        """Logistic Regression based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            features = self.extract_ml_features(clean_word, set(self.guessed_letters), len(clean_word))
            
            probs = self.lr_model.predict_proba([features])[0]
            letter_probs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                           for i, prob in enumerate(probs)]
            letter_probs.sort(key=lambda x: x[1], reverse=True)
            
            for letter, prob in letter_probs:
                if letter not in self.guessed_letters:
                    return letter
                    
        except Exception as e:
            print(f"LR Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 4: XGBoost
    def algo4_xgboost(self, word):
        """XGBoost based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            features = self.extract_ml_features(clean_word, set(self.guessed_letters), len(clean_word))
            
            probs = self.xgb_model.predict_proba([features])[0]
            letter_probs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                           for i, prob in enumerate(probs)]
            letter_probs.sort(key=lambda x: x[1], reverse=True)
            
            for letter, prob in letter_probs:
                if letter not in self.guessed_letters:
                    return letter
                    
        except Exception as e:
            print(f"XGB Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 5: LightGBM
    def algo5_lightgbm(self, word):
        """LightGBM based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            features = self.extract_ml_features(clean_word, set(self.guessed_letters), len(clean_word))
            
            probs = self.lgb_model.predict_proba([features])[0]
            letter_probs = [(self.label_encoder.inverse_transform([i])[0], prob) 
                           for i, prob in enumerate(probs)]
            letter_probs.sort(key=lambda x: x[1], reverse=True)
            
            for letter, prob in letter_probs:
                if letter not in self.guessed_letters:
                    return letter
                    
        except Exception as e:
            print(f"LGB Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 6: LSTM Neural Network
    def algo6_lstm(self, word):
        """LSTM based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            # Convert partial word to sequence
            partial_seq = []
            for char in clean_word:
                if char != '.':
                    if char in self.tokenizer.word_index:
                        partial_seq.append(self.tokenizer.word_index[char])
            
            if len(partial_seq) > 0:
                padded_seq = pad_sequences([partial_seq], maxlen=self.max_len)
                probs = self.lstm_model.predict(padded_seq, verbose=0)[0]
                
                # Convert back to letters and sort by probability
                letter_probs = []
                for i, prob in enumerate(probs):
                    if i < len(self.tokenizer.index_word):
                        letter = self.tokenizer.index_word.get(i+1, '')
                        if letter and letter not in self.guessed_letters:
                            letter_probs.append((letter, prob))
                
                if letter_probs:
                    letter_probs.sort(key=lambda x: x[1], reverse=True)
                    return letter_probs[0][0]
                    
        except Exception as e:
            print(f"LSTM Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 7: GRU Neural Network
    def algo7_gru(self, word):
        """GRU based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            partial_seq = []
            for char in clean_word:
                if char != '.':
                    if char in self.tokenizer.word_index:
                        partial_seq.append(self.tokenizer.word_index[char])
            
            if len(partial_seq) > 0:
                padded_seq = pad_sequences([partial_seq], maxlen=self.max_len)
                probs = self.gru_model.predict(padded_seq, verbose=0)[0]
                
                letter_probs = []
                for i, prob in enumerate(probs):
                    if i < len(self.tokenizer.index_word):
                        letter = self.tokenizer.index_word.get(i+1, '')
                        if letter and letter not in self.guessed_letters:
                            letter_probs.append((letter, prob))
                
                if letter_probs:
                    letter_probs.sort(key=lambda x: x[1], reverse=True)
                    return letter_probs[0][0]
                    
        except Exception as e:
            print(f"GRU Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 8: Bidirectional LSTM
    def algo8_bilstm(self, word):
        """Bidirectional LSTM based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            partial_seq = []
            for char in clean_word:
                if char != '.':
                    if char in self.tokenizer.word_index:
                        partial_seq.append(self.tokenizer.word_index[char])
            
            if len(partial_seq) > 0:
                padded_seq = pad_sequences([partial_seq], maxlen=self.max_len)
                probs = self.bilstm_model.predict(padded_seq, verbose=0)[0]
                
                letter_probs = []
                for i, prob in enumerate(probs):
                    if i < len(self.tokenizer.index_word):
                        letter = self.tokenizer.index_word.get(i+1, '')
                        if letter and letter not in self.guessed_letters:
                            letter_probs.append((letter, prob))
                
                if letter_probs:
                    letter_probs.sort(key=lambda x: x[1], reverse=True)
                    return letter_probs[0][0]
                    
        except Exception as e:
            print(f"BiLSTM Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 9: Simple RNN
    def algo9_rnn(self, word):
        """Simple RNN based prediction"""
        try:
            clean_word = word[::2].replace("_", ".")
            partial_seq = []
            for char in clean_word:
                if char != '.':
                    if char in self.tokenizer.word_index:
                        partial_seq.append(self.tokenizer.word_index[char])
            
            if len(partial_seq) > 0:
                padded_seq = pad_sequences([partial_seq], maxlen=self.max_len)
                probs = self.rnn_model.predict(padded_seq, verbose=0)[0]
                
                letter_probs = []
                for i, prob in enumerate(probs):
                    if i < len(self.tokenizer.index_word):
                        letter = self.tokenizer.index_word.get(i+1, '')
                        if letter and letter not in self.guessed_letters:
                            letter_probs.append((letter, prob))
                
                if letter_probs:
                    letter_probs.sort(key=lambda x: x[1], reverse=True)
                    return letter_probs[0][0]
                    
        except Exception as e:
            print(f"RNN Algorithm error: {e}")
            
        return self.fallback_guess()

    # ALGORITHM 10: N-gram Analysis
    def algo10_ngram_analysis(self, word):
        """N-gram based prediction"""
        clean_word = word[::2].replace("_", ".")
        letter_scores = defaultdict(float)
        
        # Trigram analysis
        for i in range(len(clean_word) - 2):
            if clean_word[i] != '.' and clean_word[i+1] != '.' and clean_word[i+2] == '.':
                prefix = clean_word[i:i+2]
                for trigram, count in self.trigrams.items():
                    if trigram.startswith(prefix):
                        candidate = trigram[2]
                        if candidate not in self.guessed_letters:
                            letter_scores[candidate] += count
        
        # Bigram analysis
        for i in range(len(clean_word) - 1):
            if clean_word[i] != '.' and clean_word[i+1] == '.':
                prefix = clean_word[i]
                for bigram, count in self.bigrams.items():
                    if bigram.startswith(prefix):
                        candidate = bigram[1]
                        if candidate not in self.guessed_letters:
                            letter_scores[candidate] += count * 0.5
        
        if letter_scores:
            return max(letter_scores.items(), key=lambda x: x[1])[0]
        
        return self.fallback_guess()

    # ALGORITHM 11: Pattern Matching
    def algo11_pattern_matching(self, word):
        """Pattern-based prediction"""
        clean_word = word[::2].replace("_", ".")
        current_pattern = self.get_partial_pattern(clean_word)
        
        # Find matching patterns
        matching_words = []
        for pattern, words in self.word_patterns.items():
            if self.pattern_matches(current_pattern, pattern):
                matching_words.extend(words)
        
        # Filter by current word constraints
        valid_words = []
        for dict_word in matching_words:
            if len(dict_word) == len(clean_word) and re.match(clean_word, dict_word):
                valid_words.append(dict_word)
        
        if valid_words:
            # Count letter frequencies in valid words
            letter_counts = Counter(''.join(valid_words))
            for letter, count in letter_counts.most_common():
                if letter not in self.guessed_letters:
                    return letter
        
        return self.fallback_guess()

    # ALGORITHM 12: Ensemble Method
    def algo12_ensemble(self, word):
        """Ensemble of multiple algorithms"""
        predictions = {}
        
        # Get predictions from multiple algorithms
        try:
            pred1 = self.algo1_random_forest(word)
            predictions[pred1] = predictions.get(pred1, 0) + 3
        except:
            pass
            
        try:
            pred2 = self.algo10_ngram_analysis(word)
            predictions[pred2] = predictions.get(pred2, 0) + 2
        except:
            pass
            
        try:
            pred3 = self.algo11_pattern_matching(word)
            predictions[pred3] = predictions.get(pred3, 0) + 2
        except:
            pass
        
        # Return most voted prediction
        if predictions:
            return max(predictions.items(), key=lambda x: x[1])[0]
        
        return self.fallback_guess()

    # ALGORITHM 13: Markov Chain
    def algo13_markov_chain(self, word):
        """Markov Chain based prediction"""
        clean_word = word[::2].replace("_", ".")
        
        # Build transition probabilities
        transitions = defaultdict(lambda: defaultdict(int))
        for dict_word in self.full_dictionary:
            for i in range(len(dict_word) - 1):
                current_char = dict_word[i]
                next_char = dict_word[i + 1]
                transitions[current_char][next_char] += 1
        
        # Predict next letter based on last known letter
        letter_scores = defaultdict(float)
        for i in range(len(clean_word) - 1):
            if clean_word[i] != '.' and clean_word[i+1] == '.':
                current_char = clean_word[i]
                for next_char, count in transitions[current_char].items():
                    if next_char not in self.guessed_letters:
                        letter_scores[next_char] += count
        
        if letter_scores:
            return max(letter_scores.items(), key=lambda x: x[1])[0]
        
        return self.fallback_guess()

    # ALGORITHM 14: Bayesian Inference
    def algo14_bayesian_inference(self, word):
        """Bayesian inference based prediction"""
        clean_word = word[::2].replace("_", ".")
        
        # Prior probabilities (letter frequencies)
        total_chars = sum(count for _, count in self.full_dictionary_common_letter_sorted)
        priors = {letter: count/total_chars for letter, count in self.full_dictionary_common_letter_sorted}
        
        # Likelihood calculation
        possible_words = []
        for dict_word in self.full_dictionary:
            if len(dict_word) == len(clean_word) and re.match(clean_word, dict_word):
                possible_words.append(dict_word)
        
        if not possible_words:
            return self.fallback_guess()
        
        # Calculate posterior probabilities
        letter_posteriors = defaultdict(float)
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            if letter not in self.guessed_letters:
                # Likelihood: how often does this letter appear in possible words?
                likelihood = sum(1 for word in possible_words if letter in word) / len(possible_words)
                posterior = likelihood * priors.get(letter, 0.001)
                letter_posteriors[letter] = posterior
        
        if letter_posteriors:
            return max(letter_posteriors.items(), key=lambda x: x[1])[0]
        
        return self.fallback_guess()

    # ALGORITHM 15: Hybrid Deep Learning
    def algo15_hybrid_deep_learning(self, word):
        """Hybrid approach combining multiple neural networks"""
        predictions = defaultdict(float)
        
        # Get predictions from neural networks
        try:
            lstm_pred = self.algo6_lstm(word)
            predictions[lstm_pred] += 0.3
        except:
            pass
            
        try:
            gru_pred = self.algo7_gru(word)
            predictions[gru_pred] += 0.3
        except:
            pass
            
        try:
            bilstm_pred = self.algo8_bilstm(word)
            predictions[bilstm_pred] += 0.4
        except:
            pass
        
        # Add traditional analysis
        try:
            ngram_pred = self.algo10_ngram_analysis(word)
            predictions[ngram_pred] += 0.2
        except:
            pass
        
        if predictions:
            return max(predictions.items(), key=lambda x: x[1])[0]
        
        return self.fallback_guess()

    # Helper methods
    def get_partial_pattern(self, clean_word):
        """Get pattern for partial word"""
        char_map = {}
        pattern = []
        next_num = 1
        for char in clean_word:
            if char == '.':
                pattern.append('?')
            else:
                if char not in char_map:
                    char_map[char] = next_num
                    next_num += 1
                pattern.append(str(char_map[char]))
        return ''.join(pattern)
    
    def pattern_matches(self, partial_pattern, full_pattern):
        """Check if partial pattern matches full pattern"""
        if len(partial_pattern) != len(full_pattern):
            return False
        
        for p, f in zip(partial_pattern, full_pattern):
            if p != '?' and p != f:
                return False
        return True
    
    def fallback_guess(self):
        """Fallback to frequency-based guessing"""
        for letter, _ in self.full_dictionary_common_letter_sorted:
            if letter not in self.guessed_letters:
                return letter
        return 'a'

    # Main guess method - change the algorithm number to test different approaches

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions
        clean_word = word[::2].replace("_",".")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []
        
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break            
        
        # CHOOSE YOUR ALGORITHM HERE - just replace the method call:
        # return self.algo1_position_frequency(word)
        # return self.algo2_vowel_consonant_pattern(word)
        # return self.algo3_letter_pairs(word)
        # return self.algo4_word_length_strategy(word)
        return self.algo2_gradient_boosting(word)

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


# # API Usage Examples

# ## To start a new game:
# 1. Make sure you have implemented your own "guess" method.
# 2. Use the access_token that we sent you to create your HangmanAPI object. 
# 3. Start a game by calling "start_game" method.
# 4. If you wish to test your function without being recorded, set "practice" parameter to 1.
# 5. Note: You have a rate limit of 20 new games per minute. DO NOT start more than 20 new games within one minute.

# In[ ]:


api = HangmanAPI(access_token="e8706baea67230c5106bf22704c839", timeout=2000)


# ## Playing practice games:
# You can use the command below to play up to 100,000 practice games.

# In[ ]:


api.start_game(practice=1,verbose=True)
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
print('total practice runs = %d, total recorded runs = %d, total recorded successes = %d, total practice successes = %d' % (total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes))

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
#     #api.start_game(practice=0,verbose=False)
    
#     # DO NOT REMOVE as otherwise the server may lock you out for too high frequency of requests
#     time.sleep(0.5)


# In[ ]:





# ## To check your game statistics
# 1. Simply use "my_status" method.
# 2. Returns your total number of games, and number of wins.

# In[ ]:


[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
print('total practice runs = %d, total recorded runs = %d, total recorded successes = %d, total practice successes = %d' % (total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes))
# success_rate = total_recorded_successes/total_recorded_runs
# print('overall success rate = %.3f' % success_rate)


# In[ ]:




