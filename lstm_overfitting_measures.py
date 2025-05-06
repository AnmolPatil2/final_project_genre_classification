import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import pickle
import logging
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# Download NLTK resources explicitly - correct resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logging.info("NLTK resources downloaded successfully")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")
    
    # Set download path explicitly
    nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
    os.makedirs(nltk_data_path, exist_ok=True)
    logging.info(f"Created NLTK data path at: {nltk_data_path}")
    
    # Try downloading again with explicit path
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        nltk.download('stopwords', download_dir=nltk_data_path)
        nltk.download('wordnet', download_dir=nltk_data_path)
        logging.info(f"NLTK resources downloaded to {nltk_data_path}")
    except Exception as e:
        logging.error(f"Failed to download NLTK resources again: {str(e)}")
        logging.warning("Will use fallback tokenization if needed")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_group_lstm_attention.log"),
        logging.StreamHandler()
    ]
)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set paths
base_dir = script_dir
data_dir = os.path.join(base_dir, "processed_data")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "lstm_attention"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "lstm_attention"), exist_ok=True)

# Data augmentation techniques for text
class TextAugmenter:
    """Text augmentation for overview texts"""
    
    def __init__(self, word_dropout_prob=0.1, shuffle_prob=0.1, max_shuffle_distance=3):
        self.word_dropout_prob = word_dropout_prob
        self.shuffle_prob = shuffle_prob
        self.max_shuffle_distance = max_shuffle_distance
    
    def apply_word_dropout(self, tokens):
        """Randomly drop words"""
        return [token for token in tokens if random.random() > self.word_dropout_prob]
    
    def apply_word_shuffle(self, tokens):
        """Randomly shuffle nearby words"""
        result = tokens.copy()
        for i in range(len(result) - 1):
            if random.random() < self.shuffle_prob:
                # Determine shuffle distance (1 to max_shuffle_distance)
                distance = min(random.randint(1, self.max_shuffle_distance), len(result) - i - 1)
                # Swap current token with one ahead by 'distance'
                result[i], result[i + distance] = result[i + distance], result[i]
        return result
    
    def augment(self, text, tokenizer_fn):
        """Apply augmentation to text"""
        tokens = tokenizer_fn(text)
        if random.random() < 0.5:  # 50% chance of applying word dropout
            tokens = self.apply_word_dropout(tokens)
        if random.random() < 0.5:  # 50% chance of applying word shuffle
            tokens = self.apply_word_shuffle(tokens)
        return " ".join(tokens)

class TextPreprocessor:
    """Text preprocessor for movie overviews"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=200):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
            logging.warning("Could not load stopwords, using empty set")
        
        self.lemmatizer = WordNetLemmatizer()
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.vocab_size = 2  # PAD and UNK tokens
    
    def tokenize(self, text):
        """Tokenize and preprocess text"""
        if not isinstance(text, str):
            return []
        
        try:
            # Lowercase and tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            tokens = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token.isalpha() and token not in self.stop_words
            ]
            
            return tokens
        except Exception as e:
            logging.warning(f"Using fallback tokenization: {str(e)}")
            # If tokenization fails, return simple whitespace-based tokens as fallback
            if isinstance(text, str):
                # Use raw string for regex
                text = re.sub(r'\s+', ' ', text).strip()
                return [token.lower() for token in text.split() if token.isalpha()]
            return []
    
    def build_vocab(self, texts):
        """Build vocabulary from list of texts"""
        # Count word frequencies
        for text in tqdm(texts, desc="Building vocabulary"):
            tokens = self.tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Take top words (minus PAD and UNK which are already in the vocab)
        top_words = sorted_words[:self.max_vocab_size - 2]
        
        # Add to vocabulary
        for word, count in top_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        logging.info(f"Vocabulary built with {self.vocab_size} words")
    
    def text_to_sequence(self, text):
        """Convert text to sequence of word indices"""
        tokens = self.tokenize(text)
        sequence = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens[:self.max_seq_length]
        ]
        
        # Pad sequence
        if len(sequence) < self.max_seq_length:
            sequence += [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))
        
        return sequence
    
    def save(self, path):
        """Save preprocessor to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'max_seq_length': self.max_seq_length,
                'vocab_size': self.vocab_size
            }, f)
        
        logging.info(f"Text preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(max_seq_length=data['max_seq_length'])
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.vocab_size = data['vocab_size']
        
        logging.info(f"Text preprocessor loaded from {path}")
        return preprocessor

class MovieGroupDataset(Dataset):
    """Dataset for multi-label movie group classification"""
    
    def __init__(self, df, text_preprocessor, group_to_idx=None, augmenter=None, is_training=False):
        self.df = df
        self.preprocessor = text_preprocessor
        self.augmenter = augmenter
        self.is_training = is_training
        
        # Build group to index mapping if not provided
        if group_to_idx is None:
            # Extract all unique groups
            all_groups = set()
            for groups_list in df['groups']:
                if isinstance(groups_list, list):
                    all_groups.update(groups_list)
            
            self.group_to_idx = {group: i for i, group in enumerate(sorted(all_groups))}
        else:
            self.group_to_idx = group_to_idx
        
        self.num_groups = len(self.group_to_idx)
        logging.info(f"Dataset initialized with {self.num_groups} possible groups")
        
        # Preprocess all texts
        self.sequences = []
        self.labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
            # Preprocess text
            overview = row['overview'] if isinstance(row['overview'], str) else ""
            sequence = self.preprocessor.text_to_sequence(overview)
            self.sequences.append(sequence)
            
            # Create multi-hot encoded label vector
            label_vector = torch.zeros(self.num_groups)
            
            # Fill in the labels for the groups this movie belongs to
            if 'groups' in row and isinstance(row['groups'], list):
                for group in row['groups']:
                    if group in self.group_to_idx:
                        label_vector[self.group_to_idx[group]] = 1.0 - 0.1  # Label smoothing: use 0.9 instead of 1.0
            
            self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Apply augmentation during training if available
        if self.is_training and self.augmenter is not None and random.random() < 0.3:  # 30% chance of applying augmentation
            # Get original text and apply augmentation
            overview = self.df.iloc[idx]['overview'] if isinstance(self.df.iloc[idx]['overview'], str) else ""
            augmented_text = self.augmenter.augment(overview, self.preprocessor.tokenize)
            sequence = self.preprocessor.text_to_sequence(augmented_text)
            sequence = torch.tensor(sequence, dtype=torch.long)
        else:
            # Use precomputed sequence
            sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        
        # Fix: Use clone() and detach() instead of torch.tensor()
        label = self.labels[idx].clone().detach()
        return {'text': sequence, 'label': label}

class SelfAttention(nn.Module):
    """Self attention module for sequence data"""
    
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Projection layers
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaled dot-product attention factor
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, hidden_states, mask=None):
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer normalization
        normed_hidden = self.layer_norm(hidden_states)
        
        # Create query, key, and value projections
        Q = self.query(normed_hidden)  # [batch_size, seq_len, attention_dim]
        K = self.key(normed_hidden)    # [batch_size, seq_len, attention_dim]
        V = self.value(normed_hidden)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        # Transpose K for matrix multiplication
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(hidden_states.device)
        # energy shape: [batch_size, seq_len, seq_len]
        
        # Apply mask if provided (e.g., to exclude padding tokens)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        # attention shape: [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        output = torch.matmul(attention, V)
        # output shape: [batch_size, seq_len, hidden_dim]
        
        # Residual connection
        output = output + hidden_states
        
        return output, attention

class LSTMWithAttentionMultiLabel(nn.Module):
    """LSTM model with self-attention for multi-label classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, attention_dim=None, 
                 n_layers=1, bidirectional=True, dropout=0.6, pad_idx=0, freeze_embeddings=True):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Option to freeze embeddings to prevent overfitting
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=0.0 if n_layers <= 1 else dropout,
            batch_first=True
        )
        
        # Determine dimensions
        self.bidirectional = bidirectional
        self.hidden_factor = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dim * self.hidden_factor, attention_dim)
        
        # Fully connected layers for multi-label classification with higher regularization
        self.fc1 = nn.Linear(hidden_dim * self.hidden_factor, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout + 0.1)  # Higher dropout for intermediate layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim * self.hidden_factor)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Save pad idx for creating attention mask
        self.pad_idx = pad_idx
        
        # Initialize weights with a safer custom function
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence, safely handling all parameter types"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' not in name:  # Skip embedding weights
                    if len(param.shape) >= 2:
                        # Use Kaiming initialization for weights of linear layers with 2+ dimensions
                        nn.init.kaiming_uniform_(param.data, nonlinearity='relu')
                    else:
                        # For 1D parameters (like some batch norm params), use normal init
                        nn.init.normal_(param.data, mean=0, std=0.01)
            elif 'bias' in name:
                # Initialize biases to small value
                nn.init.constant_(param.data, 0.0)
    
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        mask = (text != self.pad_idx).unsqueeze(1).repeat(1, text.size(1), 1)
        # mask shape: [batch size, seq len, seq len]
        
        # Embed text
        embedded = self.dropout2(self.embedding(text))
        # embedded shape: [batch size, seq len, embedding dim]
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: [batch size, seq len, hidden dim * num directions]
        # hidden shape: [num layers * num directions, batch size, hidden dim]
        
        # Apply layer normalization
        lstm_output = self.layer_norm1(lstm_output)
        
        # Apply self-attention to LSTM outputs
        attended_output, attention_weights = self.attention(lstm_output, mask)
        # attended_output shape: [batch size, seq len, hidden dim * num directions]
        # attention_weights shape: [batch size, seq len, seq len]
        
        # Global max pooling to get the most important features across the sequence
        pooled_output = torch.max(attended_output, dim=1)[0]
        # pooled_output shape: [batch size, hidden dim * num directions]
        
        # Pass through fully connected layers with stronger regularization
        x = self.fc1(self.dropout2(pooled_output))
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        logits = self.fc2(x)
        # logits shape: [batch size, output dim]
        
        return logits, attention_weights

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, labels):
        bce_loss = self.bce(logits, labels)
        probas = torch.sigmoid(logits)
        loss = torch.where(
            labels == 1,
            self.alpha * (1 - probas) ** self.gamma * bce_loss,
            (1 - self.alpha) * probas ** self.gamma * bce_loss
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def calculate_multilabel_f1(y_true, y_pred):
    """
    Calculate F1 score for multi-label classification
    Handles the case where the sklearn metrics function might fail
    due to a mix of continuous and multi-label targets.
    """
    # Convert to binary (0, 1) based on a threshold of 0.5
    y_true_binary = (y_true > 0.5).astype(np.int64)
    y_pred_binary = (y_pred > 0.5).astype(np.int64)
    
    # Calculate F1 score manually
    # True positives: sum(y_true_binary * y_pred_binary)
    # False positives: sum(y_pred_binary) - tp
    # False negatives: sum(y_true_binary) - tp
    
    tp = np.sum(y_true_binary * y_pred_binary, axis=0)
    fp = np.sum(y_pred_binary, axis=0) - tp
    fn = np.sum(y_true_binary, axis=0) - tp
    
    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Average across all classes (macro averaging)
    return np.mean(f1)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    
    # For tracking F1 score
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(text)  # Ignore attention weights during training
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Convert logits to predictions using sigmoid
        preds = torch.sigmoid(logits)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate F1 score using our custom function
    f1 = calculate_multilabel_f1(all_labels, all_preds)
    
    return epoch_loss / len(dataloader), f1

def evaluate(model, dataloader, criterion, device, thresholds=None):
    """Evaluate model on dataloader"""
    model.eval()
    epoch_loss = 0
    
    all_logits = []
    all_preds = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits, attention_weights = model(text)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Store for metrics calculation
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # Store a sample of attention weights (to avoid storing too much data)
            if len(all_attention_weights) < 10:
                all_attention_weights.append(attention_weights[0].cpu().numpy())
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    
    # Apply thresholds (default 0.5 if not provided)
    if thresholds is None:
        thresholds = [0.5] * all_logits.shape[1]
    
    # Apply thresholds to get binary predictions
    all_preds = np.zeros_like(all_probs)
    for i, threshold in enumerate(thresholds):
        # Ensure threshold is a float
        threshold_float = float(threshold)
        all_preds[:, i] = all_probs[:, i] > threshold_float
    
    # Calculate F1 score using our custom function
    f1 = calculate_multilabel_f1(all_labels, all_preds)
    
    return epoch_loss / len(dataloader), f1, all_probs, all_preds, all_labels, all_attention_weights

def find_optimal_thresholds(probs, labels):
    """Find optimal thresholds for each group using validation data"""
    n_classes = probs.shape[1]
    thresholds = []
    
    for i in range(n_classes):
        # Convert to binary for threshold calculation
        binary_labels = (labels[:, i] > 0.5).astype(int)
        
        # Find precision-recall curve
        precisions = []
        recalls = []
        thrs = []
        
        # Try different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            binary_preds = (probs[:, i] > threshold).astype(int)
            
            # Compute precision and recall
            tp = np.sum(binary_labels & binary_preds)
            fp = np.sum(~binary_labels & binary_preds)
            fn = np.sum(binary_labels & ~binary_preds)
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            
            precisions.append(precision)
            recalls.append(recall)
            thrs.append(threshold)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * np.array(precisions) * np.array(recalls) / (np.array(precisions) + np.array(recalls) + 1e-10)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thrs[best_idx]) if best_idx < len(thrs) else 0.5
        
        thresholds.append(best_threshold)
    
    return thresholds

def visualize_attention(attention_weights, texts, preprocessor, save_path=None):
    """Visualize attention weights for sample texts"""
    if not attention_weights:
        logging.warning("No attention weights to visualize")
        return
    
    # Convert text samples to tokens for better visualization
    tokenized_texts = []
    for text in texts:
        if isinstance(text, str):
            tokens = preprocessor.tokenize(text)[:preprocessor.max_seq_length]
            # Pad if necessary
            if len(tokens) < preprocessor.max_seq_length:
                tokens = tokens + ['<PAD>'] * (preprocessor.max_seq_length - len(tokens))
            tokenized_texts.append(tokens)
    
    # Create figure
    fig, axes = plt.subplots(len(attention_weights), 1, figsize=(15, 5 * len(attention_weights)))
    if len(attention_weights) == 1:
        axes = [axes]
    
    for i, weights in enumerate(attention_weights):
        if i < len(tokenized_texts):
            # Get the tokens for this text
            tokens = tokenized_texts[i]
            
            # Create heatmap for only the first 50 tokens (to keep it readable)
            display_len = min(50, len(tokens))
            
            # Select the attention weights for the first token attending to all other tokens
            # This shows what the model focuses on for classification
            token_weights = weights[0, :display_len]
            
            # Create heatmap
            sns.heatmap(
                token_weights.reshape(1, -1),
                cmap='viridis',
                annot=False,
                xticklabels=tokens[:display_len],
                yticklabels=['Attention'],
                ax=axes[i]
            )
            
            # Set title and labels
            axes[i].set_title(f"Attention Weights for Sample {i+1}")
            axes[i].set_xlabel("Words")
            
            # Rotate x-axis labels for readability
            axes[i].set_xticklabels(tokens[:display_len], rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save or show the visualization
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
        logging.info("Attention visualization displayed")

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Settings - reduced complexity to prevent overfitting
    EMBEDDING_DIM = 200
    HIDDEN_DIM = 128
    ATTENTION_DIM = 128
    N_LAYERS = 1
    BIDIRECTIONAL = True
    DROPOUT = 0.6
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-4
    N_EPOCHS = 50
    FREEZE_EMBEDDINGS = True
    PATIENCE = 5  # Early stopping patience
    
    # Device - support for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Load data
    logging.info("Loading data...")
    try:
        train_df = pd.read_pickle(os.path.join(data_dir, "splits", "train.pkl"))
        val_df = pd.read_pickle(os.path.join(data_dir, "splits", "val.pkl"))
        test_df = pd.read_pickle(os.path.join(data_dir, "splits", "test.pkl"))
        logging.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        
        # Log sample data
        logging.info(f"Sample data: {train_df.iloc[0].to_dict()}")
    except FileNotFoundError as e:
        logging.error(f"Data files not found: {str(e)}")
        logging.error("Please ensure that the processed data files are in the correct location")
        return
    
    # Initialize text preprocessor and build vocabulary
    logging.info("Initializing text preprocessor...")
    preprocessor_path = os.path.join(data_dir, "text_preprocessor.pkl")
    
    if os.path.exists(preprocessor_path):
        # Load existing preprocessor
        text_preprocessor = TextPreprocessor.load(preprocessor_path)
    else:
        # Create new preprocessor and build vocabulary
        text_preprocessor = TextPreprocessor(max_vocab_size=10000, max_seq_length=200)
        text_preprocessor.build_vocab(train_df['overview'])
        text_preprocessor.save(preprocessor_path)
    
    # Initialize text augmenter for training
    text_augmenter = TextAugmenter(word_dropout_prob=0.1, shuffle_prob=0.1)
    
    # Initialize datasets
    logging.info("Creating datasets...")
    train_dataset = MovieGroupDataset(
        train_df, text_preprocessor, group_to_idx=None, 
        augmenter=text_augmenter, is_training=True
    )
    
    # Use the group_to_idx from the training set for consistency
    val_dataset = MovieGroupDataset(
        val_df, text_preprocessor, group_to_idx=train_dataset.group_to_idx,
        augmenter=None, is_training=False
    )
    
    test_dataset = MovieGroupDataset(
        test_df, text_preprocessor, group_to_idx=train_dataset.group_to_idx,
        augmenter=None, is_training=False
    )
    
    # Save group_to_idx mapping for inference
    with open(os.path.join(data_dir, "group_to_idx.json"), 'w') as f:
        json.dump(train_dataset.group_to_idx, f)
    
    # Create reverse mapping for prediction interpretation
    idx_to_group = {idx: group for group, idx in train_dataset.group_to_idx.items()}
    
    # Initialize data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    logging.info("Initializing model...")
    model = LSTMWithAttentionMultiLabel(
        vocab_size=text_preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=train_dataset.num_groups,
        attention_dim=ATTENTION_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=text_preprocessor.word2idx['<PAD>'],
        freeze_embeddings=FREEZE_EMBEDDINGS
    )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Use focal loss to handle class imbalance
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Track best model for early stopping
    best_val_f1 = 0.0
    best_epoch = 0
    early_stop_counter = 0
    
    # Training loop
    logging.info("Starting training...")
    for epoch in range(N_EPOCHS):
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS}")
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        logging.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        # Evaluate on validation set
        val_loss, val_f1, val_probs, val_preds, val_labels, attention_weights = evaluate(
            model, val_loader, criterion, device
        )
        logging.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Update learning rate based on validation F1
        scheduler.step(val_f1)
        
        # Check for improvement
        if val_f1 > best_val_f1:
            logging.info(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}. Saving model...")
            best_val_f1 = val_f1
            best_epoch = epoch
            early_stop_counter = 0
            
            # Save model
            model_path = os.path.join(models_dir, "lstm_attention", "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'train_f1': train_f1,
                'train_loss': train_loss
            }, model_path)
            
            # Find optimal thresholds on validation set
            logging.info("Finding optimal thresholds...")
            optimal_thresholds = find_optimal_thresholds(val_probs, val_labels)
            
            # Save thresholds
            with open(os.path.join(models_dir, "lstm_attention", "optimal_thresholds.json"), 'w') as f:
                json.dump(optimal_thresholds, f)
            
            # Save a visualization of attention weights
            sample_texts = val_df['overview'].iloc[:min(3, len(val_df))].tolist()
            viz_path = os.path.join(results_dir, "lstm_attention", f"attention_viz_epoch_{epoch+1}.png")
            visualize_attention(attention_weights, sample_texts, text_preprocessor, viz_path)
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in validation F1. Counter: {early_stop_counter}/{PATIENCE}")
            
            if early_stop_counter >= PATIENCE:
                logging.info(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    logging.info(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch+1}")
    
    # Load best model for testing - with error handling
    checkpoint_path = os.path.join(models_dir, "lstm_attention", "best_model.pt")
    if os.path.exists(checkpoint_path):
        logging.info("Loading best model for evaluation on test set...")
        try:
            checkpoint = torch.load(checkpoint_path)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info("Loaded model checkpoint successfully")
            else:
                logging.warning("Checkpoint exists but doesn't contain model_state_dict. Using current model state.")
        except Exception as e:
            logging.error(f"Error loading checkpoint: {str(e)}. Using current model state.")
    else:
        logging.warning("No saved model checkpoint found. Using current model state.")
    
    # Load optimal thresholds - with better error handling and explicit float conversion
    optimal_thresholds_path = os.path.join(models_dir, "lstm_attention", "optimal_thresholds.json")
    if os.path.exists(optimal_thresholds_path):
        try:
            with open(optimal_thresholds_path, 'r') as f:
                raw_thresholds = json.load(f)
            # Make sure all thresholds are floats
            optimal_thresholds = [float(t) for t in raw_thresholds]
            logging.info(f"Loaded {len(optimal_thresholds)} optimal thresholds")
        except Exception as e:
            logging.error(f"Error loading optimal thresholds: {str(e)}")
            optimal_thresholds = [0.5] * train_dataset.num_groups
    else:
        # Use default thresholds if file doesn't exist
        logging.warning("Optimal thresholds file not found. Using default threshold of 0.5")
        optimal_thresholds = [0.5] * train_dataset.num_groups
    
    # Evaluate on test set
    test_loss, test_f1, test_probs, test_preds, test_labels, test_attention = evaluate(
        model, test_loader, criterion, device, optimal_thresholds
    )
    logging.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    
    # Calculate per-class metrics on test set
    test_preds_binary = test_preds.astype(int)
    test_labels_binary = (test_labels > 0.5).astype(int)
    
    # Calculate per-class precision, recall, and F1
    class_precision = precision_score(test_labels_binary, test_preds_binary, average=None, zero_division=0)
    class_recall = recall_score(test_labels_binary, test_preds_binary, average=None, zero_division=0)
    class_f1 = f1_score(test_labels_binary, test_preds_binary, average=None, zero_division=0)
    
    # Report per-class results
    class_results = {
        'precision': {},
        'recall': {},
        'f1': {}
    }
    
    for i, group in idx_to_group.items():
        class_results['precision'][group] = float(class_precision[i])
        class_results['recall'][group] = float(class_recall[i])
        class_results['f1'][group] = float(class_f1[i])
    
    # Save classification report
    with open(os.path.join(results_dir, "lstm_attention", "class_results.json"), 'w') as f:
        json.dump(class_results, f, indent=2)
    
    # Log top and bottom performing classes
    top_classes = sorted(
        [(group, score) for group, score in class_results['f1'].items()],
        key=lambda x: x[1], reverse=True
    )[:10]
    
    bottom_classes = sorted(
        [(group, score) for group, score in class_results['f1'].items()],
        key=lambda x: x[1]
    )[:10]
    
    logging.info("Top performing classes:")
    for group, score in top_classes:
        logging.info(f"{group}: F1={score:.4f}")
    
    logging.info("Bottom performing classes:")
    for group, score in bottom_classes:
        logging.info(f"{group}: F1={score:.4f}")
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        np.mean(test_preds_binary == test_labels_binary, axis=0).reshape(1, -1),
        cmap='viridis',
        annot=False,
        xticklabels=False,
        yticklabels=['Accuracy']
    )
    plt.title("Per-Class Accuracy on Test Set")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lstm_attention", "class_accuracy.png"))
    
    # Create inference function
    def predict_groups(text, top_k=5):
        """Predict movie groups for a given text"""
        # Preprocess text
        sequence = text_preprocessor.text_to_sequence(text)
        sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            logits, attention_weights = model(sequence)
            probs = torch.sigmoid(logits)
        
        # Get probabilities and convert to numpy
        probs = probs.squeeze(0).cpu().numpy()
        
        # Apply optimal thresholds
        preds = np.zeros_like(probs)
        for i, threshold in enumerate(optimal_thresholds):
            preds[i] = probs[i] > threshold
        
        # Get predicted groups
        predicted_groups = []
        for i, pred in enumerate(preds):
            if pred:
                group = idx_to_group[i]
                predicted_groups.append((group, float(probs[i])))
        
        # Sort by probability
        predicted_groups.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return predicted_groups[:top_k]
    
    # Save inference function
    with open(os.path.join(models_dir, "lstm_attention", "inference.py"), 'w') as f:
        f.write("""import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle

class SelfAttention(nn.Module):
    \"\"\"Self attention module for sequence data\"\"\"
    
    def __init__(self, hidden_dim, attention_dim=None):
        super().__init__()
        
        if attention_dim is None:
            attention_dim = hidden_dim
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Projection layers
        self.query = nn.Linear(hidden_dim, attention_dim)
        self.key = nn.Linear(hidden_dim, attention_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaled dot-product attention factor
        self.scale = torch.sqrt(torch.FloatTensor([attention_dim]))
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, hidden_states, mask=None):
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer normalization
        normed_hidden = self.layer_norm(hidden_states)
        
        # Create query, key, and value projections
        Q = self.query(normed_hidden)  # [batch_size, seq_len, attention_dim]
        K = self.key(normed_hidden)    # [batch_size, seq_len, attention_dim]
        V = self.value(normed_hidden)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        # Transpose K for matrix multiplication
        energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(hidden_states.device)
        # energy shape: [batch_size, seq_len, seq_len]
        
        # Apply mask if provided (e.g., to exclude padding tokens)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Apply softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        # attention shape: [batch_size, seq_len, seq_len]
        
        # Apply attention weights to values
        output = torch.matmul(attention, V)
        # output shape: [batch_size, seq_len, hidden_dim]
        
        # Residual connection
        output = output + hidden_states
        
        return output, attention

class LSTMWithAttentionMultiLabel(nn.Module):
    \"\"\"LSTM model with self-attention for multi-label classification\"\"\"
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, attention_dim=None, 
                 n_layers=1, bidirectional=True, dropout=0.6, pad_idx=0, freeze_embeddings=True):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # Option to freeze embeddings to prevent overfitting
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=0.0 if n_layers <= 1 else dropout,
            batch_first=True
        )
        
        # Determine dimensions
        self.bidirectional = bidirectional
        self.hidden_factor = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dim * self.hidden_factor, attention_dim)
        
        # Fully connected layers for multi-label classification with higher regularization
        self.fc1 = nn.Linear(hidden_dim * self.hidden_factor, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout + 0.1)  # Higher dropout for intermediate layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim * self.hidden_factor)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Save pad idx for creating attention mask
        self.pad_idx = pad_idx
    
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # Create attention mask (1 for valid tokens, 0 for padding)
        mask = (text != self.pad_idx).unsqueeze(1).repeat(1, text.size(1), 1)
        # mask shape: [batch size, seq len, seq len]
        
        # Embed text
        embedded = self.dropout2(self.embedding(text))
        # embedded shape: [batch size, seq len, embedding dim]
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.lstm(embedded)
        # lstm_output shape: [batch size, seq len, hidden dim * num directions]
        # hidden shape: [num layers * num directions, batch size, hidden dim]
        
        # Apply layer normalization
        lstm_output = self.layer_norm1(lstm_output)
        
        # Apply self-attention to LSTM outputs
        attended_output, attention_weights = self.attention(lstm_output, mask)
        # attended_output shape: [batch size, seq len, hidden dim * num directions]
        # attention_weights shape: [batch size, seq len, seq len]
        
        # Global max pooling to get the most important features across the sequence
        pooled_output = torch.max(attended_output, dim=1)[0]
        # pooled_output shape: [batch size, hidden dim * num directions]
        
        # Pass through fully connected layers with stronger regularization
        x = self.fc1(self.dropout2(pooled_output))
        x = self.layer_norm2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        logits = self.fc2(x)
        # logits shape: [batch size, output dim]
        
        return logits, attention_weights

class TextPreprocessor:
    @classmethod
    def load(cls, path):
        \"\"\"Load preprocessor from file\"\"\"
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls.__new__(cls)
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.max_seq_length = data['max_seq_length']
        preprocessor.vocab_size = data['vocab_size']
        
        return preprocessor
    
    def text_to_sequence(self, text):
        \"\"\"Convert text to sequence of word indices\"\"\"
        import re
        
        # Tokenize and preprocess text
        if not isinstance(text, str):
            tokens = []
        else:
            # Simple tokenization as fallback
            text = text.lower()
            text = re.sub(r'[^a-z0-9\\s]', '', text)
            tokens = text.split()
        
        # Convert tokens to indices
        sequence = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens[:self.max_seq_length]
        ]
        
        # Pad sequence
        if len(sequence) < self.max_seq_length:
            sequence += [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))
        
        return sequence

def load_model(model_dir):
    \"\"\"Load the model and necessary files for inference\"\"\"
    # Load text preprocessor
    preprocessor_path = os.path.join(model_dir, '..', '..', 'processed_data', 'text_preprocessor.pkl')
    text_preprocessor = TextPreprocessor.load(preprocessor_path)
    
    # Load group_to_idx mapping
    with open(os.path.join(model_dir, '..', '..', 'processed_data', 'group_to_idx.json'), 'r') as f:
        group_to_idx = json.load(f)
    
    # Create reverse mapping
    idx_to_group = {int(idx): group for group, idx in group_to_idx.items()}
    
    # Load optimal thresholds
    with open(os.path.join(model_dir, 'optimal_thresholds.json'), 'r') as f:
        optimal_thresholds = json.load(f)
    
    # Load model
    model_path = os.path.join(model_dir, 'best_model.pt')
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Initialize model with same parameters
    model = LSTMWithAttentionMultiLabel(
        vocab_size=text_preprocessor.vocab_size,
        embedding_dim=200,
        hidden_dim=128,
        output_dim=len(group_to_idx),
        attention_dim=128,
        n_layers=1,
        bidirectional=True,
        dropout=0.6,
        pad_idx=text_preprocessor.word2idx['<PAD>'],
        freeze_embeddings=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, text_preprocessor, idx_to_group, optimal_thresholds

def predict_groups(text, model_dir, top_k=5):
    \"\"\"Predict movie groups for a given text\"\"\"
    # Load model and files
    model, text_preprocessor, idx_to_group, optimal_thresholds = load_model(model_dir)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Preprocess text
    sequence = text_preprocessor.text_to_sequence(text)
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        logits, attention_weights = model(sequence)
        probs = torch.sigmoid(logits)
    
    # Get probabilities and convert to numpy
    probs = probs.squeeze(0).cpu().numpy()
    
    # Apply optimal thresholds
    preds = np.zeros_like(probs)
    for i, threshold in enumerate(optimal_thresholds):
        preds[i] = probs[i] > threshold
    
    # Get predicted groups
    predicted_groups = []
    for i, pred in enumerate(preds):
        if pred:
            group = idx_to_group[str(i)]
            predicted_groups.append((group, float(probs[i])))
    
    # Sort by probability
    predicted_groups.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return predicted_groups[:top_k]

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict movie groups from text')
    parser.add_argument('text', type=str, help='Text to predict groups for')
    parser.add_argument('--model_dir', type=str, default='models/lstm_attention', help='Directory containing model files')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    
    args = parser.parse_args()
    
    # Predict groups
    predicted_groups = predict_groups(args.text, args.model_dir, args.top_k)
    
    # Print predictions
    print('Predicted groups:')
    for group, prob in predicted_groups:
        print(f'{group}: {prob:.4f}')
""")
    
    logging.info("Training and evaluation complete. Results saved to results directory.")
    
    # Example prediction on a sample text
    sample_text = "A superhero with special powers fights against evil forces to save the world"
    logging.info(f"Example prediction for: '{sample_text}'")
    predictions = predict_groups(sample_text)
    for group, prob in predictions:
        logging.info(f"{group}: {prob:.4f}")

if __name__ == "__main__":
    main()