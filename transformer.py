import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import pickle
import logging
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("genre_transformer.log"),
        logging.StreamHandler()
    ]
)

# Download NLTK resources - MODIFIED with better error handling
try:
    # Set NLTK data path to a local directory
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    
    # Download resources to the specified directory
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
    nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
    
    logging.info(f"NLTK resources downloaded to {nltk_data_dir}")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")
    logging.warning("Will use fallback tokenization if NLTK tokenization fails")

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = script_dir
data_dir = os.path.join(base_dir, "data", "processed", "genre_balanced")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(os.path.join(models_dir, "genre_transformer"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "genre_transformer"), exist_ok=True)

class TextPreprocessor:
    """Text preprocessor for movie overviews"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=200, add_cls_token=True):
        # Try to load stopwords, with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logging.warning(f"Could not load stopwords: {str(e)}. Using empty stopwords set.")
            self.stop_words = set()
        
        # Try to initialize lemmatizer, with fallback
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logging.warning(f"Could not initialize lemmatizer: {str(e)}. Will not use lemmatization.")
            self.lemmatizer = None
            
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        if add_cls_token:
            self.word2idx['<CLS>'] = 2  # Add CLS token for classification
            
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.word_counts = {}
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.vocab_size = len(self.word2idx)
        self.add_cls_token = add_cls_token
    
    def tokenize(self, text):
        """Tokenize and preprocess text with fallback options"""
        if not isinstance(text, str):
            return []
        
        try:
            # Try NLTK tokenization first
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and lemmatize
            if self.lemmatizer:
                tokens = [
                    self.lemmatizer.lemmatize(token) 
                    for token in tokens 
                    if token.isalpha() and token not in self.stop_words
                ]
            else:
                tokens = [
                    token 
                    for token in tokens 
                    if token.isalpha() and token not in self.stop_words
                ]
                
            return tokens
        except Exception as e:
            # Fallback to simple whitespace tokenization
            logging.warning(f"NLTK tokenization failed: {str(e)}. Using fallback tokenization.")
            if isinstance(text, str):
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
        
        # Take top words (minus PAD, UNK, and CLS which are already in the vocab)
        reserved_tokens = len(self.word2idx)
        top_words = sorted_words[:self.max_vocab_size - reserved_tokens]
        
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
        
        # Add CLS token at the beginning if needed
        sequence = []
        if self.add_cls_token:
            sequence = [self.word2idx['<CLS>']]
            
        # Add tokenized text
        max_text_length = self.max_seq_length - len(sequence)
        sequence += [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens[:max_text_length]
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
                'vocab_size': self.vocab_size,
                'add_cls_token': self.add_cls_token
            }, f)
        
        logging.info(f"Text preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls(
            max_seq_length=data['max_seq_length'],
            add_cls_token=data.get('add_cls_token', True)
        )
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.vocab_size = data['vocab_size']
        
        logging.info(f"Text preprocessor loaded from {path}")
        return preprocessor

# Data augmentation functions
def random_deletion(tokens, p=0.1):
    """Randomly delete words from the sentence with probability p"""
    if len(tokens) == 1:
        return tokens
    
    # Randomly delete words with probability p
    new_tokens = []
    for token in tokens:
        if random.random() > p:
            new_tokens.append(token)
            
    # If we deleted all tokens, just return a random token
    if len(new_tokens) == 0:
        rand_int = random.randint(0, len(tokens)-1)
        return [tokens[rand_int]]
    
    return new_tokens

def random_swap(tokens, n=1):
    """Randomly swap n pairs of words in the sentence"""
    if len(tokens) < 2:
        return tokens
    
    new_tokens = tokens.copy()
    for _ in range(n):
        # Choose two random positions
        idx1, idx2 = random.sample(range(len(new_tokens)), 2)
        # Swap tokens
        new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        
    return new_tokens

class MovieGenreDataset(Dataset):
    """Dataset for multilabel movie genre classification with data augmentation"""
    
    def __init__(self, df, text_preprocessor, genre_columns, augment=False):
        self.df = df
        self.preprocessor = text_preprocessor
        self.genre_columns = genre_columns
        self.augment = augment
        
        # Preprocess all texts
        self.sequences = []
        self.labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
            # Preprocess text
            sequence = self.preprocessor.text_to_sequence(row['overview'])
            self.sequences.append(sequence)
            
            # Get multi-hot labels
            label = row[genre_columns].astype(float).values
            self.labels.append(label)
    
    def __len__(self):
        return len(self.df)
    
    def augment_text(self, text):
        """Apply text augmentation techniques"""
        tokens = self.preprocessor.tokenize(text)
        
        # Apply random augmentation
        aug_type = random.random()
        
        if aug_type < 0.5:  # 50% chance for deletion
            tokens = random_deletion(tokens, p=0.1)
        else:  # 50% chance for swap
            tokens = random_swap(tokens, n=min(2, len(tokens)//4))
            
        # Convert back to text and then to sequence
        text = " ".join(tokens)
        return self.preprocessor.text_to_sequence(text)
    
    def __getitem__(self, idx):
        original_sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training if enabled
        if self.augment and random.random() < 0.3:  # 30% chance to augment
            overview = " ".join([self.preprocessor.idx2word.get(idx, "") 
                                for idx in original_sequence 
                                if idx not in [0, 2]])  # Remove PAD and CLS tokens
            sequence = self.augment_text(overview)
        else:
            sequence = original_sequence
            
        return {
            'text': torch.tensor(sequence, dtype=torch.long), 
            'label': torch.tensor(label, dtype=torch.float)
        }

class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in the Transformer paper.
    Uses sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module as in the Transformer paper"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Arguments:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, 1, seq_len) or None
        
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.W_q(query)  # (batch_size, seq_len, d_model)
        k = self.W_k(key)    # (batch_size, seq_len, d_model)
        v = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        # Split the embedding dimension into num_heads * d_k
        q = q.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        k = k.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        v = v.view(batch_size, -1, self.num_heads, self.d_k)  # (batch_size, seq_len, num_heads, d_k)
        
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        # Transpose k for matrix multiplication to (batch_size, num_heads, d_k, seq_len)
        k_t = k.transpose(-2, -1)
        
        # Matrix multiplication: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(q, k_t) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Make sure mask has the right shape for broadcasting
            if mask.dim() == 3:  # If mask is (batch_size, 1, seq_len)
                mask = mask.unsqueeze(1)  # Add head dimension (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention weights to values
        # (batch_size, num_heads, seq_len, seq_len) x (batch_size, num_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, d_k)
        
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        
        # Combine heads
        # The -1 here makes PyTorch infer the correct size rather than hardcoding it
        seq_len = context.size(1)  # Get the actual sequence length
        context = context.view(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Final linear projection
        output = self.W_o(context)  # (batch_size, seq_len, d_model)
        
        return output, attention_weights

class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network as in the Transformer paper"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """Encoder layer as in the Transformer paper"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
        """
        # Self-attention with residual connection and layer norm
        attn_output, self_attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.norm1(x)  # Layer normalization
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.norm2(x)  # Layer normalization
        
        return x, self_attn_weights

class TransformerEncoder(nn.Module):
    """Transformer Encoder stack for multi-label classification"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, 
                 num_classes, dropout=0.1, pad_idx=0, use_cls_token=True):
        super().__init__()
        
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.pad_idx = pad_idx
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.fc = nn.Linear(d_model, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_padding_mask(self, src):
        """Create mask for padding tokens"""
        # src shape: [batch_size, seq_len]
        # Create a mask for PAD tokens: 1 for valid tokens, 0 for PAD
        mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        # mask shape: [batch_size, 1, 1, seq_len]
        return mask
                
    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
        """
        # Create attention mask for padding
        mask = self.create_padding_mask(src)
        
        # Embed tokens and add positional encoding
        x = self.embedding(src) * math.sqrt(self.d_model)  # Scale embeddings
        x = self.pos_encoding(x)
        
        # Store attention weights from each layer
        attention_weights = []
        
        # Apply transformer encoder layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Get the final representation
        if self.use_cls_token:
            # Use the [CLS] token (first token) for classification
            x = x[:, 0, :]
        else:
            # Global average pooling
            # Create a mask to exclude padding tokens from average
            src_mask = (src != self.pad_idx).float().unsqueeze(-1)
            # Apply mask and compute average
            x = (x * src_mask).sum(dim=1) / src_mask.sum(dim=1)
        
        # Apply classification head
        output = self.fc(x)
        
        return output, attention_weights

# EarlyStopping class
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=3, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_weights = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop
    
    def restore_weights(self, model):
        """Restore model to best weights"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logging.info("Restored model to best weights")

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
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Convert logits to predictions (binary with 0.5 threshold)
        preds = torch.sigmoid(logits) > 0.5
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='macro')
    
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
            
            # Store a sample of attention weights
            if len(all_attention_weights) < 5:  # Store just a few samples
                # Get the last layer's attention weights
                layer_weights = attention_weights[-1]  # Last layer
                # Get the first head's attention weights for the first sample in batch
                head_weights = layer_weights[0, 0].cpu().numpy()  # First head, first sample
                all_attention_weights.append(head_weights)
    
    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Apply thresholds (default 0.5 if not provided)
    if thresholds is None:
        thresholds = [0.5] * all_logits.shape[1]
    
    # Apply sigmoid and thresholds to get binary predictions
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    all_preds = np.zeros_like(all_probs)
    for i, threshold in enumerate(thresholds):
        all_preds[:, i] = all_probs[:, i] > threshold
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss / len(dataloader), f1, all_probs, all_preds, all_labels, all_attention_weights

def visualize_attention(attention_weights, sequences, preprocessor):
    """Visualize attention weights for sample sequences"""
    if not attention_weights:
        logging.warning("No attention weights to visualize")
        return
    
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights[0], cmap='viridis')
    
    # Get word tokens for the first sequence
    tokens = [preprocessor.idx2word.get(idx, '<UNK>') for idx in sequences[0].cpu().numpy()]
    tokens = [t for t in tokens if t != '<PAD>'][:20]  # Just show first 20 tokens
    
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    
    plt.colorbar()
    plt.title('Self-Attention Weights (First Head, First Layer)')
    plt.tight_layout()
    plt.show()
    
    logging.info("Attention visualization displayed")

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Settings - MODIFIED
    D_MODEL = 192  # Reduced from 256
    NUM_HEADS = 6  # Reduced from 8
    NUM_LAYERS = 4  # Reduced from 6
    D_FF = 768  # Reduced from 1024
    DROPOUT = 0.3  # Increased from 0.1
    BATCH_SIZE = 32  
    LEARNING_RATE = 5e-5  # Reduced from 1e-4
    N_EPOCHS = 30  # Increased for early stopping
    MAX_SEQ_LENGTH = 150  # Reduced from 200
    USE_CLS_TOKEN = True
    PATIENCE = 5  # Early stopping patience
    USE_DATA_AUGMENTATION = True  # Enable data augmentation
    WEIGHT_DECAY = 1e-4  # L2 regularization
    
    # Device - support for MPS (Apple Silicon), CUDA, or CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Load genre config
    logging.info("Loading genre configuration...")
    try:
        with open(os.path.join(data_dir, "genre_config.json"), "r") as f:
            genre_config = json.load(f)
        
        top_genres = genre_config["top_genres"]
        genre_weights = genre_config["genre_weights"]
        genre_columns = [f"genre_{genre}" for genre in top_genres]
        
        logging.info(f"Loaded configuration for {len(top_genres)} genres")
        logging.info(f"Genres: {top_genres}")
        
    except FileNotFoundError:
        logging.error("Genre configuration not found. Run genre preprocessing first.")
        return
    except Exception as e:
        logging.error(f"Error loading genre configuration: {str(e)}")
        return
    
    # Load data
    logging.info("Loading data...")
    try:
        train_df = pd.read_pickle(os.path.join(data_dir, "splits", "train.pkl"))
        val_df = pd.read_pickle(os.path.join(data_dir, "splits", "val.pkl"))
        test_df = pd.read_pickle(os.path.join(data_dir, "splits", "test.pkl"))
        logging.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
    except FileNotFoundError as e:
        logging.error(f"Data splits not found: {str(e)}")
        logging.error("Run genre preprocessing first to create the splits.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Ensure genre columns are numeric
    logging.info("Ensuring genre columns are numeric...")
    for df in [train_df, val_df, test_df]:
        for col in genre_columns:
            df[col] = df[col].astype(float)
    
    # Create text preprocessor
    logging.info("Creating text preprocessor...")
    preprocessor = TextPreprocessor(
        max_vocab_size=10000, 
        max_seq_length=MAX_SEQ_LENGTH,
        add_cls_token=USE_CLS_TOKEN
    )
    
    # Build vocabulary
    preprocessor.build_vocab(train_df['overview'])
    
    # Save preprocessor
    preprocessor.save(os.path.join(models_dir, "genre_transformer", "preprocessor.pkl"))
    
    # Create datasets - now with augmentation for training set
    logging.info("Creating datasets...")
    train_dataset = MovieGenreDataset(train_df, preprocessor, genre_columns, augment=USE_DATA_AUGMENTATION)
    val_dataset = MovieGenreDataset(val_df, preprocessor, genre_columns, augment=False)
    test_dataset = MovieGenreDataset(test_df, preprocessor, genre_columns, augment=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    logging.info("Creating Transformer encoder for multi-label classification...")
    model = TransformerEncoder(
        vocab_size=preprocessor.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        num_classes=len(genre_columns),
        dropout=DROPOUT,
        pad_idx=preprocessor.word2idx['<PAD>'],
        use_cls_token=USE_CLS_TOKEN
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model architecture
    logging.info(f"Model architecture:\n{model}")
    logging.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    # Create weighted loss function using genre weights
    pos_weight = torch.tensor([genre_weights[genre] for genre in top_genres], dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logging.info("Using weighted BCEWithLogitsLoss with the following weights:")
    for genre, weight in zip(top_genres, pos_weight.cpu().numpy()):
        logging.info(f"  {genre}: {weight:.4f}")
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=PATIENCE)
    
    # Train model
    logging.info("Training model...")
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    
    best_val_f1 = 0
    
    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)
        
        # Evaluate
        val_loss, val_f1, val_probs, val_preds, val_labels, val_attention = evaluate(
            model, val_dataloader, criterion, device
        )
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start_time
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        logging.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(models_dir, "genre_transformer", "best_model.pt"))
            logging.info(f"  New best model saved with val F1: {val_f1:.4f}")
        
        # Check for early stopping
        if early_stopping(val_f1, model):
            logging.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore_weights(model)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Training and Validation F1 Score')
    
    plt.tight_layout()
    plt.show()  # Show the plot instead of saving it
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_f1, test_probs, test_preds, test_labels, test_attention = evaluate(
        model, test_dataloader, criterion, device
    )
    logging.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    
    # Generate per-genre metrics
    per_genre_metrics = {}
    for i, genre in enumerate(top_genres):
        precision = precision_score(test_labels[:, i], test_preds[:, i])
        recall = recall_score(test_labels[:, i], test_preds[:, i])
        f1 = f1_score(test_labels[:, i], test_preds[:, i])
        
        per_genre_metrics[genre] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    # Log per-genre metrics
    logging.info("Per-genre metrics:")
    logging.info(f"{'Genre':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    logging.info("-" * 50)
    for genre, metrics in per_genre_metrics.items():
        logging.info(f"{genre:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    
    # Plot genre-wise F1 scores
    plt.figure(figsize=(10, 8))
    genres = list(per_genre_metrics.keys())
    f1_scores = [metrics['f1'] for metrics in per_genre_metrics.values()]
    
    # Sort genres by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_genres = [genres[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.barh(sorted_genres, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1 Score')
    plt.title('F1 Score by Genre')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.show()
    
    # Visualize attention weights for test set
    if test_attention:
        test_batch = next(iter(test_dataloader))
        visualize_attention(
            test_attention,
            test_batch['text'][:1],  # Just use the first sequence
            preprocessor
        )
    
    # Compare with previous models
    try:
        with open(os.path.join(results_dir, "model_comparison.json"), "r") as f:
            model_comparison = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        model_comparison = {}
    
    # Add this model's results
    model_comparison['transformer_improved'] = {
        'macro_f1': float(test_f1),
        'per_genre_f1': {genre: float(metrics['f1']) for genre, metrics in per_genre_metrics.items()},
        'epochs_trained': len(train_f1s),
        'early_stopped': len(train_f1s) < N_EPOCHS,
        'hyperparameters': {
            'd_model': D_MODEL,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS, 
            'd_ff': D_FF,
            'dropout': DROPOUT,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'batch_size': BATCH_SIZE,
            'max_seq_length': MAX_SEQ_LENGTH,
            'data_augmentation': USE_DATA_AUGMENTATION
        }
    }
    
    # Save model comparison
    with open(os.path.join(results_dir, "model_comparison.json"), "w") as f:
        json.dump(model_comparison, f, indent=2)
    
    logging.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()