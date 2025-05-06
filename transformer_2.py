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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_group_transformer.log"),
        logging.StreamHandler()
    ]
)

# Download NLTK resources - With better error handling
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
data_dir = os.path.join(base_dir, "processed_data")  # Adjusted for LSTM dataset structure
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(os.path.join(models_dir, "movie_group_transformer"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "movie_group_transformer"), exist_ok=True)

# Layer Dropout for Transformers
class LayerDropout(nn.Module):
    """Apply layer dropout to transformer layers"""
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
        
    def forward(self, x, residual=None):
        if self.training:
            if random.random() < self.p:
                if residual is not None:
                    return residual
                return x
        return x

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
def random_deletion(tokens, p=0.2):  # Increased from 0.1 to 0.2
    """Randomly delete words from the sentence with probability p"""
    # Handle empty or single token list
    if len(tokens) <= 1:
        return tokens
    
    # Randomly delete words with probability p
    new_tokens = []
    for token in tokens:
        if random.random() > p:
            new_tokens.append(token)
    
    # If we deleted all tokens, just return a random token
    if len(new_tokens) == 0:
        # Make sure tokens list is not empty before selecting random element
        if len(tokens) > 0:
            rand_int = random.randint(0, len(tokens)-1)
            return [tokens[rand_int]]
        else:
            # If tokens list is somehow empty, return a placeholder
            return ["<UNK>"]
    
    return new_tokens

def random_swap(tokens, n=2):  # Increased from 1 to 2
    """Randomly swap n pairs of words in the sentence"""
    # Handle case of empty or single token list
    if len(tokens) < 2:
        return tokens
    
    # Copy the token list to avoid modifying the original
    new_tokens = tokens.copy()
    
    # Perform n swaps (or fewer if n exceeds available pairs)
    n = min(n, len(new_tokens) // 2)  # Ensure we don't try more swaps than possible
    for _ in range(n):
        # Choose two random positions
        try:
            idx1, idx2 = random.sample(range(len(new_tokens)), 2)
            # Swap tokens
            new_tokens[idx1], new_tokens[idx2] = new_tokens[idx2], new_tokens[idx1]
        except ValueError:
            # This should not happen now, but just in case
            break
    
    return new_tokens

def random_mask(tokens, p=0.15):
    """Randomly mask tokens with <UNK> with probability p"""
    if not tokens:
        return tokens
    
    new_tokens = tokens.copy()
    for i in range(len(new_tokens)):
        if random.random() < p:
            new_tokens[i] = "<UNK>"
    
    return new_tokens

class MovieGroupDataset(Dataset):
    """Dataset for multilabel movie group classification with data augmentation"""
    
    def __init__(self, df, text_preprocessor, group_to_idx=None, augment=False):
        self.df = df
        self.preprocessor = text_preprocessor
        self.augment = augment
        
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
                        # Label smoothing: use 0.9 instead of 1.0
                        label_vector[self.group_to_idx[group]] = 0.9
            
            self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.df)
    
    def augment_text(self, text):
        """Apply text augmentation techniques"""
        tokens = self.preprocessor.tokenize(text)
        
        # Apply random augmentation - now with more diversity
        aug_type = random.random()
        if aug_type < 0.4:  # 40% chance for deletion (up from 30%)
            tokens = random_deletion(tokens, p=0.2)
        elif aug_type < 0.8:  # 40% chance for swap (up from 20%)
            tokens = random_swap(tokens, n=min(3, len(tokens)//3))
        else:  # 20% chance for masking (new)
            tokens = random_mask(tokens, p=0.15)
        
        # Convert back to text and then to sequence
        text = " ".join(tokens)
        return self.preprocessor.text_to_sequence(text)
    
    def __getitem__(self, idx):
        original_sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Apply augmentation during training if enabled (increased probability from 30% to 50%)
        if self.augment and random.random() < 0.5:
            # Get original text by joining tokens, removing PAD and CLS tokens
            tokens = [self.preprocessor.idx2word.get(idx, "") 
                     for idx in original_sequence 
                     if idx not in [0, 2]]  # Remove PAD and CLS tokens
            
            # Make sure we have some tokens before augmenting
            if tokens:
                overview = " ".join(tokens)
                sequence = self.augment_text(overview)
            else:
                sequence = original_sequence
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
        self.layer_norm = nn.LayerNorm(d_model)  # Added LayerNorm for better stability
    
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
        
        # Apply layer normalization first (Pre-LN Transformer architecture)
        query = self.layer_norm(query)
        
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
        self.layer_norm = nn.LayerNorm(d_model)  # Added LayerNorm
        
        # Activation with GELU instead of ReLU for better performance
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Apply layer normalization first (Pre-LN Transformer)
        x_norm = self.layer_norm(x)
        
        # Apply feed forward network
        x_ff = self.activation(self.linear1(x_norm))
        x_ff = self.dropout(x_ff)
        x_ff = self.linear2(x_ff)
        
        return x_ff

class EncoderLayer(nn.Module):
    """Encoder layer as in the Transformer paper with improved regularization"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, layer_dropout_rate=0.2):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        
        # Using higher dropout rates to combat overfitting, but capped at 0.9
        attn_dropout = min(dropout * 1.5, 0.9)
        ffn_dropout = min(dropout * 1.5, 0.9)
        
        self.dropout1 = nn.Dropout(attn_dropout)  # Higher dropout for attention output
        self.dropout2 = nn.Dropout(ffn_dropout)   # Higher dropout for ffn output
        
        # Layer dropout
        self.layer_dropout = LayerDropout(layer_dropout_rate)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
        """
        # Self-attention with layer dropout
        attn_output, self_attn_weights = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        
        # Apply layer dropout to attention - 20% chance to skip this sublayer
        x = x + self.layer_dropout(attn_output, x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        
        # Apply layer dropout to feed-forward - 20% chance to skip this sublayer
        x = x + self.layer_dropout(ff_output, x)
        
        return x, self_attn_weights

class TransformerEncoder(nn.Module):
    """Transformer Encoder stack for multi-label classification with anti-overfitting measures"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                num_classes, dropout=0.1, layer_dropout=0.2, pad_idx=0, use_cls_token=True,
                embedding_dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.pad_idx = pad_idx
        
        # Embedding layers with higher dropout
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)  # Higher dropout on embeddings
        
        # Encoder layers with layer dropout
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, layer_dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head with additional regularization
        self.pre_classifier = nn.Linear(d_model, d_model // 2)
        self.classifier_act = nn.GELU()
        self.classifier_dropout = nn.Dropout(dropout * 1.5)  # Higher dropout for classifier
        self.fc = nn.Linear(d_model // 2, num_classes)
        
        # Layer normalization for final output
        self.final_layer_norm = nn.LayerNorm(d_model)
        
        # Initialize parameters with smaller values to reduce initial overfitting
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters with smaller weights to combat overfitting"""
        for p in self.parameters():
            if p.dim() > 1:
                # Use smaller standard deviation for initialization
                nn.init.xavier_normal_(p, gain=0.7)
            elif p.dim() == 1:
                # Initialize biases to small positive value to encourage activations
                nn.init.constant_(p, 0.01)
    
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
        
        # Apply higher dropout to embeddings
        x = self.embedding_dropout(x)
        
        # Store attention weights from each layer
        attention_weights = []
        
        # Apply transformer encoder layers
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer normalization
        x = self.final_layer_norm(x)
        
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
        
        # Apply enhanced classification head with extra regularization
        x = self.pre_classifier(x)
        x = self.classifier_act(x)
        x = self.classifier_dropout(x)
        output = self.fc(x)
        
        return output, attention_weights

# Label smoothing loss for multi-label classification
class LabelSmoothingBCELoss(nn.Module):
    """
    Label smoothing for multi-label classification using BCE loss
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, pred, target):
        # Apply label smoothing - move targets away from 0 and 1
        smooth_targets = target.clone()
        smooth_targets[target > 0.5] = 1.0 - self.smoothing  # Move 1's to 0.9
        smooth_targets[target <= 0.5] = self.smoothing       # Move 0's to 0.1
        
        # Calculate BCE loss
        loss = self.criterion(pred, smooth_targets)
        
        # Return mean loss
        return loss.mean()

# EarlyStopping class
class EarlyStopping:
    """Early stopping to prevent overfitting with improved patience handling"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
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
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        return self.early_stop
    
    def restore_weights(self, model):
        """Restore model to best weights"""
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logging.info("Restored model to best weights")

def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=0.5):
    """Train model for one epoch with gradient clipping"""
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
        
        # Gradient clipping to prevent exploding gradients (reduced from 1.0 to 0.5)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Convert logits to predictions (binary with 0.5 threshold)
        preds = torch.sigmoid(logits) > 0.5
        all_preds.append(preds.detach().cpu().numpy())
        
        # Convert labels to binary (threshold 0.5) to handle label smoothing
        binary_labels = (labels > 0.5).float()
        all_labels.append(binary_labels.detach().cpu().numpy())
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Ensure both are integer type for F1 calculation
    all_preds = all_preds.astype(np.int32)
    all_labels = all_labels.astype(np.int32)
    
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
            
            # Convert labels to binary (threshold 0.5) to handle label smoothing
            binary_labels = (labels > 0.5).float()
            all_labels.append(binary_labels.cpu().numpy())
            
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
        threshold_float = float(threshold)
        all_preds[:, i] = all_probs[:, i] > threshold_float
    
    # Ensure both are integer type for F1 calculation
    all_preds = all_preds.astype(np.int32)
    all_labels = all_labels.astype(np.int32)
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss / len(dataloader), f1, all_probs, all_preds, all_labels, all_attention_weights

def find_optimal_thresholds(probs, labels):
    """Find optimal thresholds for each group using validation data"""
    n_classes = probs.shape[1]
    thresholds = []
    
    for i in range(n_classes):
        # Try different thresholds
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] > threshold).astype(int)
            
            # Convert labels to binary for threshold calculation
            binary_labels = (labels[:, i] > 0.5).astype(int)
            
            f1 = f1_score(binary_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        thresholds.append(float(best_threshold))
    
    return thresholds

def visualize_attention(attention_weights, sequences, preprocessor, save_path=None):
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
    plt.title('Self-Attention Weights (First Head, Last Layer)')
    plt.tight_layout()
    
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
    
    # Settings - MODIFIED to combat overfitting
    D_MODEL = 128          # Reduced from 192 to 128
    NUM_HEADS = 4          # Reduced from 6 to 4
    NUM_LAYERS = 3         # Reduced from 4 to 3
    D_FF = 512            # Reduced from 768 to 512
    DROPOUT = 0.6          # Increased but kept below 0.7 to avoid overflow when multiplied
    EMBEDDING_DROPOUT = 0.5  # Separate higher dropout for embeddings
    LAYER_DROPOUT = 0.2     # Add layer dropout
    BATCH_SIZE = 16         # Reduced from 32 to 16
    LEARNING_RATE = 2e-5   # Reduced from 5e-5
    N_EPOCHS = 25          # Reduced from 30 
    MAX_SEQ_LENGTH = 128   # Reduced from 150
    USE_CLS_TOKEN = True
    PATIENCE = 7           # Increased from 5 to 7
    USE_DATA_AUGMENTATION = True
    WEIGHT_DECAY = 1e-3    # Increased from 1e-4 to 1e-3
    LABEL_SMOOTHING = 0.1  # Add label smoothing
    MAX_GRAD_NORM = 0.5    # Reduced from 1.0
    
    # Device - support for MPS (Apple Silicon), CUDA, or CPU
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
    except FileNotFoundError as e:
        logging.error(f"Data splits not found: {str(e)}")
        logging.error("Please ensure the data files are in the correct location")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Create text preprocessor
    logging.info("Creating text preprocessor...")
    preprocessor_path = os.path.join(models_dir, "movie_group_transformer", "preprocessor.pkl")
    
    if os.path.exists(preprocessor_path):
        # Load existing preprocessor
        preprocessor = TextPreprocessor.load(preprocessor_path)
        # Use the sequence length from the loaded preprocessor
        MAX_SEQ_LENGTH = preprocessor.max_seq_length
        logging.info(f"Using sequence length from loaded preprocessor: {MAX_SEQ_LENGTH}")
    else:
        # Create new preprocessor and build vocabulary
        preprocessor = TextPreprocessor(
            max_vocab_size=8000,  # Reduced from 10000
            max_seq_length=MAX_SEQ_LENGTH,
            add_cls_token=USE_CLS_TOKEN
        )
        preprocessor.build_vocab(train_df['overview'])
        preprocessor.save(preprocessor_path)
    
    # Create datasets - with augmentation for training set
    logging.info("Creating datasets...")
    train_dataset = MovieGroupDataset(
        train_df, preprocessor, group_to_idx=None, 
        augment=USE_DATA_AUGMENTATION
    )
    
    # Use the group_to_idx from the training set for consistency
    val_dataset = MovieGroupDataset(
        val_df, preprocessor, group_to_idx=train_dataset.group_to_idx,
        augment=False
    )
    
    test_dataset = MovieGroupDataset(
        test_df, preprocessor, group_to_idx=train_dataset.group_to_idx,
        augment=False
    )
    
    # Save group_to_idx mapping for inference
    with open(os.path.join(models_dir, "movie_group_transformer", "group_to_idx.json"), 'w') as f:
        json.dump(train_dataset.group_to_idx, f)
    
    # Create dataloaders with smaller batch size
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model with enhanced regularization
    logging.info("Creating Transformer encoder with anti-overfitting measures...")
    model = TransformerEncoder(
        vocab_size=preprocessor.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        num_classes=train_dataset.num_groups,
        dropout=DROPOUT,
        layer_dropout=LAYER_DROPOUT,
        pad_idx=preprocessor.word2idx['<PAD>'],
        use_cls_token=USE_CLS_TOKEN,
        embedding_dropout=EMBEDDING_DROPOUT
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model architecture
    logging.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer with weight decay (L2 regularization)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    
    # Create learning rate scheduler - Cosine annealing with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)
    
    # Create loss function with label smoothing
    criterion = LabelSmoothingBCELoss(smoothing=LABEL_SMOOTHING)
    
    # Initialize early stopping with more patience
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
    
    # Train model
    logging.info("Training model with anti-overfitting measures...")
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    best_val_f1 = 0
    
    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        
        # Train with gradient clipping
        train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, criterion, device, max_grad_norm=MAX_GRAD_NORM)
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
        scheduler.step()
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f" Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logging.info(f" Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        logging.info(f" Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(models_dir, "movie_group_transformer", "best_model.pt"))
            logging.info(f" New best model saved with val F1: {val_f1:.4f}")
            
            # Find optimal thresholds
            optimal_thresholds = find_optimal_thresholds(val_probs, val_labels)
            # Save thresholds
            with open(os.path.join(models_dir, "movie_group_transformer", "optimal_thresholds.json"), 'w') as f:
                json.dump(optimal_thresholds, f)
        
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
    plt.savefig(os.path.join(results_dir, "movie_group_transformer", "training_curves.png"))
    
    # Load best model for testing
    logging.info("Loading best model for evaluation on test set...")
    model_path = os.path.join(models_dir, "movie_group_transformer", "best_model.pt")
    thresholds_path = os.path.join(models_dir, "movie_group_transformer", "optimal_thresholds.json")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            logging.info("Best model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading best model: {str(e)}")
    else:
        logging.warning("No saved model found, using current model state")
    
    # Load optimal thresholds
    if os.path.exists(thresholds_path):
        try:
            with open(thresholds_path, 'r') as f:
                optimal_thresholds = json.load(f)
            optimal_thresholds = [float(t) for t in optimal_thresholds]
            logging.info(f"Loaded {len(optimal_thresholds)} optimal thresholds")
        except Exception as e:
            logging.error(f"Error loading optimal thresholds: {str(e)}")
            optimal_thresholds = None
    else:
        logging.warning("No optimal thresholds file found, using default threshold of 0.5")
        optimal_thresholds = None
    
    # Evaluate on test set
    test_loss, test_f1, test_probs, test_preds, test_labels, test_attention = evaluate(
        model, test_dataloader, criterion, device, optimal_thresholds
    )
    logging.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    
    # Create reverse mapping for result interpretation
    idx_to_group = {idx: group for group, idx in train_dataset.group_to_idx.items()}
    
    # Generate per-group metrics
    per_group_metrics = {}
    for i, group_idx in enumerate(train_dataset.group_to_idx.values()):
        group = idx_to_group[group_idx]
        precision = precision_score(test_labels[:, i] > 0.5, test_preds[:, i])
        recall = recall_score(test_labels[:, i] > 0.5, test_preds[:, i])
        f1 = f1_score(test_labels[:, i] > 0.5, test_preds[:, i])
        per_group_metrics[group] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    # Log per-group metrics
    logging.info("Per-group metrics:")
    logging.info(f"{'Group':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    logging.info("-" * 55)
    for group, metrics in sorted(per_group_metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        logging.info(f"{group[:20]:<20} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f}")
    
    # Save per-group metrics
    with open(os.path.join(results_dir, "movie_group_transformer", "per_group_metrics.json"), 'w') as f:
        json.dump(per_group_metrics, f, indent=2)
    
    # Plot group-wise F1 scores (top and bottom 10)
    plt.figure(figsize=(12, 10))
    
    groups = list(per_group_metrics.keys())
    f1_scores = [metrics['f1'] for metrics in per_group_metrics.values()]
    
    # Sort groups by F1 score
    sorted_indices = np.argsort(f1_scores)
    
    # Plot top 10 groups
    plt.subplot(1, 2, 1)
    top_indices = sorted_indices[-10:][::-1]
    top_groups = [groups[i] for i in top_indices]
    top_f1_scores = [f1_scores[i] for i in top_indices]
    plt.barh(top_groups, top_f1_scores, color='skyblue')
    plt.xlabel('F1 Score')
    plt.title('Top 10 Groups by F1 Score')
    plt.xlim(0, 1.0)
    
    # Plot bottom 10 groups
    plt.subplot(1, 2, 2)
    bottom_indices = sorted_indices[:10]
    bottom_groups = [groups[i] for i in bottom_indices]
    bottom_f1_scores = [f1_scores[i] for i in bottom_indices]
    plt.barh(bottom_groups, bottom_f1_scores, color='salmon')
    plt.xlabel('F1 Score')
    plt.title('Bottom 10 Groups by F1 Score')
    plt.xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "movie_group_transformer", "group_f1_scores.png"))
    
    # Visualize attention weights
    if test_attention:
        test_batch = next(iter(test_dataloader))
        viz_path = os.path.join(results_dir, "movie_group_transformer", "attention_weights.png")
        visualize_attention(
            test_attention,
            test_batch['text'][:1],  # Just use the first sequence
            preprocessor,
            save_path=viz_path
        )
    
    # Create model comparison info
    model_info = {
        'transformer_anti_overfitting': {
            'macro_f1': float(test_f1),
            'epochs_trained': len(train_f1s),
            'early_stopped': len(train_f1s) < N_EPOCHS,
            'hyperparameters': {
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'd_ff': D_FF,
                'dropout': DROPOUT,
                'layer_dropout': LAYER_DROPOUT,
                'embedding_dropout': EMBEDDING_DROPOUT,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'batch_size': BATCH_SIZE,
                'max_seq_length': MAX_SEQ_LENGTH,
                'data_augmentation': USE_DATA_AUGMENTATION,
                'label_smoothing': LABEL_SMOOTHING,
                'max_grad_norm': MAX_GRAD_NORM
            }
        }
    }
    
    # Save model info
    with open(os.path.join(results_dir, "movie_group_transformer", "model_info.json"), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Create inference function
    def predict_groups(text, top_k=5):
        """Predict movie groups for a given text"""
        # Preprocess text
        sequence = preprocessor.text_to_sequence(text)
        sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions
        with torch.no_grad():
            logits, _ = model(sequence)
            probs = torch.sigmoid(logits)
        
        # Get probabilities and convert to numpy
        probs = probs.squeeze(0).cpu().numpy()
        
        # Apply optimal thresholds if available
        preds = np.zeros_like(probs)
        for i, threshold in enumerate(optimal_thresholds if optimal_thresholds else [0.5] * len(probs)):
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
    
    # Test inference with a sample
    sample_text = "A superhero with special powers fights against evil forces to save the world"
    logging.info(f"Example prediction for: '{sample_text}'")
    predictions = predict_groups(sample_text)
    for group, prob in predictions:
        logging.info(f"{group}: {prob:.4f}")
    
    logging.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()