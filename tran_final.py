import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, multilabel_confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("movie_group_transformer.log"),
        logging.StreamHandler()
    ]
)

# Download NLTK resources directly without using custom directories to avoid path issues
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    logging.info("NLTK resources downloaded successfully")
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {str(e)}")
    logging.warning("Will use fallback tokenization if NLTK tokenization fails")

# Set paths
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = script_dir
data_dir = os.path.join(base_dir, "processed_data")
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
            # Simple whitespace tokenization as primary method to avoid NLTK issues
            tokens = text.lower().split()
            
            # Filter tokens
            tokens = [
                token for token in tokens
                if token.isalpha() and token not in self.stop_words
            ]
            
            # Try lemmatization if available
            if self.lemmatizer:
                try:
                    tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
                except:
                    pass
            
            return tokens
        except Exception as e:
            # Ultimate fallback
            logging.warning(f"Tokenization failed: {str(e)}. Using minimal fallback.")
            if isinstance(text, str):
                return [w.lower() for w in text.split()]
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
def random_deletion(tokens, p=0.2):
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

def random_swap(tokens, n=2):
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
        
        # Track label counts for calculating class weights and sample weights
        self.label_counts = np.zeros(self.num_groups)
        
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
                        self.label_counts[self.group_to_idx[group]] += 1
            
            self.labels.append(label_vector)
        
        # Calculate sample weights for balanced sampling
        self.sample_weights = self._calculate_sample_weights()
    
    def _calculate_sample_weights(self):
        """Calculate sample weights for balanced sampling"""
        # Convert multi-hot encoded labels to a list of weights per sample
        weights = []
        
        # Calculate inverse class frequency
        class_weights = np.zeros(self.num_groups)
        for i in range(self.num_groups):
            if self.label_counts[i] > 0:
                class_weights[i] = len(self.df) / self.label_counts[i]
            else:
                class_weights[i] = 1.0
        
        # Normalize to sum to 1
        class_weights = class_weights / np.sum(class_weights)
        
        # For each sample, assign weight based on its labels
        for label_vector in self.labels:
            label_indices = torch.nonzero(label_vector).squeeze().cpu().numpy()
            
            # Handle case of single label
            if len(np.shape(label_indices)) == 0:  # Changed to np.shape to handle scalar case better
                label_indices = np.array([label_indices])
                
            if len(label_indices) > 0:
                # Assign weight as average of class weights for all labels
                weight = np.mean([class_weights[idx] for idx in label_indices])
            else:
                weight = 1.0  # Default weight for samples with no labels
            
            weights.append(weight)
            
        return weights
    
    def get_class_weights(self):
        """Calculate class weights for weighted loss function"""
        # Convert multi-hot encoded labels to class counts
        label_counts = self.label_counts
        
        # Calculate class weights (inverse of frequency)
        class_weights = []
        for count in label_counts:
            if count > 0:
                # Use inverse frequency with smoothing to avoid extreme weights
                weight = len(self.df) / (count * self.num_groups)
                # Apply log smoothing for extreme imbalances
                weight = 1 + np.log(weight)
                class_weights.append(weight)
            else:
                # If class doesn't appear, use a default weight
                class_weights.append(1.0)
        
        # Log class weights for each class
        logging.info("Class weights:")
        for i, weight in enumerate(class_weights):
            logging.info(f"  Class {i}: {weight:.4f} (count: {label_counts[i]})")
            
        return torch.tensor(class_weights, dtype=torch.float)
    
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
            # Use clone().detach() as recommended to avoid the warning
            'label': label.clone().detach()
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
        
        # Add batch normalization for better training stability
        self.bn = nn.BatchNorm1d(d_ff)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # Apply layer normalization first (Pre-LN Transformer)
        x_norm = self.layer_norm(x)
        
        # Get batch size and sequence length
        batch_size, seq_len, _ = x.shape
        
        # Apply feed forward network
        x_ff = self.activation(self.linear1(x_norm))
        
        # Reshape for batch norm (batch_norm expects [N,C,*])
        x_ff = x_ff.transpose(1, 2)  # [batch_size, d_ff, seq_len]
        x_ff = self.bn(x_ff)
        x_ff = x_ff.transpose(1, 2)  # [batch_size, seq_len, d_ff]
        
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
        
        # Add batch normalization for better stability
        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
            mask: Tensor, shape [batch_size, 1, seq_len, seq_len]
        """
        # Self-attention with layer dropout
        attn_output, self_attn_weights = self.self_attn(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        
        # Apply layer dropout to attention
        attn_output = self.layer_dropout(attn_output, x)
        x = x + attn_output
        
        # Apply batch normalization
        batch_size, seq_len, d_model = x.shape
        x_trans = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x_trans = self.bn1(x_trans)
        x = x_trans.transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        
        # Apply layer dropout to feed-forward
        ff_output = self.layer_dropout(ff_output, x)
        x = x + ff_output
        
        # Apply batch normalization
        x_trans = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x_trans = self.bn2(x_trans)
        x = x_trans.transpose(1, 2)  # [batch_size, seq_len, d_model]
        
        return x, self_attn_weights

class FocalLoss(nn.Module):
    """
    Focal Loss for better handling of hard examples and class imbalance
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply sigmoid activation to get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        ce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        
        # Calculate focal loss component
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TransformerEncoder(nn.Module):
    """Transformer Encoder stack for multi-label classification with anti-overfitting measures"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length,
                num_classes, dropout=0.1, layer_dropout=0.2, pad_idx=0, use_cls_token=True,
                embedding_dropout=0.3, freeze_embedding_layers=False):
        super().__init__()
        self.d_model = d_model
        self.use_cls_token = use_cls_token
        self.pad_idx = pad_idx
        
        # Embedding layers with higher dropout
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)  # Higher dropout on embeddings
        
        # Option to freeze embedding layers to prevent overfitting
        if freeze_embedding_layers:
            logging.info("Freezing embedding layers")
            for param in self.embedding.parameters():
                param.requires_grad = False
        
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
        
        # Batch normalization for classifier
        self.bn_classifier = nn.BatchNorm1d(d_model // 2)
        
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
        
        # Apply batch normalization
        x = self.bn_classifier(x)
        
        x = self.classifier_act(x)
        x = self.classifier_dropout(x)
        output = self.fc(x)
        
        return output, attention_weights

def weighted_bce_loss(outputs, targets, pos_weight):
    """
    Custom weighted BCE loss for handling class imbalance
    """
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss(outputs, targets)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, class_weights=None, max_grad_norm=0.5, scheduler_type="linear_warmup"):
    """Train model for one epoch with gradient clipping"""
    model.train()
    epoch_loss = 0
    
    # For tracking F1 score
    all_preds = []
    all_labels = []
    
    # Track learning rates through epoch for plotting
    lr_history = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(text)  # Ignore attention weights during training
        
        # Calculate loss with class weights if provided
        if isinstance(criterion, FocalLoss):
            loss = criterion(logits, labels)
        elif class_weights is not None:
            pos_weight = class_weights.to(device)
            loss = weighted_bce_loss(logits, labels, pos_weight)
        else:
            loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        # Update parameters
        optimizer.step()
        
        # Update learning rate - different behavior for different schedulers
        if scheduler_type == "linear_warmup" or scheduler_type == "cosine_warmup":
            scheduler.step()
        # For ReduceLROnPlateau, we'll update after validation
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
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
    
    # Calculate F1 scores - both macro and micro
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss / len(dataloader), macro_f1, micro_f1, weighted_f1, lr_history

def evaluate(model, dataloader, criterion, device, thresholds=None, class_weights=None):
    """Evaluate model on dataloader"""
    model.eval()
    epoch_loss = 0
    
    all_logits = []
    all_probs = []
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
            
            # Calculate loss with class weights if provided
            if isinstance(criterion, FocalLoss):
                loss = criterion(logits, labels)
            elif class_weights is not None:
                pos_weight = class_weights.to(device)
                loss = weighted_bce_loss(logits, labels, pos_weight)
            else:
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
    
    # Apply sigmoid to get probabilities
    all_probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    
    # Apply thresholds (default 0.5 if not provided)
    if thresholds is None:
        thresholds = [0.5] * all_logits.shape[1]
    
    # Apply thresholds to get binary predictions
    all_preds = np.zeros_like(all_probs)
    for i, threshold in enumerate(thresholds):
        threshold_float = float(threshold)
        all_preds[:, i] = all_probs[:, i] > threshold_float
    
    # Ensure both are integer type for F1 calculation
    all_preds = all_preds.astype(np.int32)
    all_labels = all_labels.astype(np.int32)
    
    # Calculate F1 scores
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return epoch_loss / len(dataloader), macro_f1, micro_f1, weighted_f1, all_probs, all_labels, all_preds, all_attention_weights

def compute_class_thresholds(probs, labels, search_step=0.05):
    """
    Compute optimal threshold for each class based on F1 score
    """
    n_classes = probs.shape[1]
    thresholds = []
    
    for i in range(n_classes):
        class_probs = probs[:, i]
        class_labels = labels[:, i]
        
        # Skip classes with no positive examples
        if np.sum(class_labels) == 0:
            thresholds.append(0.3)  # Default threshold
            continue
            
        best_f1 = 0
        best_threshold = 0.3  # Default
        
        # Search for optimal threshold
        for threshold in np.arange(0.1, 0.9, search_step):
            class_preds = (class_probs > threshold).astype(float)
            f1 = f1_score(class_labels, class_preds, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        thresholds.append(best_threshold)
        
    return np.array(thresholds)

def analyze_predictions(test_preds, test_labels, idx_to_label):
    """Analyze predictions to identify potential issues"""
    # Convert to numpy arrays
    preds = np.array(test_preds)
    labels = np.array(test_labels)
    
    # Calculate per-class metrics
    precision = []
    recall = []
    f1 = []
    
    n_classes = len(idx_to_label)
    
    for i in range(n_classes):
        # Extract class predictions
        class_preds = preds[:, i]
        class_labels = labels[:, i]
        
        # Calculate metrics
        true_pos = np.sum((class_preds == 1) & (class_labels == 1))
        false_pos = np.sum((class_preds == 1) & (class_labels == 0))
        false_neg = np.sum((class_preds == 0) & (class_labels == 1))
        
        # Handle division by zero
        p = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        r = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        precision.append(p)
        recall.append(r)
        f1.append(f)
    
    # Create a dataframe for analysis
    analysis_df = pd.DataFrame({
        'Class': [idx_to_label[i] for i in range(n_classes)],
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Support': np.sum(labels, axis=0)
    })
    
    return analysis_df

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
    N_EPOCHS = 25          # Set number of epochs to train for
    MAX_SEQ_LENGTH = 128   # Reduced from 150
    USE_CLS_TOKEN = True
    USE_DATA_AUGMENTATION = True
    WEIGHT_DECAY = 1e-3    # Increased from 1e-4 to 1e-3
    LABEL_SMOOTHING = 0.1  # Add label smoothing
    MAX_GRAD_NORM = 0.5    # Reduced from 1.0
    
    # New anti-overfitting settings (similar to BERT model)
    USE_FOCAL_LOSS = True        # Use focal loss instead of BCE
    FOCAL_GAMMA = 2.0            # Focal loss gamma parameter
    USE_CLASS_WEIGHTS = True     # Use class weights to combat imbalance
    USE_BALANCED_SAMPLING = True # Use weighted random sampler
    USE_DYNAMIC_THRESHOLDS = True # Dynamic threshold per class
    FREEZE_EMBEDDING_LAYERS = False # Option to freeze embedding layers
    EARLY_STOPPING_PATIENCE = 3  # Early stopping patience
    
    # Scheduler settings
    SCHEDULER_TYPE = "linear_warmup"  # Options: "linear_warmup", "cosine_warmup", "reduce_on_plateau"
    WARMUP_PROPORTION = 0.1  # Proportion of training steps for warmup
    
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
    
    # Calculate class weights from training data to address class imbalance
    if USE_CLASS_WEIGHTS:
        class_weights = train_dataset.get_class_weights()
        logging.info(f"Class weights will be applied during training")
    else:
        class_weights = None
    
    # Create dataloaders with balanced sampling for training
    if USE_BALANCED_SAMPLING:
        logging.info("Using balanced sampling strategy")
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    else:
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
        embedding_dropout=EMBEDDING_DROPOUT,
        freeze_embedding_layers=FREEZE_EMBEDDING_LAYERS
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model architecture
    logging.info(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Percentage frozen: {(1 - trainable_params/total_params) * 100:.2f}%")
    
    # Create optimizer with weight decay (L2 regularization)
    # Use different learning rates for embedding and other layers
    if FREEZE_EMBEDDING_LAYERS:
        # Only optimize non-embedding parameters
        optimizer = optim.AdamW(
            [p for n, p in model.named_parameters() if 'embedding' not in n], 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY, 
            betas=(0.9, 0.999), 
            eps=1e-8
        )
    else:
        # Two parameter groups with different learning rates
        embedding_params = {'params': [p for n, p in model.named_parameters() if 'embedding' in n], 'lr': LEARNING_RATE * 0.1}
        other_params = {'params': [p for n, p in model.named_parameters() if 'embedding' not in n]}
        optimizer = optim.AdamW([embedding_params, other_params], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create learning rate scheduler based on type
    total_steps = len(train_dataloader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_PROPORTION)
    
    if SCHEDULER_TYPE == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logging.info(f"Using linear warmup scheduler with {warmup_steps} warmup steps")
    
    elif SCHEDULER_TYPE == "cosine_warmup":
        # Cosine Annealing with Warm Restarts
        # T_0 is the number of iterations for the first restart
        # T_mult is the factor by which T_i increases after each restart
        t_0 = len(train_dataloader)  # One epoch
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=t_0,
            T_mult=2,  # Double the cycle length after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        logging.info(f"Using cosine annealing with warm restarts scheduler")
    
    elif SCHEDULER_TYPE == "reduce_on_plateau":
        # Reduce learning rate when a metric has stopped improving
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # Lower val_loss is better
            factor=0.5,  # Multiply LR by 0.5 when reducing
            patience=1,  # Wait for 1 epoch for improvement
            verbose=True,
            min_lr=1e-6
        )
        logging.info(f"Using reduce on plateau scheduler")
    
    # Create loss function based on settings
    if USE_FOCAL_LOSS:
        logging.info(f"Using Focal Loss with gamma={FOCAL_GAMMA}")
        criterion = FocalLoss(gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize class thresholds with default value
    class_thresholds = np.ones(train_dataset.num_groups) * 0.3
    
    # Train model
    logging.info("Training model with anti-overfitting measures...")
    logging.info(f"Using dynamic thresholds: {USE_DYNAMIC_THRESHOLDS}")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    train_macro_f1s = []
    train_micro_f1s = []
    train_weighted_f1s = []
    val_macro_f1s = []
    val_micro_f1s = []
    val_weighted_f1s = []
    lr_history = []
    
    best_val_f1 = 0
    best_val_loss = float('inf')
    no_improve_epochs = 0
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Train with class weights
        train_loss, train_macro_f1, train_micro_f1, train_weighted_f1, epoch_lr_history = train_epoch(
            model, train_dataloader, optimizer, scheduler, criterion, device, 
            class_weights=class_weights if USE_CLASS_WEIGHTS else None,
            max_grad_norm=MAX_GRAD_NORM,
            scheduler_type=SCHEDULER_TYPE
        )
        
        train_losses.append(train_loss)
        train_macro_f1s.append(train_macro_f1)
        train_micro_f1s.append(train_micro_f1)
        train_weighted_f1s.append(train_weighted_f1)
        lr_history.extend(epoch_lr_history)
        
        # Evaluate with class weights and thresholds
        val_loss, val_macro_f1, val_micro_f1, val_weighted_f1, val_probs, val_labels, val_preds, val_attention = evaluate(
            model, val_dataloader, criterion, device, 
            thresholds=class_thresholds if USE_DYNAMIC_THRESHOLDS else None,
            class_weights=class_weights if USE_CLASS_WEIGHTS else None
        )
        
        val_losses.append(val_loss)
        val_macro_f1s.append(val_macro_f1)
        val_micro_f1s.append(val_micro_f1)
        val_weighted_f1s.append(val_weighted_f1)
        
        # Update dynamic thresholds based on validation results
        if USE_DYNAMIC_THRESHOLDS:
            class_thresholds = compute_class_thresholds(val_probs, val_labels)
            logging.info(f"Updated class thresholds: min={min(class_thresholds):.2f}, max={max(class_thresholds):.2f}, mean={np.mean(class_thresholds):.2f}")
        
        # Update ReduceLROnPlateau scheduler based on validation loss if using that scheduler
        if SCHEDULER_TYPE == "reduce_on_plateau":
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train Macro F1: {train_macro_f1:.4f}, Train Micro F1: {train_micro_f1:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Macro F1: {val_macro_f1:.4f}, Val Micro F1: {val_micro_f1:.4f}")
        logging.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Save best model based on F1 score
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            
            # FIXED: Save model dictionary properly for easy loading
            # Only save state_dict and use torch.save directly instead of checkpoint dictionary
            torch.save(model.state_dict(), os.path.join(models_dir, "movie_group_transformer", "best_model.pt"))
            
            # Save other information separately as JSON
            best_model_info = {
                'epoch': epoch,
                'val_loss': float(val_loss),
                'val_macro_f1': float(val_macro_f1),
                'val_micro_f1': float(val_micro_f1),
                'val_weighted_f1': float(val_weighted_f1)
            }
            
            # Save thresholds separately
            np.save(os.path.join(models_dir, "movie_group_transformer", "class_thresholds.npy"), class_thresholds)
            
            # Save best model info as JSON
            with open(os.path.join(models_dir, "movie_group_transformer", "best_model_info.json"), 'w') as f:
                json.dump(best_model_info, f, indent=2)
                
            logging.info(f"  New best model saved with val F1: {val_macro_f1:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"  No improvement for {no_improve_epochs} epochs")
        
        # Early stopping check
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot Macro F1 score
    plt.subplot(2, 2, 2)
    plt.plot(train_macro_f1s, label='Train Macro F1')
    plt.plot(val_macro_f1s, label='Val Macro F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Macro F1 Score')
    
    # Plot Micro F1 score
    plt.subplot(2, 2, 3)
    plt.plot(train_micro_f1s, label='Train Micro F1')
    plt.plot(val_micro_f1s, label='Val Micro F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Micro F1 Score')
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(lr_history)
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate ({SCHEDULER_TYPE})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "movie_group_transformer", "training_curves.png"))
    
    # Load best model for testing
    logging.info("Loading best model for evaluation on test set...")
    model_path = os.path.join(models_dir, "movie_group_transformer", "best_model.pt")
    thresholds_path = os.path.join(models_dir, "movie_group_transformer", "class_thresholds.npy")
    
    # FIXED: Improved model loading to avoid pickle/serialization issues
    try:
        if os.path.exists(model_path):
            # Load only the state_dict, not a complete checkpoint
            model.load_state_dict(torch.load(model_path))
            logging.info("Best model loaded successfully")
            
            # Load thresholds if available
            if os.path.exists(thresholds_path):
                optimal_thresholds = np.load(thresholds_path)
                logging.info(f"Loaded optimal thresholds: min={min(optimal_thresholds):.2f}, max={max(optimal_thresholds):.2f}")
            else:
                optimal_thresholds = None
                logging.warning("No optimal thresholds file found, using default threshold of 0.5")
        else:
            optimal_thresholds = None
            logging.warning("No saved model found, using current model state")
    except Exception as e:
        logging.error(f"Error loading best model: {str(e)}")
        logging.warning("Continuing with current model state")
        optimal_thresholds = class_thresholds if USE_DYNAMIC_THRESHOLDS else None
    
    # Evaluate on test set
    test_loss, test_macro_f1, test_micro_f1, test_weighted_f1, test_probs, test_labels, test_preds, test_attention = evaluate(
        model, test_dataloader, criterion, device, 
        thresholds=optimal_thresholds,
        class_weights=class_weights if USE_CLASS_WEIGHTS else None
    )
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test Macro F1: {test_macro_f1:.4f}")
    logging.info(f"Test Micro F1: {test_micro_f1:.4f}")
    logging.info(f"Test Weighted F1: {test_weighted_f1:.4f}")
    
    # Create reverse mapping for result interpretation
    idx_to_group = {idx: group for group, idx in train_dataset.group_to_idx.items()}
    
    # Generate per-group metrics analysis
    analysis_df = analyze_predictions(test_preds, test_labels, idx_to_group)
    
    # Save analysis to CSV
    analysis_df.to_csv(os.path.join(results_dir, "movie_group_transformer", "class_analysis.csv"), index=False)
    
    # Log analysis results
    logging.info("Class-wise analysis:")
    for _, row in analysis_df.iterrows():
        logging.info(f"  {row['Class']}: Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, Support={row['Support']}")
    
    # Generate confusion matrix per class and save as image
    plt.figure(figsize=(15, 12))
    conf_matrices = multilabel_confusion_matrix(test_labels, test_preds)
    
    # Plot confusion matrices for each class (top 12 classes by support)
    top_classes = analysis_df.sort_values('Support', ascending=False).head(12)
    top_indices = [train_dataset.group_to_idx[group] for group in top_classes['Class']]
    
    rows = 4
    cols = 3
    for i, idx in enumerate(top_indices):
        if i >= rows * cols:
            break
            
        plt.subplot(rows, cols, i+1)
        sns.heatmap(conf_matrices[idx], annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        threshold = optimal_thresholds[idx] if optimal_thresholds is not None else 0.5
        plt.title(f'{idx_to_group[idx]} (T={threshold:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "movie_group_transformer", "confusion_matrices.png"))
    
    # Create class distribution plot
    plt.figure(figsize=(12, 8))
    
    # Get class frequencies
    class_support = analysis_df.sort_values('Support', ascending=False)
    top_20_classes = class_support.head(20)
    
    # Plot class support
    plt.subplot(2, 1, 1)
    plt.bar(top_20_classes['Class'], top_20_classes['Support'], color='skyblue')
    plt.xticks(rotation=90)
    plt.title('Top 20 Classes by Support')
    plt.ylabel('Number of Samples')
    
    # Plot class F1 scores
    plt.subplot(2, 1, 2)
    plt.bar(top_20_classes['Class'], top_20_classes['F1'], color='salmon')
    plt.xticks(rotation=90)
    plt.title('F1 Scores for Top 20 Classes')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "movie_group_transformer", "class_distribution.png"))
    
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
        'transformer_enhanced_with_anti_overfitting': {
            'macro_f1': float(test_macro_f1),
            'micro_f1': float(test_micro_f1),
            'weighted_f1': float(test_weighted_f1),
            'epochs_trained': epoch + 1,  # Account for 0-indexing
            'best_epoch': epoch if not 'best_model_info' in locals() else best_model_info.get('epoch', epoch),
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
                'max_grad_norm': MAX_GRAD_NORM,
                'use_focal_loss': USE_FOCAL_LOSS,
                'focal_gamma': FOCAL_GAMMA if USE_FOCAL_LOSS else None,
                'use_class_weights': USE_CLASS_WEIGHTS,
                'use_balanced_sampling': USE_BALANCED_SAMPLING,
                'use_dynamic_thresholds': USE_DYNAMIC_THRESHOLDS,
                'freeze_embedding_layers': FREEZE_EMBEDDING_LAYERS,
                'scheduler_type': SCHEDULER_TYPE,
                'warmup_proportion': WARMUP_PROPORTION,
                'early_stopping_patience': EARLY_STOPPING_PATIENCE
            },
            'anti_overfitting_techniques': [
                "Reduced model size (parameters)",
                "Increased dropout rates",
                "Layer dropout",
                "Batch normalization",
                "Gradient clipping",
                "Weight decay (L2 regularization)",
                "Learning rate scheduling with warmup",
                "Label smoothing",
                "Data augmentation",
                "Focal Loss" if USE_FOCAL_LOSS else "BCE Loss",
                "Class weighting" if USE_CLASS_WEIGHTS else "No class weighting", 
                "Balanced sampling" if USE_BALANCED_SAMPLING else "Random sampling",
                "Dynamic thresholds per class" if USE_DYNAMIC_THRESHOLDS else "Fixed thresholds",
                "Early stopping" if EARLY_STOPPING_PATIENCE > 0 else "No early stopping",
                "Embedding layer freezing" if FREEZE_EMBEDDING_LAYERS else "No embedding freezing"
            ]
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
        for i, threshold in enumerate(optimal_thresholds if optimal_thresholds is not None else [0.5] * len(probs)):
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