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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, precision_recall_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Download NLTK resources explicitly with error handling
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
        logging.warning("You might need to manually download NLTK resources")

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

class TextPreprocessor:
    """Text preprocessor for movie overviews"""
    
    def __init__(self, max_vocab_size=10000, max_seq_length=200):
        self.stop_words = set(stopwords.words('english'))
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
            logging.error(f"Tokenization error: {str(e)}")
            # If tokenization fails, return simple whitespace-based tokens as fallback
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
    
    def __init__(self, df, text_preprocessor, group_to_idx=None):
        self.df = df
        self.preprocessor = text_preprocessor
        
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
            sequence = self.preprocessor.text_to_sequence(row['overview'])
            self.sequences.append(sequence)
            
            # Create multi-hot encoded label vector
            label_vector = torch.zeros(self.num_groups)
            
            # Fill in the labels for the groups this movie belongs to
            if 'groups' in row and isinstance(row['groups'], list):
                for group in row['groups']:
                    if group in self.group_to_idx:
                        label_vector[self.group_to_idx[group]] = 1
            
            self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
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
        
    def forward(self, hidden_states, mask=None):
        # hidden_states shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = hidden_states.shape
        
        # Create query, key, and value projections
        Q = self.query(hidden_states)  # [batch_size, seq_len, attention_dim]
        K = self.key(hidden_states)    # [batch_size, seq_len, attention_dim]
        V = self.value(hidden_states)  # [batch_size, seq_len, hidden_dim]
        
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
        
        return output, attention

class LSTMWithAttentionMultiLabel(nn.Module):
    """LSTM model with self-attention for multi-label classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, attention_dim=None, 
                 n_layers=2, bidirectional=True, dropout=0.5, pad_idx=0, freeze_embeddings=False):
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
            dropout=dropout if n_layers > 1 else 0,
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
        
        # Apply self-attention to LSTM outputs
        attended_output, attention_weights = self.attention(lstm_output, mask)
        # attended_output shape: [batch size, seq len, hidden dim * num directions]
        # attention_weights shape: [batch size, seq len, seq len]
        
        # Global max pooling to get the most important features across the sequence
        pooled_output = torch.max(attended_output, dim=1)[0]
        # pooled_output shape: [batch size, hidden dim * num directions]
        
        # Pass through fully connected layers with stronger regularization
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout1(x)
        logits = self.fc2(x)
        # logits shape: [batch size, output dim]
        
        return logits, attention_weights

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
            
            # Store a sample of attention weights (to avoid storing too much data)
            if len(all_attention_weights) < 10:
                all_attention_weights.append(attention_weights[0].cpu().numpy())
    
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

def find_optimal_thresholds(probs, labels):
    """Find optimal thresholds for each group using validation data"""
    n_classes = probs.shape[1]
    thresholds = []
    
    for i in range(n_classes):
        # Calculate precision, recall, thresholds for this group
        precision, recall, threshold_values = precision_recall_curve(labels[:, i], probs[:, i])
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = threshold_values[best_idx] if best_idx < len(threshold_values) else 0.5
        
        thresholds.append(float(best_threshold))
    
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
    
    # Settings
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    ATTENTION_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    FREEZE_EMBEDDINGS = True  # Freeze embeddings to prevent overfitting
    
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
        logging.error(f"Data splits not found: {str(e)}")
        logging.error("Run data preprocessing first to create the splits.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Extract all unique groups across the dataset
    all_groups = set()
    for df in [train_df, val_df, test_df]:
        for groups_list in df['groups']:
            if isinstance(groups_list, list):
                all_groups.update(groups_list)
    
    group_to_idx = {group: i for i, group in enumerate(sorted(list(all_groups)))}
    idx_to_group = {i: group for group, i in group_to_idx.items()}
    
    logging.info(f"Found {len(group_to_idx)} unique groups: {sorted(list(all_groups))}")
    
    # Save group mapping
    with open(os.path.join(models_dir, "lstm_attention", "group_mapping.json"), "w") as f:
        json.dump({
            "group_to_idx": group_to_idx,
            "idx_to_group": idx_to_group
        }, f, indent=2)
    
    # Create text preprocessor
    logging.info("Creating text preprocessor...")
    preprocessor = TextPreprocessor(max_vocab_size=15000, max_seq_length=200)
    
    # Build vocabulary
    preprocessor.build_vocab(train_df['overview'])
    
    # Save preprocessor
    preprocessor.save(os.path.join(models_dir, "lstm_attention", "preprocessor.pkl"))
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = MovieGroupDataset(train_df, preprocessor, group_to_idx)
    val_dataset = MovieGroupDataset(val_df, preprocessor, group_to_idx)
    test_dataset = MovieGroupDataset(test_df, preprocessor, group_to_idx)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4 if device.type != 'cpu' else 0
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=4 if device.type != 'cpu' else 0
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=4 if device.type != 'cpu' else 0
    )
    
    # Create model
    logging.info("Creating LSTM model with self-attention for multi-label classification...")
    model = LSTMWithAttentionMultiLabel(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        attention_dim=ATTENTION_DIM,
        output_dim=len(group_to_idx),
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=preprocessor.word2idx['<PAD>'],
        freeze_embeddings=FREEZE_EMBEDDINGS
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    # Use different learning rates for embedding/LSTM and attention/classifier layers
    params = [
        {'params': model.embedding.parameters(), 'lr': LEARNING_RATE / 10},
        {'params': model.lstm.parameters(), 'lr': LEARNING_RATE / 5},
        {'params': model.attention.parameters()},
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()}
    ]
    optimizer = optim.Adam(params, lr=LEARNING_RATE, weight_decay=1e-5)  # Add weight decay for regularization
    
    # Create loss function for multi-label classification
    # Calculate class weights for balanced learning
    group_counts = np.zeros(len(group_to_idx))
    for groups_list in train_df['groups']:
        if isinstance(groups_list, list):
            for group in groups_list:
                if group in group_to_idx:
                    group_counts[group_to_idx[group]] += 1
    
    # Create pos_weight tensor (inverse frequency)
    total_samples = len(train_df)
    pos_weight = torch.tensor(total_samples / (group_counts + 1), dtype=torch.float).to(device)
    
    # Use BCEWithLogitsLoss with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    logging.info("Using weighted BCEWithLogitsLoss with weights based on inverse frequency")
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',  # Monitor F1 score
        factor=0.5,  # Reduce learning rate by half
        patience=2,  # Wait for 2 epochs without improvement
        verbose=True,
        min_lr=1e-6
    )
    
    # Train model
    logging.info("Training model...")
    train_losses = []
    train_f1s = []
    val_losses = []
    val_f1s = []
    
    best_val_f1 = 0
    early_stopping_counter = 0
    EARLY_STOPPING_PATIENCE = 5
    
    for epoch in range(N_EPOCHS):
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
        
        # Update learning rate scheduler
        scheduler.step(val_f1)
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS}")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(models_dir, "lstm_attention", "best_model.pt"))
            logging.info(f"  New best model saved with val F1: {val_f1:.4f}")
            early_stopping_counter = 0
            
            # Visualize attention weights for the best model
            visualize_attention(
                val_attention,
                val_df['overview'].iloc[:len(val_attention)].tolist(),
                preprocessor,
                save_path=os.path.join(results_dir, "lstm_attention", "attention_visualization.png")
            )
        else:
            # Early stopping logic
            early_stopping_counter += 1
            if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping after {epoch+1} epochs without improvement")
                break
    
    # Plot training curves
    plt.figure(figsize=(12, 6))
    
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
    plt.savefig(os.path.join(results_dir, "lstm_attention", "training_curves.png"))
    logging.info("Training curves saved")
    
    # Load best model
    logging.info("Loading best model...")
    model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_attention", "best_model.pt")))
    
    # Find optimal thresholds on validation set
    logging.info("Finding optimal thresholds for each group...")
    _, _, val_probs, _, val_labels, _ = evaluate(model, val_dataloader, criterion, device)
    optimal_thresholds = find_optimal_thresholds(val_probs, val_labels)
    
    # Log optimal thresholds
    logging.info("Optimal thresholds for each group:")
    for group, threshold in zip(sorted(list(all_groups)), optimal_thresholds):
        logging.info(f"  {group}: {threshold:.4f}")
    
    # Save optimal thresholds
    threshold_dict = {group: float(threshold) for group, threshold in zip(sorted(list(all_groups)), optimal_thresholds)}
    with open(os.path.join(models_dir, "lstm_attention", "optimal_thresholds.json"), "w") as f:
        json.dump(threshold_dict, f, indent=2)
    
    # Evaluate on test set with optimal thresholds
    logging.info("Evaluating on test set with optimal thresholds...")
    test_loss, test_f1, test_probs, test_preds, test_labels, test_attention = evaluate(
        model, test_dataloader, criterion, device, thresholds=optimal_thresholds
    )
    logging.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    
    # Generate per-group metrics
    per_group_metrics = {}
    for i, group in enumerate(sorted(list(all_groups))):
        precision = precision_score(test_labels[:, i], test_preds[:, i])
        recall = recall_score(test_labels[:, i], test_preds[:, i])
        f1 = f1_score(test_labels[:, i], test_preds[:, i])
        
        per_group_metrics[group] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'threshold': float(optimal_thresholds[i])
        }
    
    # Save per-group metrics
    with open(os.path.join(results_dir, "lstm_attention", "group_metrics.json"), "w") as f:
        json.dump(per_group_metrics, f, indent=2)
    
    # Log per-group metrics
    logging.info("Per-group metrics:")
    logging.info(f"{'Group':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Threshold':>10}")
    logging.info("-" * 60)
    for group, metrics in per_group_metrics.items():
        logging.info(f"{group:<15} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['threshold']:>10.4f}")
    
    # Visualize attention weights for test set
    visualize_attention(
        test_attention,
        test_df['overview'].iloc[:len(test_attention)].tolist(),
        preprocessor,
        save_path=os.path.join(results_dir, "lstm_attention", "test_attention_visualization.png")
    )
    
    # Plot group-wise F1 scores
    plt.figure(figsize=(10, 8))
    groups = list(per_group_metrics.keys())
    f1_scores = [metrics['f1'] for metrics in per_group_metrics.values()]
    
    # Sort groups by F1 score
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_groups = [groups[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    plt.barh(sorted_groups, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1 Score')
    plt.title('F1 Score by Group')
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "lstm_attention", "group_f1_scores.png"))
    logging.info("Group F1 score chart saved")
    
    # Create and save model info
    model_info = {
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "attention_dim": ATTENTION_DIM,
        "n_layers": N_LAYERS,
        "bidirectional": BIDIRECTIONAL,
        "dropout": DROPOUT,
        "vocab_size": preprocessor.vocab_size,
        "num_groups": len(group_to_idx),
        "freeze_embeddings": FREEZE_EMBEDDINGS,
        "best_val_f1": float(best_val_f1),
        "test_f1": float(test_f1)
    }
    
    with open(os.path.join(models_dir, "lstm_attention", "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Save a prediction example function for later use
    def predict_groups_from_text(text, threshold_dict=None):
        """Predict groups for a new movie overview text"""
        if threshold_dict is None:
            # Load thresholds if not provided
            with open(os.path.join(models_dir, "lstm_attention", "optimal_thresholds.json"), "r") as f:
                threshold_dict = json.load(f)
        
        # Load preprocessor if not already loaded
        try:
            preprocessor
        except NameError:
            preprocessor = TextPreprocessor.load(os.path.join(models_dir, "lstm_attention", "preprocessor.pkl"))
        
        # Load model if not already loaded
        try:
            model
        except NameError:
            # Load group mapping
            with open(os.path.join(models_dir, "lstm_attention", "group_mapping.json"), "r") as f:
                mapping_data = json.load(f)
                group_to_idx = mapping_data["group_to_idx"]
            
            # Load model info for parameters
            with open(os.path.join(models_dir, "lstm_attention", "model_info.json"), "r") as f:
                model_info = json.load(f)
            
            # Create model with same parameters
            model = LSTMWithAttentionMultiLabel(
                vocab_size=model_info["vocab_size"],
                embedding_dim=model_info["embedding_dim"],
                hidden_dim=model_info["hidden_dim"],
                attention_dim=model_info["attention_dim"],
                output_dim=model_info["num_groups"],
                n_layers=model_info["n_layers"],
                bidirectional=model_info["bidirectional"],
                dropout=model_info["dropout"],
                pad_idx=preprocessor.word2idx['<PAD>'],
                freeze_embeddings=model_info["freeze_embeddings"]
            )
            
            # Load model weights
            model.load_state_dict(torch.load(os.path.join(models_dir, "lstm_attention", "best_model.pt")))
            model.eval()
        
        # Process text
        sequence = preprocessor.text_to_sequence(text)
        sequence_tensor = torch.tensor([sequence], dtype=torch.long)
        
        # Make prediction
        with torch.no_grad():
            logits, attention_weights = model(sequence_tensor)
            probabilities = torch.sigmoid(logits).numpy()[0]
        
        # Apply thresholds
        predictions = []
        for i, (group, threshold) in enumerate(threshold_dict.items()):
            if probabilities[i] > threshold:
                predictions.append(group)
        
        return {
            "predicted_groups": predictions,
            "probabilities": {group: float(probabilities[i]) for i, group in enumerate(threshold_dict.keys())},
            "attention_weights": attention_weights.numpy()[0]
        }
    
    # Save example function
    with open(os.path.join(models_dir, "lstm_attention", "prediction_function.py"), "w") as f:
        f.write("""
import os
import torch
import json
import numpy as np
from text_preprocessor import TextPreprocessor
from model import LSTMWithAttentionMultiLabel

def predict_groups_from_text(text, model_dir="lstm_attention"):
    \"\"\"Predict groups for a new movie overview text\"\"\"
    # Load thresholds
    with open(os.path.join(model_dir, "optimal_thresholds.json"), "r") as f:
        threshold_dict = json.load(f)
    
    # Load preprocessor
    preprocessor = TextPreprocessor.load(os.path.join(model_dir, "preprocessor.pkl"))
    
    # Load group mapping
    with open(os.path.join(model_dir, "group_mapping.json"), "r") as f:
        mapping_data = json.load(f)
        group_to_idx = mapping_data["group_to_idx"]
    
    # Load model info for parameters
    with open(os.path.join(model_dir, "model_info.json"), "r") as f:
        model_info = json.load(f)
    
    # Create model with same parameters
    model = LSTMWithAttentionMultiLabel(
        vocab_size=model_info["vocab_size"],
        embedding_dim=model_info["embedding_dim"],
        hidden_dim=model_info["hidden_dim"],
        attention_dim=model_info["attention_dim"],
        output_dim=model_info["num_groups"],
        n_layers=model_info["n_layers"],
        bidirectional=model_info["bidirectional"],
        dropout=model_info["dropout"],
        pad_idx=preprocessor.word2idx['<PAD>'],
        freeze_embeddings=model_info["freeze_embeddings"]
    )
    
    # Load model weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_model.pt")))
    model.eval()
    
    # Process text
    sequence = preprocessor.text_to_sequence(text)
    sequence_tensor = torch.tensor([sequence], dtype=torch.long)
    
    # Make prediction
    with torch.no_grad():
        logits, attention_weights = model(sequence_tensor)
        probabilities = torch.sigmoid(logits).numpy()[0]
    
    # Apply thresholds
    predictions = []
    for i, (group, threshold) in enumerate(threshold_dict.items()):
        if probabilities[i] > threshold:
            predictions.append(group)
    
    return {
        "predicted_groups": predictions,
        "probabilities": {group: float(probabilities[i]) for i, group in enumerate(threshold_dict.keys())},
        "attention_weights": attention_weights.numpy()[0]
    }
""")
    
    logging.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()