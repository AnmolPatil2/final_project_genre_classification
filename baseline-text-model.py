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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
        logging.FileHandler("text_model.log"),
        logging.StreamHandler()
    ]
)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set paths
base_dir = script_dir
data_dir = os.path.join(base_dir, "data")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(os.path.join(models_dir, "text"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "text"), exist_ok=True)

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

class MovieTextDataset(Dataset):
    """Dataset for movie overviews"""
    
    def __init__(self, df, text_preprocessor, label_mapping=None):
        self.df = df
        self.preprocessor = text_preprocessor
        
        # Get unique genres
        if label_mapping is None:
            all_genres = sorted(df['primary_genre'].unique())
            self.label_mapping = {genre: i for i, genre in enumerate(all_genres)}
        else:
            self.label_mapping = label_mapping
        
        # Preprocess all texts
        self.sequences = []
        self.labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
            # Preprocess text
            sequence = self.preprocessor.text_to_sequence(row['overview'])
            self.sequences.append(sequence)
            
            # Get label
            genre = row['primary_genre']
            label = self.label_mapping[genre]
            self.labels.append(label)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'text': sequence, 'label': label}

class TextLSTM(nn.Module):
    """LSTM model for text classification"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, 
                 bidirectional=True, dropout=0.5, pad_idx=0):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layer
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # Embed text
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch size, seq len, embedding dim]
        
        # Pass through LSTM
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch size, seq len, hidden dim * num directions]
        # hidden shape: [num layers * num directions, batch size, hidden dim]
        
        # Use hidden state for classification
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
        
        # hidden shape: [batch size, hidden dim * num directions]
        
        # Pass through dropout and fully connected layer
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        # output shape: [batch size, output dim]
        
        return output

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        text = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(text)
        
        # Calculate loss
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(predictions, dim=1)
        correct = (predictions == labels).float().sum()
        accuracy = correct / len(labels)
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataloader"""
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(text)
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(predictions, dim=1)
            correct = (predictions == labels).float().sum()
            accuracy = correct / len(labels)
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            
            # Save predictions and labels for classification report
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), all_predictions, all_labels

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Settings
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    
    # Device
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
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
        logging.error("Run dataset analysis first to create the splits.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Create text preprocessor
    logging.info("Creating text preprocessor...")
    preprocessor = TextPreprocessor(max_vocab_size=10000, max_seq_length=200)
    
    # Build vocabulary
    preprocessor.build_vocab(train_df['overview'])
    
    # Save preprocessor
    preprocessor.save(os.path.join(models_dir, "text", "preprocessor.pkl"))
    
    # Create label mapping
    all_genres = sorted(train_df['primary_genre'].unique())
    label_mapping = {genre: i for i, genre in enumerate(all_genres)}
    n_classes = len(label_mapping)
    
    # Save label mapping
    with open(os.path.join(models_dir, "text", "label_mapping.json"), "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = MovieTextDataset(train_df, preprocessor, label_mapping)
    val_dataset = MovieTextDataset(val_df, preprocessor, label_mapping)
    test_dataset = MovieTextDataset(test_df, preprocessor, label_mapping)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    logging.info("Creating model...")
    model = TextLSTM(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=n_classes,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=preprocessor.word2idx['<PAD>']
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    logging.info("Training model...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0
    
    for epoch in range(N_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_dataloader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Evaluate
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS}")
        logging.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, "text", "best_model.pt"))
            logging.info(f"  New best model saved with val acc: {val_acc:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "text", "training_curves.png"))
    
    # Load best model
    logging.info("Loading best model...")
    model.load_state_dict(torch.load(os.path.join(models_dir, "text", "best_model.pt")))
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_dataloader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    # Generate classification report
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [reverse_label_mapping[i] for i in range(n_classes)]
    
    report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report
    with open(os.path.join(results_dir, "text", "classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "text", "confusion_matrix.png"))
    
    # Save model info
    model_info = {
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'bidirectional': BIDIRECTIONAL,
        'dropout': DROPOUT,
        'vocab_size': preprocessor.vocab_size,
        'n_classes': n_classes,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }
    
    with open(os.path.join(models_dir, "text", "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main()