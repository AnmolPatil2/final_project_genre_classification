import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle

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
        """Load preprocessor from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        preprocessor = cls.__new__(cls)
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.max_seq_length = data['max_seq_length']
        preprocessor.vocab_size = data['vocab_size']
        
        return preprocessor
    
    def text_to_sequence(self, text):
        """Convert text to sequence of word indices"""
        import re
        
        # Tokenize and preprocess text
        if not isinstance(text, str):
            tokens = []
        else:
            # Simple tokenization as fallback
            text = text.lower()
            text = re.sub(r'[^a-z0-9\s]', '', text)
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
    """Load the model and necessary files for inference"""
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
    """Predict movie groups for a given text"""
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
