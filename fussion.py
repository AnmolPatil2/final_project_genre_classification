import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import json
import pickle
import logging
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_model.log"),
        logging.StreamHandler()
    ]
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("models/multimodal", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/multimodal", exist_ok=True)

class MultiModalDataset(Dataset):
    """Dataset for both movie posters and overviews"""
    
    def __init__(self, df, text_preprocessor, image_transform=None, label_mapping=None):
        self.df = df
        self.text_preprocessor = text_preprocessor
        self.image_transform = image_transform
        
        # Get unique genres
        if label_mapping is None:
            all_genres = sorted(df['primary_genre'].unique())
            self.label_mapping = {genre: i for i, genre in enumerate(all_genres)}
        else:
            self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get poster path and overview
        poster_path = self.df.iloc[idx]['poster_path']
        overview = self.df.iloc[idx]['overview']
        
        # Load image
        try:
            image = Image.open(poster_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {poster_path}: {str(e)}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply image transformations
        if self.image_transform:
            image = self.image_transform(image)
        
        # Preprocess text
        text = torch.tensor(self.text_preprocessor.text_to_sequence(overview), dtype=torch.long)
        
        # Get label
        genre = self.df.iloc[idx]['primary_genre']
        label = torch.tensor(self.label_mapping[genre], dtype=torch.long)
        
        return {'image': image, 'text': text, 'label': label}

class TextEncoder(nn.Module):
    """LSTM encoder for text data"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, 
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
        
        # Output dimension
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
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
        
        # Use hidden state for representation
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1]
        
        # hidden shape: [batch size, hidden dim * num directions]
        
        # Apply dropout
        hidden = self.dropout(hidden)
        
        return hidden

class ImageEncoder(nn.Module):
    """CNN encoder for image data"""
    
    def __init__(self, model_name='resnet18', pretrained=True, freeze_base=True):
        super().__init__()
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
            # Remove the final fully connected layer
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
            # Remove the final fully connected layer
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        elif model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=pretrained)
            num_ftrs = self.base_model.classifier[6].in_features
            # Use the features part only
            self.base_model = self.base_model.features
            # Add adaptive pooling
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Set output dimension
        self.output_dim = num_ftrs
        self.model_name = model_name
    
    def forward(self, x):
        # Forward pass through base model
        features = self.base_model(x)
        
        # Handle different model architectures
        if self.model_name.startswith('vgg'):
            features = self.adaptive_pool(features)
        
        # Flatten features
        features = features.view(features.size(0), -1)
        
        return features

class FusionModel(nn.Module):
    """Fusion model for combining text and image features"""
    
    def __init__(self, text_encoder, image_encoder, output_dim, fusion_method='concat', dropout=0.5):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion_method = fusion_method
        
        # Get input dimension based on fusion method
        if fusion_method == 'concat':
            fusion_dim = text_encoder.output_dim + image_encoder.output_dim
        elif fusion_method == 'sum':
            assert text_encoder.output_dim == image_encoder.output_dim, "Output dimensions must match for sum fusion"
            fusion_dim = text_encoder.output_dim
        elif fusion_method == 'attention':
            # Attention-based fusion
            self.text_attention = nn.Linear(text_encoder.output_dim, 1)
            self.image_attention = nn.Linear(image_encoder.output_dim, 1)
            fusion_dim = text_encoder.output_dim + image_encoder.output_dim
        else:
            raise ValueError(f"Unsupported fusion method: {fusion_method}")
        
        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, text, image):
        # Encode text and image
        text_features = self.text_encoder(text)
        image_features = self.image_encoder(image)
        
        # Fusion
        if self.fusion_method == 'concat':
            # Simple concatenation
            fused_features = torch.cat((text_features, image_features), dim=1)
        elif self.fusion_method == 'sum':
            # Element-wise sum
            fused_features = text_features + image_features
        elif self.fusion_method == 'attention':
            # Attention-based fusion
            text_attention = torch.sigmoid(self.text_attention(text_features))
            image_attention = torch.sigmoid(self.image_attention(image_features))
            
            # Normalize attention weights
            attention_sum = text_attention + image_attention
            text_attention = text_attention / attention_sum
            image_attention = image_attention / attention_sum
            
            # Apply attention
            text_features = text_features * text_attention
            image_features = image_features * image_attention
            
            # Concatenate attended features
            fused_features = torch.cat((text_features, image_features), dim=1)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(texts, images)
        
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
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(texts, images)
            
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

def load_text_preprocessor(path):
    """Load text preprocessor from file"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        from baseline_text_model import TextPreprocessor
        preprocessor = TextPreprocessor(max_seq_length=data['max_seq_length'])
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        preprocessor.vocab_size = data['vocab_size']
        
        logging.info(f"Text preprocessor loaded from {path}")
        return preprocessor
    except Exception as e:
        logging.error(f"Error loading text preprocessor: {str(e)}")
        return None

def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Settings
    TEXT_EMBEDDING_DIM = 300
    TEXT_HIDDEN_DIM = 256
    TEXT_N_LAYERS = 2
    TEXT_BIDIRECTIONAL = True
    
    IMAGE_MODEL_NAME = 'resnet18'
    IMAGE_PRETRAINED = True
    IMAGE_FREEZE_BASE = True
    
    FUSION_METHOD = 'concat'  # 'concat', 'sum', or 'attention'
    DROPOUT = 0.5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 15
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    logging.info("Loading data...")
    try:
        train_df = pd.read_pickle("data/splits/train.pkl")
        val_df = pd.read_pickle("data/splits/val.pkl")
        test_df = pd.read_pickle("data/splits/test.pkl")
    except FileNotFoundError:
        logging.error("Data splits not found. Run dataset analysis first.")
        return
    
    # Load text preprocessor
    text_preprocessor = load_text_preprocessor("models/text/preprocessor.pkl")
    if text_preprocessor is None:
        logging.error("No text preprocessor found. Run text model training first.")
        return
    
    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create label mapping
    all_genres = sorted(train_df['primary_genre'].unique())
    label_mapping = {genre: i for i, genre in enumerate(all_genres)}
    n_classes = len(label_mapping)
    
    # Save label mapping
    with open("models/multimodal/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = MultiModalDataset(
        train_df, text_preprocessor, image_transform=train_transform, label_mapping=label_mapping
    )
    val_dataset = MultiModalDataset(
        val_df, text_preprocessor, image_transform=val_transform, label_mapping=label_mapping
    )
    test_dataset = MultiModalDataset(
        test_df, text_preprocessor, image_transform=val_transform, label_mapping=label_mapping
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Create encoders
    logging.info("Creating encoders...")
    text_encoder = TextEncoder(
        vocab_size=text_preprocessor.vocab_size,
        embedding_dim=TEXT_EMBEDDING_DIM,
        hidden_dim=TEXT_HIDDEN_DIM,
        n_layers=TEXT_N_LAYERS,
        bidirectional=TEXT_BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=text_preprocessor.word2idx['<PAD>']
    )
    
    image_encoder = ImageEncoder(
        model_name=IMAGE_MODEL_NAME,
        pretrained=IMAGE_PRETRAINED,
        freeze_base=IMAGE_FREEZE_BASE
    )
    
    # Create fusion model
    logging.info(f"Creating fusion model with method: {FUSION_METHOD}...")
    model = FusionModel(
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        output_dim=n_classes,
        fusion_method=FUSION_METHOD,
        dropout=DROPOUT
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
            torch.save(model.state_dict(), "models/multimodal/best_model.pt")
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
    plt.savefig("results/multimodal/training_curves.png")
    
    # Load best model
    logging.info("Loading best model...")
    model.load_state_dict(torch.load("models/multimodal/best_model.pt"))
    
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
    with open("results/multimodal/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/multimodal/confusion_matrix.png")
    
    # Save model info
    model_info = {
        'text_embedding_dim': TEXT_EMBEDDING_DIM,
        'text_hidden_dim': TEXT_HIDDEN_DIM,
        'text_n_layers': TEXT_N_LAYERS,
        'text_bidirectional': TEXT_BIDIRECTIONAL,
        'image_model_name': IMAGE_MODEL_NAME,
        'image_pretrained': IMAGE_PRETRAINED,
        'image_freeze_base': IMAGE_FREEZE_BASE,
        'fusion_method': FUSION_METHOD,
        'dropout': DROPOUT,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'n_classes': n_classes,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }
    
    with open("models/multimodal/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Compare with individual modalities
    logging.info("Comparing with individual modalities...")
    
    # Load text and image model results if available
    text_report_path = "results/text/classification_report.json"
    image_report_path = "results/image/classification_report.json"
    
    text_acc = None
    image_acc = None
    
    if os.path.exists(text_report_path):
        with open(text_report_path, 'r') as f:
            text_report = json.load(f)
            text_acc = text_report['accuracy']
            logging.info(f"Text model accuracy: {text_acc:.4f}")
    
    if os.path.exists(image_report_path):
        with open(image_report_path, 'r') as f:
            image_report = json.load(f)
            image_acc = image_report['accuracy']
            logging.info(f"Image model accuracy: {image_acc:.4f}")
    
    if text_acc is not None and image_acc is not None:
        logging.info(f"Multimodal model accuracy: {test_acc:.4f}")
        text_improvement = ((test_acc - text_acc) / text_acc) * 100
        image_improvement = ((test_acc - image_acc) / image_acc) * 100
        logging.info(f"Improvement over text model: {text_improvement:.2f}%")
        logging.info(f"Improvement over image model: {image_improvement:.2f}%")
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        models = ['Text', 'Image', 'Multimodal']
        accuracies = [text_acc, image_acc, test_acc]
        
        plt.bar(models, accuracies, color=['blue', 'green', 'red'])
        plt.ylabel('Accuracy')
        plt.title('Model Comparison')
        plt.ylim(0, 1.0)
        
        # Add value labels
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig("results/multimodal/model_comparison.png")
    
    logging.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()