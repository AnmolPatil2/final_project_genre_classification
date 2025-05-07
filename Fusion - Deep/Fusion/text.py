import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, f1_score
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("multimodal_classification.log"),
        logging.StreamHandler()
    ]
)

# Get the working directory
script_dir = os.getcwd()

# Set paths
base_dir = script_dir
data_dir = os.path.join(base_dir, "processed_data")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "multimodal"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "multimodal"), exist_ok=True)

# Define the top genres we want to classify
TOP_GENRES = [
    'Drama', 'Comedy', 'Thriller', 'Action', 'Romance',
    'Horror', 'Adventure', 'Crime', 'Science Fiction', 'Family'
]

#======= DATASET CLASS =======

class MultiModalMovieDataset(Dataset):
    """Dataset that combines both text and image data for movies"""
    
    def __init__(self, df, tokenizer, transform=None, max_length=128, label_to_idx=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Set image transform
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        # Get all possible genre labels
        if label_to_idx is None:
            # Extract all unique genres across the dataset
            all_genres = set()
            for genres_list in df['groups']:
                if isinstance(genres_list, list):
                    all_genres.update(genres_list)
            
            all_genres = sorted(list(all_genres))
            self.label_to_idx = {genre: i for i, genre in enumerate(all_genres)}
        else:
            self.label_to_idx = label_to_idx
        
        self.num_labels = len(self.label_to_idx)
        logging.info(f"Dataset initialized with {self.num_labels} possible labels")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process text (overview)
        overview = row['overview'] if 'overview' in row and pd.notna(row['overview']) else ""
        encoding = self.tokenizer(
            overview,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image (poster)
        image_tensor = torch.zeros((3, 224, 224))  # Default empty tensor
        try:
            if 'poster_path' in row and pd.notna(row['poster_path']):
                img_path = row['poster_path']
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    image_tensor = self.transform(img)
        except Exception as e:
            logging.warning(f"Error loading image for idx {idx}: {e}")
        
        # Create multi-hot encoded label vector
        label_vector = torch.zeros(self.num_labels)
        try:
            if isinstance(row['groups'], list):
                for genre in row['groups']:
                    if genre in self.label_to_idx:
                        label_vector[self.label_to_idx[genre]] = 1
        except KeyError:
            logging.warning(f"Groups not found for sample idx {idx}. Using empty label vector.")
        
        # Extract metadata features
        features = []
        for feature in ['vote_average', 'title_sentiment', 'overview_sentiment', 'title_length', 'overview_length']:
            if feature in row and pd.notna(row[feature]):
                features.append(float(row[feature]))
            else:
                features.append(0.0)
        
        metadata = torch.tensor(features, dtype=torch.float)
        
        return {
            'input_ids': encoding['input_ids'][0],
            'attention_mask': encoding['attention_mask'][0],
            'image': image_tensor,
            'metadata': metadata,
            'labels': label_vector
        }

#======= MODEL ARCHITECTURE =======

class MultiModalGenreClassifier(nn.Module):
    """
    A model that combines text, image, and metadata for movie genre classification
    """
    def __init__(self, num_genres, dropout=0.3):
        super().__init__()
        
        # Text encoder (BERT)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_dim = self.bert.config.hidden_size  # 768
        
        # Freeze BERT layers to prevent overfitting
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        # Image encoder (ResNet features)
        self.image_encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        # Remove the final classification layer
        self.image_encoder = nn.Sequential(*list(self.image_encoder.children())[:-1])
        image_dim = 512  # ResNet34 output dimension
        
        # Freeze image encoder layers
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers
        for param in list(self.image_encoder.parameters())[-20:]:
            param.requires_grad = True
        
        # Metadata processing (5 features: vote_average, sentiments, lengths)
        metadata_dim = 5
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64)
        )
        metadata_encoded_dim = 64
        
        # Fusion dimension
        fusion_dim = bert_dim + image_dim + metadata_encoded_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_genres)
        )
    
    def forward(self, input_ids, attention_mask, image, metadata):
        # Text encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.pooler_output  # [batch_size, 768]
        
        # Image encoding
        image_features = self.image_encoder(image)
        image_features = image_features.squeeze(-1).squeeze(-1)  # [batch_size, 512]
        
        # Metadata encoding
        metadata_features = self.metadata_encoder(metadata)  # [batch_size, 64]
        
        # Concatenate features
        combined_features = torch.cat((text_features, image_features, metadata_features), dim=1)
        
        # Classification
        logits = self.classifier(combined_features)
        
        return logits

#======= TRAINING AND EVALUATION FUNCTIONS =======

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask, images, metadata)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model on dataloader"""
    model.eval()
    epoch_loss = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask, images, metadata)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Make predictions
            predictions = (torch.sigmoid(logits) > threshold).float()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='micro')
    
    return epoch_loss / len(dataloader), f1, all_predictions, all_labels

#======= MAIN TRAINING FUNCTION =======

def train_multimodal_classifier():
    """Train and evaluate the multimodal genre classifier"""
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Settings
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.3
    N_EPOCHS = 10
    MAX_TEXT_LENGTH = 128
    THRESHOLD = 0.4
    
    # Select device
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
        
        # Filter rows without posters (if necessary)
        train_df = train_df[train_df['poster_path'].notna()]
        val_df = val_df[val_df['poster_path'].notna()]
        test_df = test_df[test_df['poster_path'].notna()]
        
        logging.info(f"After filtering for posters: {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        
        # Load or create genre mapping from the TOP_GENRES list
        label_to_idx = {genre: i for i, genre in enumerate(TOP_GENRES)}
        idx_to_label = {i: genre for i, genre in enumerate(TOP_GENRES)}
        
        logging.info(f"Using {len(label_to_idx)} genres: {list(label_to_idx.keys())}")
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create image transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = MultiModalMovieDataset(
        train_df, tokenizer, transform=train_transform, 
        max_length=MAX_TEXT_LENGTH, label_to_idx=label_to_idx
    )
    
    val_dataset = MultiModalMovieDataset(
        val_df, tokenizer, transform=val_transform, 
        max_length=MAX_TEXT_LENGTH, label_to_idx=label_to_idx
    )
    
    test_dataset = MultiModalMovieDataset(
        test_df, tokenizer, transform=val_transform, 
        max_length=MAX_TEXT_LENGTH, label_to_idx=label_to_idx
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2 if torch.cuda.is_available() else 0, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, 
        num_workers=2 if torch.cuda.is_available() else 0, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, 
        num_workers=2 if torch.cuda.is_available() else 0, pin_memory=True
    )
    
    # Create model
    logging.info("Creating multimodal classifier model...")
    model = MultiModalGenreClassifier(
        num_genres=len(label_to_idx),
        dropout=DROPOUT
    )
    
    # Move model to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    
    # Create loss function for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Create optimizer with different learning rates for different parts
    param_groups = [
        {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': LEARNING_RATE / 10},
        {'params': [p for n, p in model.image_encoder.named_parameters() if p.requires_grad], 'lr': LEARNING_RATE / 10},
        {'params': [p for n, p in model.metadata_encoder.named_parameters()], 'lr': LEARNING_RATE},
        {'params': [p for n, p in model.classifier.named_parameters()], 'lr': LEARNING_RATE}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    
    # Training loop
    logging.info("Starting training...")
    best_val_f1 = 0.0
    best_model_state = None
    patience = 5
    patience_counter = 0
    
    # Track metrics
    train_losses = []
    val_losses = []
    val_f1s = []
    
    start_time = time.time()
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_f1, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device, THRESHOLD
        )
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Update learning rate based on validation F1
        scheduler.step(val_f1)
        
        # Save best model
        if val_f1 > best_val_f1:
            logging.info(f"  New best model! F1: {val_f1:.4f} (previous: {best_val_f1:.4f})")
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Save best model
    torch.save(model.state_dict(), os.path.join(models_dir, "multimodal", "best_model.pt"))
    
    # Save genre mapping
    with open(os.path.join(models_dir, "multimodal", "genre_mapping.json"), 'w') as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(val_f1s)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    
    plt.subplot(1, 3, 3)
    current_lr = optimizer.param_groups[0]['lr']
    plt.text(0.5, 0.5, f"Final LR: {current_lr:.2e}", ha='center', va='center', fontsize=12)
    plt.title('Learning Rate')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "multimodal", "training_curves.png"))
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device, THRESHOLD
    )
    
    logging.info(f"Test Loss: {test_loss:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    
    # Calculate per-class F1 scores
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    
    # Log per-class F1 scores
    logging.info("Per-class F1 scores:")
    class_metrics = {}
    for i, f1 in enumerate(per_class_f1):
        genre = idx_to_label[i]
        support = np.sum(np.array(test_labels)[:, i])
        logging.info(f"  {genre}: F1={f1:.4f}, Support={support}")
        class_metrics[genre] = {"f1": float(f1), "support": int(support)}
    
    # Save results
    results = {
        "test_f1": float(test_f1),
        "per_class_metrics": class_metrics,
        "train_time": float(total_time),
        "best_val_f1": float(best_val_f1),
        "threshold": float(THRESHOLD)
    }
    
    with open(os.path.join(results_dir, "multimodal", "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot per-class F1 scores
    plt.figure(figsize=(10, 6))
    genres = []
    f1_scores = []
    
    for genre, metrics in sorted(class_metrics.items(), key=lambda x: x[1]['f1'], reverse=True):
        genres.append(genre)
        f1_scores.append(metrics['f1'])
    
    plt.barh(genres, f1_scores)
    plt.xlabel('F1 Score')
    plt.title('F1 Score by Genre')
    plt.xlim(0, 1.0)
    
    # Add values on bars
    for i, v in enumerate(f1_scores):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "multimodal", "per_class_f1.png"))
    
    # Generate confusion matrix for each class
    test_preds_array = np.array(test_preds)
    test_labels_array = np.array(test_labels)
    
    plt.figure(figsize=(15, 10))
    for i, genre in enumerate(TOP_GENRES):
        plt.subplot(3, 4, i+1)
        
        # Calculate confusion matrix values
        tp = np.sum((test_preds_array[:, i] == 1) & (test_labels_array[:, i] == 1))
        fp = np.sum((test_preds_array[:, i] == 1) & (test_labels_array[:, i] == 0))
        fn = np.sum((test_preds_array[:, i] == 0) & (test_labels_array[:, i] == 1))
        tn = np.sum((test_preds_array[:, i] == 0) & (test_labels_array[:, i] == 0))
        
        cm = np.array([[tn, fp], [fn, tp]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        
        plt.title(f'{genre}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "multimodal", "confusion_matrices.png"))
    
    return model, results

if __name__ == "__main__":
    train_multimodal_classifier()
    