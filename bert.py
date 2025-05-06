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
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bert_model.log"),
        logging.StreamHandler()
    ]
)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set paths
base_dir = script_dir
data_dir = os.path.join(base_dir, "processed_data")  # Changed to match our processor output
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "bert"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "bert"), exist_ok=True)

class MovieBertMultiLabelDataset(Dataset):
    """Dataset for BERT model with multi-label classification"""
    
    def __init__(self, df, tokenizer, max_length=200, label_to_idx=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Get all possible group labels
        if label_to_idx is None:
            # Extract all unique groups across the dataset
            all_groups = set()
            for groups_list in df['groups']:
                if isinstance(groups_list, list):
                    all_groups.update(groups_list)
            
            all_groups = sorted(list(all_groups))
            self.label_to_idx = {group: i for i, group in enumerate(all_groups)}
        else:
            self.label_to_idx = label_to_idx
        
        self.num_labels = len(self.label_to_idx)
        logging.info(f"Dataset initialized with {self.num_labels} possible labels")
        
        # Preprocess all texts
        self.encodings = []
        self.labels = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparing BERT dataset"):
            # Get the overview text
            text = row['overview'] if 'overview' in row else ""
            
            # Make sure text is a string
            if not isinstance(text, str):
                text = "" if pd.isna(text) else str(text)
            
            # Log a sample to verify we're getting the right data
            if len(self.encodings) == 0:
                logging.info(f"Sample overview text: {text[:100]}...")
            
            # Tokenize and encode
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Store as individual tensors
            self.encodings.append({
                'input_ids': encoding['input_ids'][0],
                'attention_mask': encoding['attention_mask'][0]
            })
            
            # Create multi-hot encoded label vector
            label_vector = torch.zeros(self.num_labels)
            try:
                if isinstance(row['groups'], list):
                    for group in row['groups']:
                        if group in self.label_to_idx:
                            label_vector[self.label_to_idx[group]] = 1
            except KeyError:
                logging.warning("Groups not found for a sample. Using empty label vector.")
            
            self.labels.append(label_vector)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'labels': self.labels[idx]
        }

class BertMultiLabelClassifier(nn.Module):
    """BERT model for multi-label text classification"""
    
    def __init__(self, output_dim, dropout=0.3, freeze_bert_layers=0):
        super().__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze some BERT layers to prevent overfitting
        if freeze_bert_layers > 0:
            logging.info(f"Freezing first {freeze_bert_layers} BERT layers")
            modules = [self.bert.embeddings]
            modules.extend(self.bert.encoder.layer[:freeze_bert_layers])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Feature extraction with dropout
        self.dropout = nn.Dropout(dropout)
        
        # Add a more complex classifier to capture multi-label relationships
        hidden_size = self.bert.config.hidden_size
        
        # Two-layer classifier with higher dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout + 0.1),  # Higher dropout for hidden layer
            nn.Linear(hidden_size // 2, output_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        x = self.dropout(pooled_output)
        
        # Classification layers
        logits = self.classifier(x)
        
        return logits

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scheduler_type="linear_warmup"):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    
    # Track learning rates through epoch for plotting
    lr_history = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Calculate loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss, lr_history

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
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Convert logits to predictions using threshold
            predictions = (torch.sigmoid(logits) > threshold).float()
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Save predictions and labels for classification report
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate F1 score (macro)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    return epoch_loss / len(dataloader), f1, all_predictions, all_labels

def main():
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Settings
    MAX_LENGTH = 200  # Maximum sequence length for BERT
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.4  # Increased dropout for regularization
    N_EPOCHS = 10
    WARMUP_PROPORTION = 0.1
    THRESHOLD = 0.5  # Threshold for converting logits to binary predictions
    FREEZE_BERT_LAYERS = 6  # Freeze lower BERT layers to prevent overfitting
    
    # Scheduler type - choose from: "linear_warmup", "cosine_warmup", "reduce_on_plateau"
    SCHEDULER_TYPE = "cosine_warmup"
    
    # Device - use MPS if available (for Apple Silicon Macs)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    
    # Load data from our processed data directory
    logging.info("Loading data...")
    try:
        # Load from processed_data/splits directory
        train_df = pd.read_pickle(os.path.join(data_dir, "splits", "train.pkl"))
        val_df = pd.read_pickle(os.path.join(data_dir, "splits", "val.pkl"))
        test_df = pd.read_pickle(os.path.join(data_dir, "splits", "test.pkl"))
        logging.info(f"Loaded {len(train_df)} training, {len(val_df)} validation, and {len(test_df)} test samples")
        
        # Log data columns to verify structure
        logging.info(f"Data columns: {train_df.columns.tolist()}")
        
        # Verify that 'overview' column exists and check its content
        if 'overview' in train_df.columns:
            sample_overview = train_df['overview'].iloc[0] if not train_df.empty else None
            logging.info(f"Sample overview: {type(sample_overview)} - {sample_overview[:100] if isinstance(sample_overview, str) else None}")
        else:
            logging.error("'overview' column not found in dataset.")
            return
        
        # Check if 'groups' column exists and its format
        if 'groups' in train_df.columns:
            sample = train_df['groups'].iloc[0] if not train_df.empty else None
            logging.info(f"Sample groups format: {type(sample)} - {sample}")
        else:
            logging.error("'groups' column not found in dataset.")
            return
        
        # Find all unique groups across the dataset for multi-label classification
        all_groups = set()
        for df in [train_df, val_df, test_df]:
            for groups_list in df['groups']:
                if isinstance(groups_list, list):
                    all_groups.update(groups_list)
        
        all_groups = sorted(list(all_groups))
        label_to_idx = {group: i for i, group in enumerate(all_groups)}
        n_classes = len(label_to_idx)
        
        logging.info(f"Found {n_classes} unique groups: {all_groups}")
        
    except FileNotFoundError as e:
        logging.error(f"Data splits not found: {str(e)}")
        logging.error("Run the data preprocessing script first to create the splits.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return
    
    # Load BERT tokenizer
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Save label mapping
    with open(os.path.join(models_dir, "bert", "label_mapping.json"), "w") as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Create datasets with multi-label support
    logging.info("Creating BERT datasets for multi-label classification...")
    train_dataset = MovieBertMultiLabelDataset(train_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    val_dataset = MovieBertMultiLabelDataset(val_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    test_dataset = MovieBertMultiLabelDataset(test_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model for multi-label classification
    logging.info("Creating BERT model for multi-label classification...")
    model = BertMultiLabelClassifier(
        output_dim=n_classes,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer with weight decay
    # Use different learning rates for BERT and classifier layers
    bert_params = {'params': [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]}
    classifier_params = {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': LEARNING_RATE * 5}
    
    optimizer = torch.optim.AdamW([bert_params, classifier_params], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
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
    
    # Create criterion (loss function) for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    
    # Train model
    logging.info("Training BERT model...")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    val_f1s = []
    lr_history = []
    
    best_val_f1 = 0
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, epoch_lr_history = train_epoch(model, train_dataloader, optimizer, scheduler, criterion, device, SCHEDULER_TYPE)
        train_losses.append(train_loss)
        lr_history.extend(epoch_lr_history)
        
        # Evaluate
        val_loss, val_f1, val_preds, val_labels = evaluate(model, val_dataloader, criterion, device, THRESHOLD)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)
        
        # Update ReduceLROnPlateau scheduler based on validation loss if using that scheduler
        if SCHEDULER_TYPE == "reduce_on_plateau":
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
        logging.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(models_dir, "bert", "best_model.pt"))
            logging.info(f"  New best model saved with val F1: {val_f1:.4f}")
    
    total_time = time.time() - start_time
    logging.info(f"Training completed in {total_time:.2f}s")
    
    # Plot training curves
    plt.figure(figsize=(18, 6))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot F1 score
    plt.subplot(1, 3, 2)
    plt.plot(val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation F1 Score')
    
    # Plot learning rate
    plt.subplot(1, 3, 3)
    plt.plot(lr_history)
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate ({SCHEDULER_TYPE})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bert", "training_curves.png"))
    
    # Load best model
    logging.info("Loading best model...")
    model.load_state_dict(torch.load(os.path.join(models_dir, "bert", "best_model.pt")))
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_f1, test_preds, test_labels = evaluate(model, test_dataloader, criterion, device, THRESHOLD)
    logging.info(f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")
    
    # Generate per-class metrics
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    
    # Calculate F1 score per class
    f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    
    class_metrics = {}
    for i, f1 in enumerate(f1_per_class):
        label = idx_to_label[i]
        class_metrics[label] = {'f1_score': float(f1)}
    
    # Save class metrics
    with open(os.path.join(results_dir, "bert", "class_metrics.json"), "w") as f:
        json.dump(class_metrics, f, indent=2)
    
    # Save model info
    model_info = {
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'n_epochs': N_EPOCHS,
        'scheduler_type': SCHEDULER_TYPE,
        'warmup_proportion': WARMUP_PROPORTION,
        'threshold': THRESHOLD,
        'freeze_bert_layers': FREEZE_BERT_LAYERS,
        'n_classes': n_classes,
        'best_val_f1': float(best_val_f1),
        'test_f1': float(test_f1)
    }
    
    with open(os.path.join(models_dir, "bert", "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logging.info("BERT multi-label model training and evaluation complete!")

if __name__ == "__main__":
    main()