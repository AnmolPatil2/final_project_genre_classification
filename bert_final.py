import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
from sklearn.utils.class_weight import compute_class_weight
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
data_dir = os.path.join(base_dir, "processed_data")
models_dir = os.path.join(base_dir, "models")
results_dir = os.path.join(base_dir, "results")

# Create directories
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(os.path.join(models_dir, "bert"), exist_ok=True)
os.makedirs(os.path.join(results_dir, "bert"), exist_ok=True)

class MovieFeatureBertDataset(Dataset):
    """Dataset for BERT model with multi-label classification and additional features"""
    
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
        
        # Preprocess all texts and extract additional features
        self.encodings = []
        self.labels = []
        self.numeric_features = []
        
        # Log feature statistics to verify feature extraction
        feature_stats = {
            'vote_average': [],
            'title_sentiment': [],
            'overview_sentiment': [] if 'overview_sentiment' in df.columns else None,
            'title_length': [] if 'title_length' in df.columns else None,
            'overview_length': [] if 'overview_length' in df.columns else None
        }
        
        # For calculating sample weights
        self.label_counts = np.zeros(self.num_labels)
        
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
            
            # Extract and store numeric features
            numeric_features = []
            
            # Extract vote_average
            try:
                vote_average = float(row['vote_average']) if 'vote_average' in row else 0.0
                numeric_features.append(vote_average)
                feature_stats['vote_average'].append(vote_average)
            except (ValueError, TypeError):
                numeric_features.append(0.0)
                feature_stats['vote_average'].append(0.0)
            
            # Extract title_sentiment
            try:
                title_sentiment = float(row['title_sentiment']) if 'title_sentiment' in row else 0.0
                numeric_features.append(title_sentiment)
                feature_stats['title_sentiment'].append(title_sentiment)
            except (ValueError, TypeError):
                numeric_features.append(0.0)
                feature_stats['title_sentiment'].append(0.0)
                
            # Add any other numeric features here
            if 'title_length' in row:
                value = float(row['title_length'])
                numeric_features.append(value)
                feature_stats['title_length'].append(value)
                
            if 'overview_length' in row:
                value = float(row['overview_length'])
                numeric_features.append(value)
                feature_stats['overview_length'].append(value)
                
            if 'overview_sentiment' in row:
                value = float(row['overview_sentiment'])
                numeric_features.append(value)
                feature_stats['overview_sentiment'].append(value)
                
            self.numeric_features.append(torch.tensor(numeric_features, dtype=torch.float))
            
            # Create multi-hot encoded label vector
            label_vector = torch.zeros(self.num_labels)
            try:
                if isinstance(row['groups'], list):
                    for group in row['groups']:
                        if group in self.label_to_idx:
                            idx = self.label_to_idx[group]
                            label_vector[idx] = 1
                            self.label_counts[idx] += 1
            except KeyError:
                logging.warning("Groups not found for a sample. Using empty label vector.")
            
            self.labels.append(label_vector)
        
        # Calculate sample weights for balanced sampling
        self.sample_weights = self._calculate_sample_weights()
            
        # Log feature statistics to verify distribution
        logging.info("Feature statistics:")
        for feature, values in feature_stats.items():
            if values is not None and len(values) > 0:
                non_zero = sum(1 for v in values if abs(v) > 0.001)
                logging.info(f"  {feature}: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}, non-zero={non_zero}/{len(values)} ({non_zero/len(values)*100:.2f}%)")
    
    def _calculate_sample_weights(self):
        """Calculate sample weights for balanced sampling"""
        # Convert multi-hot encoded labels to a list of weights per sample
        weights = []
        
        # Calculate inverse class frequency
        class_weights = np.zeros(self.num_labels)
        for i in range(self.num_labels):
            if self.label_counts[i] > 0:
                class_weights[i] = len(self.df) / self.label_counts[i]
            else:
                class_weights[i] = 1.0
        
        # Normalize to sum to 1
        class_weights = class_weights / np.sum(class_weights)
        
        # For each sample, assign weight based on its labels
        for label_vector in self.labels:
            label_indices = torch.nonzero(label_vector).squeeze().cpu().numpy()
            
            if len(label_indices.shape) == 0:  # Handle case of single label
                label_indices = np.array([label_indices])
                
            if len(label_indices) > 0:
                # Assign weight as average of class weights for all labels
                weight = np.mean([class_weights[idx] for idx in label_indices])
            else:
                weight = 1.0  # Default weight for samples with no labels
            
            weights.append(weight)
            
        return weights
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'numeric_features': self.numeric_features[idx],
            'labels': self.labels[idx]
        }
    
    def get_class_weights(self):
        """Calculate class weights for weighted loss function"""
        # Convert multi-hot encoded labels to class counts
        label_counts = np.sum(np.array([label.numpy() for label in self.labels]), axis=0)
        
        # Calculate class weights (inverse of frequency)
        class_weights = []
        for count in label_counts:
            if count > 0:
                # Use inverse frequency with smoothing to avoid extreme weights
                weight = len(self.df) / (count * self.num_labels)
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

class EnhancedFeatureBertMultiLabelClassifier(nn.Module):
    """
    Enhanced BERT model for multi-label text classification with additional features
    """
    
    def __init__(self, output_dim, num_features, dropout=0.4, freeze_bert_layers=8):
        super().__init__()
        
        # Load pretrained BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT layers to prevent overfitting
        if freeze_bert_layers > 0:
            logging.info(f"Freezing first {freeze_bert_layers} BERT layers")
            modules = [self.bert.embeddings]
            modules.extend(self.bert.encoder.layer[:freeze_bert_layers])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Feature extraction with dropout
        self.dropout = nn.Dropout(dropout)
        
        # BERT hidden size
        hidden_size = self.bert.config.hidden_size
        
        # Process numeric features with more layers and dropouts
        # Simplified architecture with fewer parameters
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size // 4),
        )
        
        # Combine BERT output with numeric features
        self.combination_layer = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 4, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size // 2),
        )
        
        # Classification layer with regularization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, output_dim)
        )
        
        # Add batch normalization for better training stability
        self.bn_bert = nn.BatchNorm1d(hidden_size)
        
        # Add layer-wise feature importance trackers for analysis
        self.feature_importance = None
    
    def forward(self, input_ids, attention_mask, numeric_features):
        # Forward pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and batch normalization to BERT output
        bert_features = self.dropout(pooled_output)
        bert_features = self.bn_bert(bert_features)
        
        # Process numeric features
        processed_features = self.feature_processor(numeric_features)
        
        # Store feature importance during training/evaluation
        if self.training and self.feature_importance is None:
            with torch.no_grad():
                # Get weights from first layer to analyze importance
                first_layer_weights = self.feature_processor[0].weight.detach().cpu()
                importance = torch.sum(torch.abs(first_layer_weights), dim=0)
                self.feature_importance = importance / torch.sum(importance)
        
        # Combine BERT and numeric features
        combined = torch.cat((bert_features, processed_features), dim=1)
        
        # Process combined features
        combined = self.combination_layer(combined)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits

def weighted_bce_loss(outputs, targets, pos_weight):
    """
    Custom weighted BCE loss for handling class imbalance
    """
    loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return loss(outputs, targets)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, class_weights=None, scheduler_type="linear_warmup"):
    """Train model for one epoch with weighted loss"""
    model.train()
    epoch_loss = 0
    
    # Track learning rates through epoch for plotting
    lr_history = []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numeric_features = batch['numeric_features'].to(device)
        labels = batch['labels'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(input_ids, attention_mask, numeric_features)
        
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

def evaluate(model, dataloader, criterion, device, class_thresholds=None, class_weights=None):
    """Evaluate model on dataloader with dynamic class thresholds"""
    model.eval()
    epoch_loss = 0
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Get data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask, numeric_features)
            
            # Calculate loss with class weights if provided
            if isinstance(criterion, FocalLoss):
                loss = criterion(logits, labels)
            elif class_weights is not None:
                pos_weight = class_weights.to(device)
                loss = weighted_bce_loss(logits, labels, pos_weight)
            else:
                loss = criterion(logits, labels)
            
            # Calculate probabilities
            probs = torch.sigmoid(logits)
            
            # Update metrics
            epoch_loss += loss.item()
            
            # Save probabilities and labels for metrics calculation
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # If class thresholds are provided, use them for dynamic thresholding
    if class_thresholds is not None:
        all_predictions = np.zeros_like(all_probs)
        for i in range(all_probs.shape[1]):
            all_predictions[:, i] = (all_probs[:, i] > class_thresholds[i]).astype(float)
    else:
        # Use fixed threshold of 0.3 for all classes
        all_predictions = (all_probs > 0.3).astype(float)
    
    # Calculate different types of F1 scores
    macro_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    return epoch_loss / len(dataloader), macro_f1, micro_f1, weighted_f1, all_probs, all_labels, all_predictions

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
    LEARNING_RATE = 1e-5  # Reduced learning rate
    WEIGHT_DECAY = 0.01
    DROPOUT = 0.4  # Reduced dropout
    N_EPOCHS = 10
    WARMUP_PROPORTION = 0.1
    FREEZE_BERT_LAYERS = 8  # Reduced frozen layers
    
    # Focal Loss parameters
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    
    # Use dynamic thresholding
    USE_DYNAMIC_THRESHOLDS = True
    
    # Use balanced sampling
    USE_BALANCED_SAMPLING = True
    
    # Scheduler type - choose from: "linear_warmup", "cosine_warmup", "reduce_on_plateau"
    SCHEDULER_TYPE = "linear_warmup"  # Changed to linear warmup for more stability
    
    # Early stopping settings
    EARLY_STOPPING_PATIENCE = 2  # Reduced patience
    EARLY_STOPPING_METRIC = "macro_f1"
    
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
        
        # Check for required columns
        required_columns = ['overview', 'groups', 'vote_average', 'title_sentiment']
        for col in required_columns:
            if col not in train_df.columns:
                logging.error(f"'{col}' column not found in dataset.")
                return
        
        # Print some statistics about title_sentiment to verify it's not all zeros
        if 'title_sentiment' in train_df.columns:
            ts_values = train_df['title_sentiment'].values
            non_zero_ts = sum(1 for v in ts_values if abs(v) > 0.001)
            logging.info(f"Title sentiment: {non_zero_ts}/{len(ts_values)} non-zero values ({non_zero_ts/len(ts_values)*100:.2f}%)")
            if non_zero_ts == 0:
                logging.warning("WARNING: All title_sentiment values are zero or near-zero!")
                # If all values are zero, suggest running the sentiment update script
                logging.warning("Please run the sentiment update script first to fix title_sentiment values.")
        
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
        
        # Log basic data statistics for groups
        group_counts = {}
        for groups_list in train_df['groups']:
            for group in groups_list:
                if group in group_counts:
                    group_counts[group] += 1
                else:
                    group_counts[group] = 1
        
        logging.info("Group distribution in training data:")
        for group, count in sorted(group_counts.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {group}: {count} samples ({count/len(train_df)*100:.2f}%)")
        
    except FileNotFoundError as e:
        logging.error(f"Data splits not found: {str(e)}")
        logging.error("Run the data preprocessing script first to create the splits.")
        return
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return
    
    # Load BERT tokenizer
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Save label mapping
    with open(os.path.join(models_dir, "bert", "label_mapping.json"), "w") as f:
        json.dump(label_to_idx, f, indent=2)
    
    # Count number of numeric features in dataset
    # Basic features: vote_average, title_sentiment
    num_features = 2
    
    # Add additional features if available
    additional_features = ['title_length', 'overview_length', 'overview_sentiment']
    for feature in additional_features:
        if feature in train_df.columns:
            num_features += 1
            logging.info(f"Adding {feature} as numeric feature")
    
    logging.info(f"Using {num_features} numeric features in addition to BERT text embeddings")
    
    # Create datasets with multi-label support and additional features
    logging.info("Creating datasets with text and numeric features...")
    train_dataset = MovieFeatureBertDataset(train_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    val_dataset = MovieFeatureBertDataset(val_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    test_dataset = MovieFeatureBertDataset(test_df, tokenizer, max_length=MAX_LENGTH, label_to_idx=label_to_idx)
    
    # Calculate class weights from training data to address class imbalance
    class_weights = train_dataset.get_class_weights()
    logging.info(f"Class weights will be applied during training")
    
    # Create balanced sampler for training loader
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
    
    # Create enhanced model with additional features
    logging.info("Creating enhanced BERT model with numeric features...")
    model = EnhancedFeatureBertMultiLabelClassifier(
        output_dim=n_classes,
        num_features=num_features,
        dropout=DROPOUT,
        freeze_bert_layers=FREEZE_BERT_LAYERS
    )
    
    # Log model architecture
    logging.info(f"Model architecture: {model}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model has {trainable_params:,} trainable parameters out of {total_params:,} total parameters")
    logging.info(f"Percentage frozen: {(1 - trainable_params/total_params) * 100:.2f}%")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer with weight decay
    # Use different learning rates for BERT and classifier layers
    bert_params = {'params': [p for n, p in model.named_parameters() if 'bert' in n and p.requires_grad]}
    classifier_params = {'params': [p for n, p in model.named_parameters() if 'bert' not in n], 'lr': LEARNING_RATE * 3}
    
    optimizer = torch.optim.AdamW([bert_params, classifier_params], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Create learning rate scheduler
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
    if USE_FOCAL_LOSS:
        logging.info(f"Using Focal Loss with gamma={FOCAL_GAMMA}")
        criterion = FocalLoss(gamma=FOCAL_GAMMA)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Initialize class thresholds with default value
    class_thresholds = np.ones(n_classes) * 0.3
    
    # Train model
    logging.info("Training enhanced BERT model with class weighting...")
    logging.info(f"Using dynamic thresholds: {USE_DYNAMIC_THRESHOLDS}")
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    val_macro_f1s = []
    val_micro_f1s = []
    val_weighted_f1s = []
    lr_history = []
    
    best_val_f1 = 0
    no_improve_epochs = 0
    
    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        
        # Train with class weights
        train_loss, epoch_lr_history = train_epoch(
            model, train_dataloader, optimizer, scheduler, criterion, device, 
            class_weights=class_weights, scheduler_type=SCHEDULER_TYPE
        )
        train_losses.append(train_loss)
        lr_history.extend(epoch_lr_history)
        
        # Evaluate with class weights and thresholds
        val_loss, val_macro_f1, val_micro_f1, val_weighted_f1, val_probs, val_labels, val_preds = evaluate(
            model, val_dataloader, criterion, device, 
            class_thresholds=class_thresholds if USE_DYNAMIC_THRESHOLDS else None,
            class_weights=class_weights
        )
        
        val_losses.append(val_loss)
        val_macro_f1s.append(val_macro_f1)
        val_micro_f1s.append(val_micro_f1)
        val_weighted_f1s.append(val_weighted_f1)
        
        # Update dynamic thresholds based on validation results
        if USE_DYNAMIC_THRESHOLDS and epoch > 0:
            class_thresholds = compute_class_thresholds(val_probs, val_labels)
            logging.info(f"Updated class thresholds: min={min(class_thresholds):.2f}, max={max(class_thresholds):.2f}, mean={np.mean(class_thresholds):.2f}")
        
        # Update ReduceLROnPlateau scheduler based on validation loss if using that scheduler
        if SCHEDULER_TYPE == "reduce_on_plateau":
            scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        # Log metrics
        logging.info(f"Epoch {epoch+1}/{N_EPOCHS} - Time: {epoch_time:.2f}s")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Val Loss: {val_loss:.4f}, Val Macro F1: {val_macro_f1:.4f}, Val Micro F1: {val_micro_f1:.4f}")
        logging.info(f"  Current LR: {optimizer.param_groups[0]['lr']:.7f}")
        
        # Select metric for model selection and early stopping
        val_f1 = val_macro_f1  # Use macro F1 as primary metric
        
        # Save best model based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save model and thresholds
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_thresholds': class_thresholds,
                'epoch': epoch,
                'macro_f1': val_macro_f1,
                'micro_f1': val_micro_f1
            }, os.path.join(models_dir, "bert", "best_model.pt"))
            logging.info(f"  New best model saved with val F1: {val_f1:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"  No improvement for {no_improve_epochs} epochs")
        
        # Early stopping check - using F1 score
        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered after {no_improve_epochs} epochs without improvement in {EARLY_STOPPING_METRIC}")
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
    plt.plot(val_macro_f1s, label='Val Macro F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation Macro F1 Score')
    
    # Plot Micro F1 score
    plt.subplot(2, 2, 3)
    plt.plot(val_micro_f1s, label='Val Micro F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('Validation Micro F1 Score')
    
    # Plot learning rate
    plt.subplot(2, 2, 4)
    plt.plot(lr_history)
    plt.xlabel('Batch')
    plt.ylabel('Learning Rate')
    plt.title(f'Learning Rate ({SCHEDULER_TYPE})')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bert", "training_curves.png"))
    
    # Load best model
    logging.info("Loading best model...")
    checkpoint = torch.load(os.path.join(models_dir, "bert", "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    class_thresholds = checkpoint['class_thresholds']
    logging.info(f"Loaded model from epoch {checkpoint['epoch']} with macro F1: {checkpoint['macro_f1']:.4f}")
    
    # Evaluate on test set
    logging.info("Evaluating on test set...")
    test_loss, test_macro_f1, test_micro_f1, test_weighted_f1, test_probs, test_labels, test_preds = evaluate(
        model, test_dataloader, criterion, device,
        class_thresholds=class_thresholds if USE_DYNAMIC_THRESHOLDS else None,
        class_weights=class_weights
    )
    logging.info(f"Test Loss: {test_loss:.4f}, Test Macro F1: {test_macro_f1:.4f}, Test Micro F1: {test_micro_f1:.4f}")
    
    # Generate per-class metrics
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    
    # Calculate F1 score per class
    f1_per_class = f1_score(test_labels, test_preds, average=None, zero_division=0)
    
    class_metrics = {}
    for i, f1 in enumerate(f1_per_class):
        label = idx_to_label[i]
        class_metrics[label] = {
            'f1_score': float(f1),
            'threshold': float(class_thresholds[i]) if USE_DYNAMIC_THRESHOLDS else 0.3
        }
    
    # Save class metrics
    with open(os.path.join(results_dir, "bert", "class_metrics.json"), "w") as f:
        json.dump(class_metrics, f, indent=2)
    
    # Generate detailed analysis of predictions
    analysis_df = analyze_predictions(test_preds, test_labels, idx_to_label)
    
    # Save analysis to CSV
    analysis_df.to_csv(os.path.join(results_dir, "bert", "class_analysis.csv"), index=False)
    
    # Log analysis results
    logging.info("Class-wise analysis:")
    for _, row in analysis_df.iterrows():
        logging.info(f"  {row['Class']}: Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1']:.4f}, Support={row['Support']}")
    
    # Generate confusion matrix per class and save as image
    plt.figure(figsize=(15, 12))
    conf_matrices = multilabel_confusion_matrix(test_labels, test_preds)
    
    # Plot confusion matrices for each class
    rows = int(np.ceil(n_classes / 3))
    for i, (label, matrix) in enumerate(zip([idx_to_label[i] for i in range(n_classes)], conf_matrices)):
        plt.subplot(rows, 3, i+1)
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        plt.title(f'Class: {label} (T={class_thresholds[i]:.2f})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bert", "confusion_matrices.png"))
    
    # Calculate feature importance from model's feature processor
    if hasattr(model, 'feature_importance') and model.feature_importance is not None:
        # Use the feature importance from the model
        importance = model.feature_importance.numpy()
    else:
        # Calculate feature importance (indirectly) by analyzing model weights
        with torch.no_grad():
            # Get weights from the feature processor's first layer
            feature_weights = model.feature_processor[0].weight.cpu().numpy()
            
            # Calculate importance as the sum of absolute weights per feature
            importance = np.sum(np.abs(feature_weights), axis=0)
            
            # Normalize to sum to 100%
            importance = importance / np.sum(importance) * 100
    
    # Create labels for features
    feature_names = ['vote_average', 'title_sentiment']
    for feature in additional_features:
        if feature in train_df.columns:
            feature_names.append(feature)
    
    # Feature importance sorted
    feature_importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    logging.info("Feature importance:")
    for feature, importance_val in sorted_features:
        logging.info(f"  {feature}: {importance_val:.2f}%")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    # Sort features by importance
    sorted_indices = np.argsort(importance)[::-1]  # Reverse to get descending order
    plt.bar([feature_names[i] for i in sorted_indices], importance[sorted_indices])
    plt.xlabel('Feature')
    plt.ylabel('Importance (%)')
    plt.title('Numeric Feature Importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bert", "feature_importance.png"))
    
    # Additional analysis: Count how often each class is predicted
    class_prediction_counts = np.sum(test_preds, axis=0)
    
    # Plot class prediction distribution
    plt.figure(figsize=(12, 6))
    plt.bar([idx_to_label[i] for i in range(n_classes)], class_prediction_counts)
    plt.xlabel('Class')
    plt.ylabel('Number of Predictions')
    plt.title('Class Prediction Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "bert", "class_prediction_distribution.png"))
    
    # Save model info
    model_info = {
        'max_length': MAX_LENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'dropout': DROPOUT,
        'n_epochs': N_EPOCHS,
        'scheduler_type': SCHEDULER_TYPE,
        'use_focal_loss': USE_FOCAL_LOSS,
        'focal_gamma': FOCAL_GAMMA if USE_FOCAL_LOSS else None,
        'use_dynamic_thresholds': USE_DYNAMIC_THRESHOLDS,
        'use_balanced_sampling': USE_BALANCED_SAMPLING,
        'class_thresholds': {idx_to_label[i]: float(class_thresholds[i]) for i in range(n_classes)},
        'freeze_bert_layers': FREEZE_BERT_LAYERS,
        'n_classes': n_classes,
        'num_features': num_features,
        'feature_names': feature_names,
        'feature_importance': {name: float(imp) for name, imp in zip(feature_names, importance)},
        'best_val_macro_f1': float(best_val_f1),
        'test_macro_f1': float(test_macro_f1),
        'test_micro_f1': float(test_micro_f1),
        'test_weighted_f1': float(test_weighted_f1),
        'class_weighting': 'Applied',
        'per_class_f1': {idx_to_label[i]: float(f1_per_class[i]) for i in range(len(f1_per_class))}
    }
    
    with open(os.path.join(models_dir, "bert", "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    logging.info("Enhanced BERT multi-label model training and evaluation complete!")

if __name__ == "__main__":
    main()