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
        logging.FileHandler("image_model.log"),
        logging.StreamHandler()
    ]
)

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("models/image", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/image", exist_ok=True)

class MoviePosterDataset(Dataset):
    """Dataset for movie posters"""
    
    def __init__(self, df, transform=None, label_mapping=None):
        self.df = df
        self.transform = transform
        
        # Get unique genres
        if label_mapping is None:
            all_genres = sorted(df['primary_genre'].unique())
            self.label_mapping = {genre: i for i, genre in enumerate(all_genres)}
        else:
            self.label_mapping = label_mapping
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get poster path
        poster_path = self.df.iloc[idx]['poster_path']
        
        # Load image
        try:
            image = Image.open(poster_path).convert('RGB')
        except Exception as e:
            logging.error(f"Error loading image {poster_path}: {str(e)}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        genre = self.df.iloc[idx]['primary_genre']
        label = self.label_mapping[genre]
        
        return {'image': image, 'label': torch.tensor(label, dtype=torch.long)}

class MoviePosterCNN(nn.Module):
    """CNN model for movie poster classification using transfer learning"""
    
    def __init__(self, output_dim, model_name='resnet18', pretrained=True, freeze_base=True):
        super().__init__()
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
        elif model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=pretrained)
            num_ftrs = self.base_model.classifier[6].in_features
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Replace the last fully connected layer
        if model_name.startswith('resnet'):
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, output_dim)
            )
        elif model_name.startswith('vgg'):
            self.base_model.classifier[6] = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, output_dim)
            )
    
    def forward(self, x):
        return self.base_model(x)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(images)
        
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
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(images)
            
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
    MODEL_NAME = 'resnet18'  # 'resnet18', 'resnet50', or 'vgg16'
    PRETRAINED = True
    FREEZE_BASE = True
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    N_EPOCHS = 10
    
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
    
    # Data transforms
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
    with open("models/image/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)
    
    # Create datasets
    logging.info("Creating datasets...")
    train_dataset = MoviePosterDataset(train_df, transform=train_transform, label_mapping=label_mapping)
    val_dataset = MoviePosterDataset(val_df, transform=val_transform, label_mapping=label_mapping)
    test_dataset = MoviePosterDataset(test_df, transform=val_transform, label_mapping=label_mapping)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Create model
    logging.info(f"Creating {MODEL_NAME} model...")
    model = MoviePosterCNN(
        output_dim=n_classes,
        model_name=MODEL_NAME,
        pretrained=PRETRAINED,
        freeze_base=FREEZE_BASE
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
            torch.save(model.state_dict(), "models/image/best_model.pt")
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
    plt.savefig("results/image/training_curves.png")
    
    # Load best model
    logging.info("Loading best model...")
    model.load_state_dict(torch.load("models/image/best_model.pt"))
    
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
    with open("results/image/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    # Generate confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("results/image/confusion_matrix.png")
    
    # Save model info
    model_info = {
        'model_name': MODEL_NAME,
        'pretrained': PRETRAINED,
        'freeze_base': FREEZE_BASE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'n_epochs': N_EPOCHS,
        'n_classes': n_classes,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc
    }
    
    with open("models/image/model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Generate visual examples of predictions
    logging.info("Generating visual examples...")
    
    # Create directory for visual examples
    os.makedirs("results/image/examples", exist_ok=True)
    
    # Get a batch of test data
    test_dataloader_examples = DataLoader(test_dataset, batch_size=16, shuffle=True)
    examples_batch = next(iter(test_dataloader_examples))
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        examples_images = examples_batch['image'].to(device)
        examples_labels = examples_batch['label'].to(device)
        examples_preds = model(examples_images)
        examples_preds = torch.argmax(examples_preds, dim=1)
    
    # Convert images back to displayable format
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    for i in range(min(16, len(examples_images))):
        # Get image and convert to numpy
        img = inv_normalize(examples_images[i]).cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Get labels
        true_label = class_names[examples_labels[i].item()]
        pred_label = class_names[examples_preds[i].item()]
        
        # Add subplot
        ax = fig.add_subplot(4, 4, i+1)
        ax.imshow(img)
        
        # Set title color based on correctness
        title_color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=title_color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/image/examples/prediction_examples.png")
    
    logging.info("Training and evaluation complete!")

if __name__ == "__main__":
    main()