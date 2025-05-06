import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import argparse

# Import local modules
from baseline_text_model import TextLSTM, TextPreprocessor, MovieTextDataset
from baseline_image_model import MoviePosterCNN, MoviePosterDataset
from multimodal_fusion_model import (
    TextEncoder, ImageEncoder, FusionModel, MultiModalDataset, load_text_preprocessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_comparison.log"),
        logging.StreamHandler()
    ]
)

# Create directories
os.makedirs("results/comparison", exist_ok=True)

class ModelEvaluator:
    """Evaluator for comparing text, image, and multimodal models"""
    
    def __init__(self, device=None):
        """
        Initialize the evaluator
        
        Args:
            device: Device to run evaluation on (default: auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logging.info(f"Using device: {self.device}")
        
        # Load text preprocessor
        self.text_preprocessor = load_text_preprocessor("models/text/preprocessor.pkl")
        if self.text_preprocessor is None:
            raise ValueError("Failed to load text preprocessor")
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load models
        self.text_model = self._load_text_model()
        self.image_model = self._load_image_model()
        self.multimodal_model = self._load_multimodal_model()
        
        # Load label mappings
        with open("models/text/label_mapping.json", 'r') as f:
            self.text_label_mapping = json.load(f)
        
        with open("models/image/label_mapping.json", 'r') as f:
            self.image_label_mapping = json.load(f)
        
        with open("models/multimodal/label_mapping.json", 'r') as f:
            self.multimodal_label_mapping = json.load(f)
        
        # Create reverse mappings
        self.text_idx_to_genre = {int(v): k for k, v in self.text_label_mapping.items()}
        self.image_idx_to_genre = {int(v): k for k, v in self.image_label_mapping.items()}
        self.multimodal_idx_to_genre = {int(v): k for k, v in self.multimodal_label_mapping.items()}
    
    def _load_text_model(self):
        """Load text model"""
        try:
            # Load model info
            with open("models/text/model_info.json", 'r') as f:
                model_info = json.load(f)
            
            # Create model
            model = TextLSTM(
                vocab_size=self.text_preprocessor.vocab_size,
                embedding_dim=model_info['embedding_dim'],
                hidden_dim=model_info['hidden_dim'],
                output_dim=model_info['n_classes'],
                n_layers=model_info['n_layers'],
                bidirectional=model_info['bidirectional'],
                dropout=model_info['dropout'],
                pad_idx=self.text_preprocessor.word2idx.get('<PAD>', 0)
            )
            
            # Load weights
            model.load_state_dict(torch.load("models/text/best_model.pt", map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            logging.info("Text model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading text model: {str(e)}")
            return None
    
    def _load_image_model(self):
        """Load image model"""
        try:
            # Load model info
            with open("models/image/model_info.json", 'r') as f:
                model_info = json.load(f)
            
            # Create model
            model = MoviePosterCNN(
                output_dim=model_info['n_classes'],
                model_name=model_info['model_name'],
                pretrained=False,  # No need for pretrained weights
                freeze_base=model_info['freeze_base']
            )
            
            # Load weights
            model.load_state_dict(torch.load("models/image/best_model.pt", map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            logging.info("Image model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading image model: {str(e)}")
            return None
    
    def _load_multimodal_model(self):
        """Load multimodal model"""
        try:
            # Load model info
            with open("models/multimodal/model_info.json", 'r') as f:
                model_info = json.load(f)
            
            # Create text encoder
            text_encoder = TextEncoder(
                vocab_size=self.text_preprocessor.vocab_size,
                embedding_dim=model_info['text_embedding_dim'],
                hidden_dim=model_info['text_hidden_dim'],
                n_layers=model_info['text_n_layers'],
                bidirectional=model_info['text_bidirectional'],
                dropout=model_info['dropout'],
                pad_idx=self.text_preprocessor.word2idx.get('<PAD>', 0)
            )
            
            # Create image encoder
            image_encoder = ImageEncoder(
                model_name=model_info['image_model_name'],
                pretrained=False,
                freeze_base=model_info['image_freeze_base']
            )
            
            # Create fusion model
            model = FusionModel(
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                output_dim=model_info['n_classes'],
                fusion_method=model_info['fusion_method'],
                dropout=model_info['dropout']
            )
            
            # Load weights
            model.load_state_dict(torch.load("models/multimodal/best_model.pt", map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            logging.info("Multimodal model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Error loading multimodal model: {str(e)}")
            return None
    
    def evaluate_text_model(self, dataloader):
        """Evaluate text model on dataloader"""
        if self.text_model is None:
            logging.error("Text model not loaded")
            return None
        
        criterion = nn.CrossEntropyLoss()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating text model"):
                # Get data
                texts = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.text_model(texts)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, predictions = torch.max(outputs, dim=1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, all_predictions, 
            target_names=[self.text_idx_to_genre[i] for i in range(len(self.text_idx_to_genre))],
            output_dict=True
        )
        
        logging.info(f"Text model evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return results
    
    def evaluate_image_model(self, dataloader):
        """Evaluate image model on dataloader"""
        if self.image_model is None:
            logging.error("Image model not loaded")
            return None
        
        criterion = nn.CrossEntropyLoss()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating image model"):
                # Get data
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.image_model(images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, predictions = torch.max(outputs, dim=1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, all_predictions, 
            target_names=[self.image_idx_to_genre[i] for i in range(len(self.image_idx_to_genre))],
            output_dict=True
        )
        
        logging.info(f"Image model evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return results
    
    def evaluate_multimodal_model(self, dataloader):
        """Evaluate multimodal model on dataloader"""
        if self.multimodal_model is None:
            logging.error("Multimodal model not loaded")
            return None
        
        criterion = nn.CrossEntropyLoss()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating multimodal model"):
                # Get data
                texts = batch['text'].to(self.device)
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.multimodal_model(texts, images)
                
                # Calculate loss
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                # Get predictions
                _, predictions = torch.max(outputs, dim=1)
                
                # Store predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(
            all_labels, all_predictions, 
            target_names=[self.multimodal_idx_to_genre[i] for i in range(len(self.multimodal_idx_to_genre))],
            output_dict=True
        )
        
        logging.info(f"Multimodal model evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'labels': all_labels
        }
        
        return results
    
    def evaluate_all_models(self, test_df, batch_size=32):
        """Evaluate all models on test data"""
        results = {}
        
        # Create datasets and dataloaders
        text_dataset = MovieTextDataset(test_df, self.text_preprocessor, self.text_label_mapping)
        text_dataloader = DataLoader(text_dataset, batch_size=batch_size)
        
        image_dataset = MoviePosterDataset(test_df, self.image_transform, self.image_label_mapping)
        image_dataloader = DataLoader(image_dataset, batch_size=batch_size)
        
        multimodal_dataset = MultiModalDataset(
            test_df, self.text_preprocessor, self.image_transform, self.multimodal_label_mapping
        )
        multimodal_dataloader = DataLoader(multimodal_dataset, batch_size=batch_size)
        
        # Evaluate text model
        if self.text_model is not None:
            results['text'] = self.evaluate_text_model(text_dataloader)
        
        # Evaluate image model
        if self.image_model is not None:
            results['image'] = self.evaluate_image_model(image_dataloader)
        
        # Evaluate multimodal model
        if self.multimodal_model is not None:
            results['multimodal'] = self.evaluate_multimodal_model(multimodal_dataloader)
        
        return results
    
    def plot_comparison(self, results, metrics=['accuracy', 'macro avg'], save_path=None):
        """Plot comparison of models"""
        if not results:
            logging.error("No results to plot")
            return
        
        # Extract metrics
        model_names = list(results.keys())
        
        # Accuracy comparison
        accuracies = [results[model]['accuracy'] for model in model_names]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy comparison
        plt.subplot(1, 2, 1)
        bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red'])
        plt.ylim(0, 1.0)
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        
        # Add accuracy values
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                     f'{acc:.4f}', ha='center')
        
        # F1-score comparison
        if 'macro avg' in metrics:
            f1_scores = [results[model]['report']['macro avg']['f1-score'] for model in model_names]
            
            plt.subplot(1, 2, 2)
            bars = plt.bar(model_names, f1_scores, color=['blue', 'green', 'red'])
            plt.ylim(0, 1.0)
            plt.ylabel('F1 Score (Macro Avg)')
            plt.title('Model F1 Score Comparison')
            
            # Add F1 values
            for bar, f1 in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                         f'{f1:.4f}', ha='center')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Comparison plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrices(self, results, save_dir=None):
        """Plot confusion matrices for all models"""
        if not results:
            logging.error("No results to plot")
            return
        
        for model_name, model_results in results.items():
            if 'confusion_matrix' not in model_results:
                continue
            
            cm = model_results['confusion_matrix']
            
            # Get class names
            if model_name == 'text':
                class_names = [self.text_idx_to_genre[i] for i in range(len(self.text_idx_to_genre))]
            elif model_name == 'image':
                class_names = [self.image_idx_to_genre[i] for i in range(len(self.image_idx_to_genre))]
            else:
                class_names = [self.multimodal_idx_to_genre[i] for i in range(len(self.multimodal_idx_to_genre))]
            
            # Plot confusion matrix
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'{model_name.capitalize()} Model Confusion Matrix')
            plt.tight_layout()
            
            # Save or show
            if save_dir:
                save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
                plt.savefig(save_path)
                logging.info(f"Confusion matrix for {model_name} model saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
    
    def analyze_per_genre_performance(self, results, save_path=None):
        """Analyze and plot per-genre performance of all models"""
        if not results:
            logging.error("No results to plot")
            return
        
        # Get all genres from multimodal model (should have the same genres as others)
        if 'multimodal' in results:
            all_genres = sorted(list(results['multimodal']['report'].keys()))
            # Remove 'accuracy', 'macro avg', 'weighted avg'
            all_genres = [g for g in all_genres if g not in ['accuracy', 'macro avg', 'weighted avg']]
        else:
            # Fallback to text model
            all_genres = sorted(list(results['text']['report'].keys()))
            all_genres = [g for g in all_genres if g not in ['accuracy', 'macro avg', 'weighted avg']]
        
        # Extract F1 scores per genre
        model_names = list(results.keys())
        f1_scores = {}
        
        for model_name in model_names:
            f1_scores[model_name] = []
            for genre in all_genres:
                if genre in results[model_name]['report']:
                    f1_scores[model_name].append(results[model_name]['report'][genre]['f1-score'])
                else:
                    f1_scores[model_name].append(0)
        
        # Plot F1 scores per genre
        fig, ax = plt.subplots(figsize=(15, 8))
        
        x = np.arange(len(all_genres))
        width = 0.25
        multiplier = 0
        
        for model_name, scores in f1_scores.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, scores, width, label=model_name)
            multiplier += 1
        
        # Add labels and legend
        ax.set_xlabel('Genre')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score per Genre by Model')
        ax.set_xticks(x + width)
        ax.set_xticklabels(all_genres, rotation=45, ha='right')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Per-genre performance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_model_errors(self, results, save_dir=None):
        """Analyze and visualize common errors for each model"""
        if not results:
            logging.error("No results to plot")
            return
        
        for model_name, model_results in results.items():
            if 'predictions' not in model_results or 'labels' not in model_results:
                continue
            
            predictions = model_results['predictions']
            labels = model_results['labels']
            
            # Get class names
            if model_name == 'text':
                idx_to_genre = self.text_idx_to_genre
            elif model_name == 'image':
                idx_to_genre = self.image_idx_to_genre
            else:
                idx_to_genre = self.multimodal_idx_to_genre
            
            # Find misclassified examples
            error_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l]
            
            # Count most common error types
            error_pairs = {}
            for i in error_indices:
                true_genre = idx_to_genre[labels[i]]
                pred_genre = idx_to_genre[predictions[i]]
                error_pair = (true_genre, pred_genre)
                error_pairs[error_pair] = error_pairs.get(error_pair, 0) + 1
            
            # Get top 10 most common errors
            top_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Plot
            plt.figure(figsize=(12, 6))
            error_labels = [f"{true} → {pred}" for (true, pred), _ in top_errors]
            error_counts = [count for _, count in top_errors]
            
            plt.barh(error_labels, error_counts, color='salmon')
            plt.xlabel('Count')
            plt.ylabel('Misclassification')
            plt.title(f'{model_name.capitalize()} Model - Top 10 Misclassification Types')
            plt.tight_layout()
            
            # Save or show
            if save_dir:
                save_path = os.path.join(save_dir, f'{model_name}_error_analysis.png')
                plt.savefig(save_path)
                logging.info(f"Error analysis for {model_name} model saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
    
    def generate_comparison_report(self, results, save_path=None):
        """Generate a comprehensive comparison report"""
        if not results:
            logging.error("No results for report")
            return
        
        # Create report dictionary
        report = {
            'model_performance': {
                model_name: {
                    'accuracy': results[model_name]['accuracy'],
                    'macro_avg_f1': results[model_name]['report']['macro avg']['f1-score'],
                    'weighted_avg_f1': results[model_name]['report']['weighted avg']['f1-score']
                }
                for model_name in results.keys()
            },
            'per_genre_performance': {},
            'model_comparison': {}
        }
        
        # Get all genres from multimodal model
        if 'multimodal' in results:
            all_genres = sorted([g for g in results['multimodal']['report'].keys() 
                                 if g not in ['accuracy', 'macro avg', 'weighted avg']])
        else:
            all_genres = sorted([g for g in results['text']['report'].keys() 
                                 if g not in ['accuracy', 'macro avg', 'weighted avg']])
        
        # Per genre performance
        for genre in all_genres:
            report['per_genre_performance'][genre] = {}
            for model_name in results.keys():
                if genre in results[model_name]['report']:
                    report['per_genre_performance'][genre][model_name] = {
                        'precision': results[model_name]['report'][genre]['precision'],
                        'recall': results[model_name]['report'][genre]['recall'],
                        'f1_score': results[model_name]['report'][genre]['f1-score'],
                        'support': results[model_name]['report'][genre]['support']
                    }
        
        # Model comparison
        if 'multimodal' in results:
            # Compare with text
            if 'text' in results:
                text_acc = results['text']['accuracy']
                mm_acc = results['multimodal']['accuracy']
                improvement = ((mm_acc - text_acc) / text_acc) * 100
                report['model_comparison']['multimodal_vs_text'] = {
                    'accuracy_improvement': improvement,
                    'better_genres': [],
                    'worse_genres': []
                }
                
                # Per genre comparison
                for genre in all_genres:
                    if genre in results['text']['report'] and genre in results['multimodal']['report']:
                        text_f1 = results['text']['report'][genre]['f1-score']
                        mm_f1 = results['multimodal']['report'][genre]['f1-score']
                        if mm_f1 > text_f1:
                            report['model_comparison']['multimodal_vs_text']['better_genres'].append({
                                'genre': genre,
                                'improvement': ((mm_f1 - text_f1) / text_f1) * 100
                            })
                        else:
                            report['model_comparison']['multimodal_vs_text']['worse_genres'].append({
                                'genre': genre,
                                'decline': ((text_f1 - mm_f1) / text_f1) * 100
                            })
            
            # Compare with image
            if 'image' in results:
                image_acc = results['image']['accuracy']
                mm_acc = results['multimodal']['accuracy']
                improvement = ((mm_acc - image_acc) / image_acc) * 100
                report['model_comparison']['multimodal_vs_image'] = {
                    'accuracy_improvement': improvement,
                    'better_genres': [],
                    'worse_genres': []
                }
                
                # Per genre comparison
                for genre in all_genres:
                    if genre in results['image']['report'] and genre in results['multimodal']['report']:
                        image_f1 = results['image']['report'][genre]['f1-score']
                        mm_f1 = results['multimodal']['report'][genre]['f1-score']
                        if mm_f1 > image_f1:
                            report['model_comparison']['multimodal_vs_image']['better_genres'].append({
                                'genre': genre,
                                'improvement': ((mm_f1 - image_f1) / image_f1) * 100
                            })
                        else:
                            report['model_comparison']['multimodal_vs_image']['worse_genres'].append({
                                'genre': genre,
                                'decline': ((image_f1 - mm_f1) / image_f1) * 100
                            })
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logging.info(f"Comparison report saved to {save_path}")
        
        return report

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Model Comparison Evaluation')
    
    parser.add_argument('--test_data', type=str, default='data/splits/test.pkl',
                       help='Path to test data')
    
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                       help='Directory to save results')
    
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       help='Device to run evaluation on')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Load test data
    try:
        test_df = pd.read_pickle(args.test_data)
        logging.info(f"Loaded {len(test_df)} test samples from {args.test_data}")
    except Exception as e:
        logging.error(f"Error loading test data: {str(e)}")
        return
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Create evaluator
    evaluator = ModelEvaluator(device=device)
    
    # Evaluate all models
    results = evaluator.evaluate_all_models(test_df, batch_size=args.batch_size)
    
    # Save raw results
    with open(os.path.join(args.output_dir, 'raw_results.json'), 'w') as f:
        # Remove numpy arrays from results for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                'accuracy': model_results['accuracy'],
                'loss': model_results['loss'],
                'report': model_results['report']
            }
        json.dump(serializable_results, f, indent=2)
    
    # Plot comparison
    evaluator.plot_comparison(results, save_path=os.path.join(args.output_dir, 'model_comparison.png'))
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices(results, save_dir=args.output_dir)
    
    # Analyze per-genre performance
    evaluator.analyze_per_genre_performance(results, save_path=os.path.join(args.output_dir, 'per_genre_performance.png'))
    
    # Analyze model errors
    evaluator.analyze_model_errors(results, save_dir=args.output_dir)
    
    # Generate comparison report
    evaluator.generate_comparison_report(results, save_path=os.path.join(args.output_dir, 'comparison_report.json'))
    
    logging.info("Model comparison evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()