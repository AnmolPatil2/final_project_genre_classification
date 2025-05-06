import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import pickle
import logging
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import argparse

# Import from baseline models and fusion model
from multimodal_fusion_model import (
    TextEncoder, ImageEncoder, FusionModel, load_text_preprocessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("prediction_demo.log"),
        logging.StreamHandler()
    ]
)

class MovieGenrePredictor:
    """Predictor for movie genres using multi-modal model"""
    
    def __init__(self, model_path, text_preprocessor_path, label_mapping_path, device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the model weights
            text_preprocessor_path: Path to the text preprocessor
            label_mapping_path: Path to the label mapping
            device: Device to run the model on (default: auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logging.info(f"Using device: {self.device}")
        
        # Load text preprocessor
        self.text_preprocessor = load_text_preprocessor(text_preprocessor_path)
        if self.text_preprocessor is None:
            raise ValueError("Failed to load text preprocessor")
        
        # Load label mapping
        with open(label_mapping_path, 'r') as f:
            self.label_mapping = json.load(f)
        
        self.idx_to_genre = {int(v): k for k, v in self.label_mapping.items()}
        self.num_classes = len(self.label_mapping)
        
        logging.info(f"Loaded label mapping with {self.num_classes} classes")
        
        # Load model info
        model_info_path = os.path.join(os.path.dirname(model_path), "model_info.json")
        with open(model_info_path, 'r') as f:
            self.model_info = json.load(f)
        
        # Create model components
        self._create_model()
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logging.info("Model loaded successfully")
        
        # Set up image transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _create_model(self):
        """Create the model based on model info"""
        # Create text encoder
        text_encoder = TextEncoder(
            vocab_size=self.text_preprocessor.vocab_size,
            embedding_dim=self.model_info.get('text_embedding_dim', 300),
            hidden_dim=self.model_info.get('text_hidden_dim', 256),
            n_layers=self.model_info.get('text_n_layers', 2),
            bidirectional=self.model_info.get('text_bidirectional', True),
            dropout=self.model_info.get('dropout', 0.5),
            pad_idx=self.text_preprocessor.word2idx.get('<PAD>', 0)
        )
        
        # Create image encoder
        image_encoder = ImageEncoder(
            model_name=self.model_info.get('image_model_name', 'resnet18'),
            pretrained=False,  # Don't need pretrained weights since we're loading our own
            freeze_base=self.model_info.get('image_freeze_base', True)
        )
        
        # Create fusion model
        self.model = FusionModel(
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            output_dim=self.num_classes,
            fusion_method=self.model_info.get('fusion_method', 'concat'),
            dropout=self.model_info.get('dropout', 0.5)
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
    
    def predict(self, overview, image_path=None, image_url=None, top_k=3):
        """
        Predict genres for a movie based on overview and poster
        
        Args:
            overview: Movie overview text
            image_path: Path to local image file (optional)
            image_url: URL to image (optional)
            top_k: Number of top predictions to return
            
        Returns:
            List of (genre, probability) tuples for top k predictions
        """
        if image_path is None and image_url is None:
            raise ValueError("Either image_path or image_url must be provided")
        
        # Preprocess text
        text = torch.tensor(
            self.text_preprocessor.text_to_sequence(overview),
            dtype=torch.long
        ).unsqueeze(0).to(self.device)
        
        # Preprocess image
        if image_path is not None:
            # Load local image
            image = Image.open(image_path).convert('RGB')
        else:
            # Load image from URL
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Apply transforms
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model(text, image)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))
        
        # Convert to (genre, probability) pairs
        predictions = [
            (self.idx_to_genre[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predictions
    
    def visualize_prediction(self, overview, image_path=None, image_url=None, top_k=5, save_path=None):
        """
        Visualize genre predictions for a movie
        
        Args:
            overview: Movie overview text
            image_path: Path to local image file (optional)
            image_url: URL to image (optional)
            top_k: Number of top predictions to display
            save_path: Path to save the visualization (optional)
            
        Returns:
            None
        """
        # Get predictions
        predictions = self.predict(overview, image_path, image_url, top_k)
        
        # Get image for display
        if image_path is not None:
            image = Image.open(image_path).convert('RGB')
        else:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Display image
        ax1.imshow(image)
        ax1.set_title("Movie Poster")
        ax1.axis('off')
        
        # Display predictions
        genres, probs = zip(*predictions)
        ax2.barh(genres, probs, color='skyblue')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('Probability')
        ax2.set_title('Genre Predictions')
        
        # Add overview as text
        plt.figtext(0.5, 0.01, f"Overview: {overview[:300]}{'...' if len(overview) > 300 else ''}", 
                    ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Movie Genre Prediction Demo')
    
    parser.add_argument('--overview', type=str, required=True,
                       help='Movie overview text')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str, help='Path to movie poster image')
    group.add_argument('--image_url', type=str, help='URL to movie poster image')
    
    parser.add_argument('--model_path', type=str, default='models/multimodal/best_model.pt',
                       help='Path to the model weights')
    
    parser.add_argument('--preprocessor_path', type=str, default='models/text/preprocessor.pkl',
                       help='Path to the text preprocessor')
    
    parser.add_argument('--label_mapping_path', type=str, default='models/multimodal/label_mapping.json',
                       help='Path to the label mapping')
    
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to display')
    
    parser.add_argument('--output', type=str, help='Path to save the visualization')
    
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], 
                       help='Device to run the model on')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    # Create predictor
    predictor = MovieGenrePredictor(
        model_path=args.model_path,
        text_preprocessor_path=args.preprocessor_path,
        label_mapping_path=args.label_mapping_path,
        device=device
    )
    
    # Visualize prediction
    predictor.visualize_prediction(
        overview=args.overview,
        image_path=args.image_path,
        image_url=args.image_url,
        top_k=args.top_k,
        save_path=args.output
    )
    
    # Print predictions
    predictions = predictor.predict(
        overview=args.overview,
        image_path=args.image_path,
        image_url=args.image_url,
        top_k=args.top_k
    )
    
    print("\nMovie Genre Predictions:")
    for genre, prob in predictions:
        print(f"{genre}: {prob:.4f} ({prob*100:.2f}%)")

if __name__ == "__main__":
    main()