
import os
import torch
import json
import numpy as np
from text_preprocessor import TextPreprocessor
from model import LSTMWithAttentionMultiLabel

def predict_groups_from_text(text, model_dir="lstm_attention"):
    """Predict groups for a new movie overview text"""
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
