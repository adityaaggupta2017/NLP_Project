import os
import argparse
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, TRANSFORMER_MODEL, BATCH_SIZE, USE_GAME_SCORES, TRUTH_FOCAL_WEIGHT
from dataset import EnhancedDeceptionDataset, custom_collate_fn
from model import ImprovedDeceptionModel
from utils import evaluate, prepare_batch_for_model

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with the Improved Deception Detection Model')
    parser.add_argument('--test_path', type=str, required=True, help='Path to the test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights')
    parser.add_argument('--model_name', type=str, default=TRANSFORMER_MODEL, help='Transformer model name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for inference')
    parser.add_argument('--use_game_scores', action='store_true', default=USE_GAME_SCORES, help='Use game scores')
    parser.add_argument('--output_file', type=str, default='predictions.jsonl', help='File to save predictions')
    return parser.parse_args()

def predict_batch(model, batch, device):
    model.eval()
    batch_data = prepare_batch_for_model(batch, device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            context_input_ids=batch_data['context_input_ids'],
            context_attention_mask=batch_data['context_attention_mask'],
            game_scores=batch_data['scores'],
            batch_adj_matrix=batch_data['batch_adj_matrix']
        )
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)
    
    return predicted.cpu().numpy(), probabilities.cpu().numpy()

def main():
    args = parse_args()
    
    print(f"Using device: {DEVICE}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Loading test dataset...")
    test_dataset = EnhancedDeceptionDataset(args.test_path, tokenizer, use_game_scores=args.use_game_scores)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    print("Initializing model...")
    model = ImprovedDeceptionModel(args.model_name, use_game_scores=args.use_game_scores).to(DEVICE)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    
    print("Running inference...")
    
    all_predictions = []
    all_probabilities = []
    all_texts = []
    
    for batch in tqdm(test_loader, desc="Predicting"):
        pred_labels, pred_probs = predict_batch(model, batch, DEVICE)
        all_predictions.extend(pred_labels.tolist())
        all_probabilities.extend(pred_probs.tolist())
        all_texts.extend(batch['text'])
    
    # Create output
    results = []
    for text, pred, prob in zip(all_texts, all_predictions, all_probabilities):
        results.append({
            "text": text,
            "prediction": "Lie" if pred == 1 else "Truth",
            "lie_probability": float(prob[1]),
            "truth_probability": float(prob[0])
        })
    
    # Save predictions
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Predictions saved to {args.output_file}")
    
    # If the dataset has labels, also compute metrics
    if hasattr(test_dataset, 'labels') and len(test_dataset.labels) > 0:
        print("Computing evaluation metrics...")
        
        # Compute metrics manually
        true_labels = np.array(test_dataset.labels)
        pred_labels = np.array(all_predictions)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(true_labels, pred_labels)
        
        print("\nEvaluation Results:")
        print(classification_report(true_labels, pred_labels, target_names=['Truth', 'Lie'], digits=4))
        print("\nConfusion Matrix:")
        print(cm)

if __name__ == "__main__":
    main()