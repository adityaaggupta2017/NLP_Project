import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config import *
from dataset import EnhancedDeceptionDataset, EnhancedBalancedSampler, custom_collate_fn
from model import ImprovedDeceptionModel
from utils import train, evaluate, plot_confusion_matrix, plot_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train the Improved Deception Detection Model with ConceptNet')
    parser.add_argument('--train_path', type=str, default=TRAIN_PATH, help='Path to the training data')
    parser.add_argument('--val_path', type=str, default=VAL_PATH, help='Path to the validation data')
    parser.add_argument('--test_path', type=str, default=TEST_PATH, help='Path to the test data')
    parser.add_argument('--conceptnet_path', type=str, default=CONCEPTNET_PATH, help='Path to the ConceptNet Numberbatch embeddings')
    parser.add_argument('--model_name', type=str, default=TRANSFORMER_MODEL, help='Transformer model name')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LR, help='Learning rate')
    parser.add_argument('--use_game_scores', action='store_true', default=USE_GAME_SCORES, help='Use game scores')
    parser.add_argument('--oversample_factor', type=int, default=OVERSAMPLING_FACTOR, help='Oversampling factor for truth class')
    parser.add_argument('--truth_focal_weight', type=float, default=TRUTH_FOCAL_WEIGHT, help='Focal loss weight for truth class')
    parser.add_argument('--early_stopping_patience', type=int, default=EARLY_STOPPING_PATIENCE, help='Early stopping patience')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=GRADIENT_ACCUMULATION_STEPS, help='Gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save model outputs')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    print("Loading datasets...")
    train_dataset = EnhancedDeceptionDataset(args.train_path, tokenizer, args.conceptnet_path, use_game_scores=args.use_game_scores)
    val_dataset = EnhancedDeceptionDataset(args.val_path, tokenizer, args.conceptnet_path, use_game_scores=args.use_game_scores)
    test_dataset = EnhancedDeceptionDataset(args.test_path, tokenizer, args.conceptnet_path, use_game_scores=args.use_game_scores)
    
    train_sampler = EnhancedBalancedSampler(train_dataset, oversample_factor=args.oversample_factor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn
    )
    
    train_class_counts = train_dataset.class_counts
    total_samples = sum(train_class_counts.values())
    weight_0 = total_samples / (train_class_counts.get(0, 1) * 2)
    weight_1 = total_samples / (train_class_counts.get(1, 1) * 2)
    class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(DEVICE)
    print(f"Class weights: Truth = {weight_0:.4f}, Lie = {weight_1:.4f}")
    
    print("Initializing improved model...")
    model = ImprovedDeceptionModel(args.model_name, use_game_scores=args.use_game_scores).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    print(f"Starting training for {args.epochs} epochs...")
    best_truth_f1 = 0
    best_macro_f1 = 0
    no_improvement_count = 0
    
    train_losses = []
    train_truth_f1s = []
    train_lie_f1s = []
    train_macro_f1s = []
    val_losses = []
    val_truth_f1s = []
    val_lie_f1s = []
    val_macro_f1s = []
    
    for epoch in range(args.epochs):
        train_loss, train_truth_f1, train_lie_f1, train_macro_f1, train_cm = train(
            model, train_loader, optimizer, scheduler, DEVICE, class_weights,
            truth_focal_weight=args.truth_focal_weight,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        val_loss, val_truth_f1, val_lie_f1, val_macro_f1, val_cm = evaluate(
            model, val_loader, DEVICE, class_weights,
            truth_focal_weight=args.truth_focal_weight
        )
        
        train_losses.append(train_loss)
        train_truth_f1s.append(train_truth_f1)
        train_lie_f1s.append(train_lie_f1)
        train_macro_f1s.append(train_macro_f1)
        
        val_losses.append(val_loss)
        val_truth_f1s.append(val_truth_f1)
        val_lie_f1s.append(val_lie_f1)
        val_macro_f1s.append(val_macro_f1)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Truth F1: {train_truth_f1:.4f}, Lie F1: {train_lie_f1:.4f}, Macro F1: {train_macro_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Truth F1: {val_truth_f1:.4f}, Lie F1: {val_lie_f1:.4f}, Macro F1: {val_macro_f1:.4f}")
        print("Confusion Matrix (Val):")
        print(val_cm)
        print("-" * 50)
        plot_confusion_matrix(val_cm, epoch, 'val')
        
        improved = False
        if val_truth_f1 > best_truth_f1:
            best_truth_f1 = val_truth_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_truth_f1_model.pt'))
            print(f"Saved new best model with Truth F1: {val_truth_f1:.4f}")
            improved = True
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_macro_f1_model.pt'))
            print(f"Saved new best model with Macro F1: {val_macro_f1:.4f}")
            improved = True
        
        if not improved:
            no_improvement_count += 1
            if no_improvement_count >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            no_improvement_count = 0
    
    plot_metrics(train_losses, val_losses, 'Loss')
    plot_metrics(train_truth_f1s, val_truth_f1s, 'Truth F1')
    plot_metrics(train_lie_f1s, val_lie_f1s, 'Lie F1')
    plot_metrics(train_macro_f1s, val_macro_f1s, 'Macro F1')
    
    print("\nEvaluating best model (by Truth F1) on test set:")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_truth_f1_model.pt')))
    test_loss, test_truth_f1, test_lie_f1, test_macro_f1, test_cm = evaluate(
        model, test_loader, DEVICE, class_weights,
        truth_focal_weight=args.truth_focal_weight
    )
    print(f"\nTest Results - Truth F1 Model:")
    print(f"Loss: {test_loss:.4f}, Truth F1: {test_truth_f1:.4f}, Lie F1: {test_lie_f1:.4f}, Macro F1: {test_macro_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    
    print("\nEvaluating best model (by Macro F1) on test set:")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_macro_f1_model.pt')))
    test_loss, test_truth_f1, test_lie_f1, test_macro_f1, test_cm = evaluate(
        model, test_loader, DEVICE, class_weights,
        truth_focal_weight=args.truth_focal_weight
    )
    print(f"\nTest Results - Macro F1 Model:")
    print(f"Loss: {test_loss:.4f}, Truth F1: {test_truth_f1:.4f}, Lie F1: {test_lie_f1:.4f}, Macro F1: {test_macro_f1:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

if __name__ == "__main__":
    main()