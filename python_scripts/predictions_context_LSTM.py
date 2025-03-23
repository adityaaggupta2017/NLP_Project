#!/usr/bin/env python3
# Inference script for Hierarchical LSTM with Power Feature

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import sys
from collections import Counter
from sklearn.metrics import f1_score, classification_report

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with trained hierarchical LSTM model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model checkpoint file')
    parser.add_argument('--input_path', type=str, help='Path to JSONL file with conversation data')
    parser.add_argument('--output_path', type=str, help='Path to save predictions')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--sample_message', type=str, help='Sample message to analyze')
    parser.add_argument('--power_delta', type=float, default=0.0, help='Power delta value for sample message')
    parser.add_argument('--conversation_file', type=str, help='JSON file containing a single conversation to analyze')
    return parser.parse_args()

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def _tokenize(text):
    """Tokenize a message text."""
    tokens = text.lower().replace("\n", " ").split()
    out = []
    for t in tokens:
        if any(ch.isdigit() for ch in t):
            out.append("<NUM>")
        else:
            out.append(t)
    return out

class DiplomacyDataset(Dataset):
    def __init__(self, path, vocab, max_tokens_per_msg=50, max_messages=50, use_game_scores=True):
        super().__init__()
        self.data = []
        self.tok2ix = vocab
        self.max_tokens_per_msg = max_tokens_per_msg
        self.max_messages = max_messages
        self.use_game_scores = use_game_scores
        
        # Read and process each line in the JSONL file
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages", [])
                labels = record.get("sender_labels", [])
                # Only use game_score_delta if use_game_scores is True, otherwise default to None
                game_scores = record.get("game_score_delta", None) if use_game_scores else None
                
                filtered_msgs, filtered_lbls = [], []
                filtered_scores = []
                # If game scores are missing, set them to zero for every message
                if game_scores is None:
                    game_scores = [0] * len(messages)
                
                # Iterate through messages, labels, and game scores in parallel
                for m, l, g in zip(messages, labels, game_scores):
                    # Accept only valid boolean labels (or their string representations)
                    if l in [True, False, "true", "false", "True", "False"]:
                        filtered_msgs.append(m)
                        # Convert string labels to 0/1 (0 for false, 1 for true)
                        if isinstance(l, str):
                            filtered_lbls.append(1 if l.lower() == "true" else 0)
                        else:
                            filtered_lbls.append(1 if l else 0)
                        filtered_scores.append(g)
                
                # Skip records with no valid messages
                if len(filtered_msgs) == 0:
                    continue
                
                # Store the processed conversation as a tuple of (messages, labels, game scores)
                self.data.append((filtered_msgs, filtered_lbls, filtered_scores))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conv, lbls, scores = self.data[idx]
        tokenized_conv = []
        for msg in conv:
            toks = _tokenize(msg)
            # Convert tokens to indices using the vocabulary, defaulting to UNK_TOKEN if not found
            tok_ix = [self.tok2ix.get(t, self.tok2ix.get(UNK_TOKEN, 1)) for t in toks]
            tokenized_conv.append(tok_ix)
        # Convert each game score to a float
        scores = [float(s) for s in scores]
        return tokenized_conv, lbls, scores

class HierarchicalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, msg_hidden, conv_hidden, dropout=0.3, num_layers=1, use_game_scores=True):
        super().__init__()
        self.use_game_scores = use_game_scores
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Message encoder (bidirectional LSTM)
        self.msg_encoder = nn.LSTM(
            input_size=embed_dim,
            hidden_size=msg_hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers
        )
        
        # Conversation encoder (unidirectional LSTM)
        self.conv_encoder = nn.LSTM(
            input_size=msg_hidden * 2,
            hidden_size=conv_hidden,
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers
        )
        
        # Classifier
        classifier_in = conv_hidden + 1 if self.use_game_scores else conv_hidden
        self.classifier = nn.Linear(classifier_in, 2)

    def forward(self, tokens, mask, game_scores=None):
        B, M, T = tokens.shape
        # Reshape tokens from [B, M, T] to [B*M, T]
        tokens = tokens.view(B * M, T)
        
        # Embed tokens
        emb = self.embedding(tokens)
        emb = self.dropout(emb)
        
        # Process with message encoder
        out, _ = self.msg_encoder(emb)
        
        # Max pooling
        out, _ = torch.max(out, dim=1)
        
        # Reshape to [B, M, 2*msg_hidden]
        msg_vecs = out.view(B, M, -1)
        msg_vecs = self.dropout(msg_vecs)
        
        # Process with conversation encoder
        conv_out, _ = self.conv_encoder(msg_vecs)
        conv_out = self.dropout(conv_out)
        
        # Add game score feature if enabled
        if self.use_game_scores and game_scores is not None:
            game_scores = game_scores.unsqueeze(-1)
            conv_out = torch.cat([conv_out, game_scores], dim=2)
        
        # Final classification
        logits = self.classifier(conv_out)
        return logits

def collate_fn(batch):
    """Collate function for batching conversations."""
    # Determine maximum number of messages in any conversation in the batch
    max_msg_count = max(len(item[0]) for item in batch)
    # Determine maximum number of tokens in any message
    max_token_count = 0
    for item in batch:
        for msg in item[0]:
            max_token_count = max(max_token_count, len(msg))
    
    padded_tokens = []
    padded_labels = []
    mask = []
    padded_scores = []
    
    # For each conversation in the batch...
    for conv, lbls, scores in batch:
        num_msgs = len(conv)
        conv_tokens = []
        conv_labels = []
        conv_mask = []
        conv_scores = []
        # Pad or truncate each conversation to max_msg_count messages
        for i in range(max_msg_count):
            if i < num_msgs:
                # Pad the message to max_token_count tokens
                msg = conv[i] + [0]*(max_token_count - len(conv[i]))
                conv_tokens.append(msg)
                conv_labels.append(lbls[i])
                conv_mask.append(1)  # Mark this message as valid
                conv_scores.append(scores[i])
            else:
                # Pad missing messages with zeros
                conv_tokens.append([0]*max_token_count)
                conv_labels.append(0)
                conv_mask.append(0)
                conv_scores.append(0)
        padded_tokens.append(conv_tokens)
        padded_labels.append(conv_labels)
        mask.append(conv_mask)
        padded_scores.append(conv_scores)
    
    padded_tokens = torch.tensor(padded_tokens, dtype=torch.long)
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)
    mask = torch.tensor(mask, dtype=torch.long)
    padded_scores = torch.tensor(padded_scores, dtype=torch.float)
    return padded_tokens, padded_labels, mask, padded_scores


def process_single_message(message, power_delta, vocab):
    """Process a single message for inference."""
    # Tokenize the message
    tokens = _tokenize(message)
    # Convert tokens to indices
    token_indices = [vocab.get(t, vocab.get(UNK_TOKEN, 1)) for t in tokens]
    
    # Create a dummy conversation with a single message
    tokenized_conv = [token_indices]
    labels = [1]  # Dummy label
    scores = [float(power_delta)]
    
    # Batch it
    return ([tokenized_conv], [labels], [scores])


def process_conversation_file(file_path, vocab):
    """Process a conversation from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        conversation = json.load(f)
    
    messages = conversation.get('messages', [])
    power_deltas = conversation.get('game_score_delta', [0] * len(messages))
    
    # Tokenize each message
    tokenized_messages = []
    for msg in messages:
        tokens = _tokenize(msg)
        token_indices = [vocab.get(t, vocab.get(UNK_TOKEN, 1)) for t in tokens]
        tokenized_messages.append(token_indices)
    
    # Create dummy labels (not used for inference)
    dummy_labels = [1] * len(messages)
    
    return ([tokenized_messages], [dummy_labels], [power_deltas])


def evaluate_model(model, loader, device):
    """Evaluate model on a test dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for tokens_batch, labels_batch, mask_batch, scores_batch in loader:
            tokens_batch = tokens_batch.to(device)
            labels_batch = labels_batch.to(device)
            mask_batch = mask_batch.to(device)
            scores_batch = scores_batch.to(device)
            
            logits = model(tokens_batch, mask_batch, scores_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            # Gather predictions, labels, and probabilities for valid messages
            preds_np = preds.cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            probs_np = probs.cpu().numpy()
            mask_np = mask_batch.cpu().numpy()
            
            for p_row, l_row, prob_row, m_row in zip(preds_np, labels_np, probs_np, mask_np):
                for p, l, prob, m in zip(p_row, l_row, prob_row, m_row):
                    if m == 1:  # Only consider valid messages
                        all_preds.append(p)
                        all_labels.append(l)
                        all_probs.append(prob)
    
    # Compute metrics
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'accuracy': accuracy,
        'f1_score': f1
    }


def main():
    args = parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint.get('args', {})
    vocab = checkpoint.get('vocab', {})
    
    if not vocab:
        print("Error: Vocabulary not found in model checkpoint")
        sys.exit(1)
    
    # Create model with the same parameters
    model = HierarchicalLSTM(
        vocab_size=len(vocab),
        embed_dim=model_args.get('embed_dim', 200),
        msg_hidden=model_args.get('msg_hidden', 100),
        conv_hidden=model_args.get('conv_hidden', 200),
        dropout=model_args.get('dropout', 0.2),
        num_layers=1,
        use_game_scores=not model_args.get('no_power', False)
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Handle different types of input
    if args.sample_message:
        # Process a single message
        print(f"Analyzing message: '{args.sample_message}'")
        print(f"Power delta: {args.power_delta}")
        
        batch_data = process_single_message(args.sample_message, args.power_delta, vocab)
        tokens_batch, _, _ = batch_data
        tokens_batch, _, mask_batch, scores_batch = collate_fn(list(zip(*batch_data)))
        
        tokens_batch = tokens_batch.to(device)
        mask_batch = mask_batch.to(device)
        scores_batch = scores_batch.to(device)
        
        with torch.no_grad():
            logits = model(tokens_batch, mask_batch, scores_batch)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1)
        
        # Extract prediction for the single message
        message_pred = pred[0, 0].item()
        message_probs = probs[0, 0].cpu().numpy()
        
        print("\nPrediction Results:")
        print(f"  Predicted: {'Truth' if message_pred == 1 else 'Lie'}")
        print(f"  Confidence: {message_probs[message_pred]:.4f}")
        print(f"  Probabilities: Truth: {message_probs[1]:.4f}, Lie: {message_probs[0]:.4f}")
    
    elif args.conversation_file:
        # Process a conversation from a file
        print(f"Analyzing conversation from: {args.conversation_file}")
        
        batch_data = process_conversation_file(args.conversation_file, vocab)
        tokens_batch, _, scores_batch = batch_data
        tokens_batch, _, mask_batch, scores_batch = collate_fn(list(zip(*batch_data)))
        
        tokens_batch = tokens_batch.to(device)
        mask_batch = mask_batch.to(device)
        scores_batch = scores_batch.to(device)
        
        with torch.no_grad():
            logits = model(tokens_batch, mask_batch, scores_batch)
            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
        
        # Print predictions for each message in the conversation
        print("\nPredictions for conversation:")
        with open(args.conversation_file, 'r', encoding='utf-8') as f:
            conversation = json.load(f)
        
        messages = conversation.get('messages', [])
        for i, message in enumerate(messages):
            if i < preds.size(1) and mask_batch[0, i].item() == 1:
                message_pred = preds[0, i].item()
                message_probs = probs[0, i].cpu().numpy()
                
                print(f"\nMessage {i+1}: '{message[:50]}...' (truncated)")
                print(f"  Predicted: {'Truth' if message_pred == 1 else 'Lie'}")
                print(f"  Confidence: {message_probs[message_pred]:.4f}")
                print(f"  Probabilities: Truth: {message_probs[1]:.4f}, Lie: {message_probs[0]:.4f}")
    
    elif args.input_path:
        # Process a JSONL file with conversations
        print(f"Running inference on data from: {args.input_path}")
        
        # Create dataset and loader
        test_dataset = DiplomacyDataset(args.input_path, vocab, use_game_scores=model.use_game_scores)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        
        # Print metrics
        print("\nEvaluation Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1 Score: {results['f1_score']:.4f}")
        
        # Generate classification report
        labels = results['labels']
        preds = results['predictions']
        target_names = ['Lie', 'Truth']
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=target_names, digits=4))
        
        # Save predictions if output path is provided
        if args.output_path:
            output_dir = os.path.dirname(args.output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Collect conversation-level predictions
            all_convs = []
            for idx, (conv, lbls, _) in enumerate(test_dataset.data):
                conv_preds = []
                conv_probs = []
                
                # Match predictions to this conversation
                start_idx = 0
                for prev_idx in range(idx):
                    start_idx += len(test_dataset.data[prev_idx][0])
                
                for i in range(len(conv)):
                    pred_idx = start_idx + i
                    if pred_idx < len(results['predictions']):
                        conv_preds.append(bool(results['predictions'][pred_idx] == 1))
                        conv_probs.append(results['probabilities'][pred_idx].tolist())
                
                all_convs.append({
                    'messages': conv,
                    'true_labels': [bool(lbl) for lbl in lbls],
                    'predictions': conv_preds,
                    'probabilities': conv_probs
                })
            
            # Save to file
            with open(args.output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'conversations': all_convs,
                    'metrics': {
                        'accuracy': results['accuracy'],
                        'f1_score': results['f1_score']
                    }
                }, f, indent=2)
            
            print(f"\nSaved predictions to {args.output_path}")
    
    else:
        print("Error: Please provide either --sample_message, --conversation_file, or --input_path")
        sys.exit(1)


if __name__ == "__main__":
    main()