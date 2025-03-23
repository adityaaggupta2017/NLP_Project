import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
import os

# Create models_lstm directory if it doesn't exist
os.makedirs('models_lstm', exist_ok=True)

# Set seeds as in the original code base . 
torch.manual_seed(1994)
np.random.seed(1994)
random.seed(1994)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# Data paths 
TRAIN_PATH = r"C:\Git\NLP_Project\NLP_Project\data\train.jsonl"
VAL_PATH = r"C:\Git\NLP_Project\NLP_Project\data\validation.jsonl"
TEST_PATH = r"C:\Git\NLP_Project\NLP_Project\data\test.jsonl"

# Pretrained GloVe embeddings file path
GLOVE_PATH = "/kaggle/input/glove-embeddings/glove.twitter.27B.200d.txt" # 200d embeddings 

# Model hyperparameters
EMBED_DIM = 200       # embedding dimension
MSG_HIDDEN = 100      # hidden size for message encoder (bidirectional -> output 200)
CONV_HIDDEN = 200     # hidden size for conversation encoder
DROPOUT = 0.2        # dropout rate
BATCH_SIZE = 1       # batch size
EPOCHS = 80         # number of epochs 
LR = 0.003         # learning rate
GRAD_CLIP = 1.0    # gradient clipping 

# Loss pos_weight (for positive class, e.g. lies)
POS_WEIGHT = 10.0

# Preprocessing tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# Use game scores (power) as additional feature
USE_GAME_SCORES = True

# Model save directory
MODEL_DIR = "models_lstm"

# this is the diplomacy dataset class 
class DiplomacyDataset(torch.utils.data.Dataset):
   
    def __init__(self, path, max_tokens_per_msg=50, max_messages=50, use_game_scores=False):
        super().__init__()
        self.data = []
        self.max_tokens_per_msg = max_tokens_per_msg
        self.max_messages = max_messages
        self.use_game_scores = use_game_scores
        
        # Read and process each line in the JSONL file.
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages", [])
                labels = record.get("sender_labels", [])
                # Only use game_score_delta if use_game_scores is True, otherwise default to None.
                game_scores = record.get("game_score_delta", None) if use_game_scores else None
                
                filtered_msgs, filtered_lbls = [], []
                filtered_scores = []
                # If game scores are missing, set them to zero for every message.
                if game_scores is None:
                    game_scores = [0] * len(messages)
                
                # Iterate through messages, labels, and game scores in parallel.
                for m, l, g in zip(messages, labels, game_scores):
                    # Accept only valid boolean labels (or their string representations).
                    if l in [True, False, "true", "false", "True", "False"]:
                        filtered_msgs.append(m)
                        # Convert string labels to 0/1 (0 for false, 1 for true).
                        if isinstance(l, str):
                            filtered_lbls.append(1 if l.lower() == "true" else 0)
                        else:
                            filtered_lbls.append(1 if l else 0)
                        filtered_scores.append(g)
                
                # Skip records with no valid messages.
                if len(filtered_msgs) == 0:
                    continue
                
                # Store the processed conversation as a tuple of (messages, labels, game scores).
                self.data.append((filtered_msgs, filtered_lbls, filtered_scores))
        
        # Build vocabulary from the dataset.
        self._build_vocab()

    def _tokenize(self, text):
     
        tokens = text.lower().replace("\n", " ").split()
        out = []
        for t in tokens:
            if any(ch.isdigit() for ch in t):
                out.append("<NUM>")
            else:
                out.append(t)
        return out

    def _build_vocab(self):
      
        token_freq = Counter()
        for conv, _, _ in self.data:
            for msg in conv:
                tokens = self._tokenize(msg)
                token_freq.update(tokens)
        
        # Initialize vocabulary with special tokens.
        self.ix2tok = [PAD_TOKEN, UNK_TOKEN]
        # Add tokens in order of decreasing frequency.
        for tok, freq in token_freq.most_common():
            self.ix2tok.append(tok)
        # Create the token-to-index mapping.
        self.tok2ix = {t: i for i, t in enumerate(self.ix2tok)}

    def __len__(self):
       
        return len(self.data)

    def __getitem__(self, idx):
        
        conv, lbls, scores = self.data[idx]
        tokenized_conv = []
        for msg in conv:
            toks = self._tokenize(msg)
            # Convert tokens to indices using the vocabulary, defaulting to UNK_TOKEN if not found.
            tok_ix = [self.tok2ix.get(t, self.tok2ix[UNK_TOKEN]) for t in toks]
            tokenized_conv.append(tok_ix)
        # Convert each game score to a float.
        scores = [float(s) for s in scores]
        return tokenized_conv, lbls, scores


# Custom collate function to pad sequences for a batch of conversations.

def collate_fn(batch):
   
    # Determine maximum number of messages in any conversation in the batch.
    max_msg_count = max(len(item[0]) for item in batch)
    # Determine maximum number of tokens in any message.
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
        # Pad or truncate each conversation to max_msg_count messages.
        for i in range(max_msg_count):
            if i < num_msgs:
                # Pad the message to max_token_count tokens.
                msg = conv[i] + [0]*(max_token_count - len(conv[i]))
                conv_tokens.append(msg)
                conv_labels.append(lbls[i])
                conv_mask.append(1)  # Mark this message as valid.
                conv_scores.append(scores[i])
            else:
                # Pad missing messages with zeros.
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
  
  
  # this is referenced from the orignal code base . 

class HierarchicalLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, msg_hidden, conv_hidden, dropout=0.3, num_layers=1, use_game_scores=False):

        super().__init__()
        self.use_game_scores = use_game_scores
        
        # Embedding layer: maps token indices to vectors.
        # padding_idx=0 ensures that the PAD_TOKEN (index 0) gets a vector of zeros.
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Dropout layer to prevent overfitting.
        self.dropout = nn.Dropout(dropout)
        
        # Message encoder: a bidirectional LSTM that processes each message.
        # Because it's bidirectional, its output size is 2 * msg_hidden.
        self.msg_encoder = nn.LSTM(
            input_size=embed_dim,       # Input size is the embedding dimension.
            hidden_size=msg_hidden,     # Hidden state dimension.
            batch_first=True,           # Input and output tensors are provided as (batch, seq, feature).
            bidirectional=True,         # Use bidirectional LSTM to capture context from both directions.
            num_layers=num_layers       # Number of stacked LSTM layers.
        )
        
        # Conversation encoder: a unidirectional LSTM that processes the sequence of message representations.
        # The input size is 2*msg_hidden since each message vector is produced by the bidirectional LSTM.
        self.conv_encoder = nn.LSTM(
            input_size=msg_hidden * 2,  # Each message is represented as a concatenation from both directions.
            hidden_size=conv_hidden,    # Hidden state dimension at conversation level.
            batch_first=True,           # Batch dimension first.
            bidirectional=False,        # Unidirectional for conversation encoding.
            num_layers=num_layers       # Number of stacked LSTM layers.
        )
        
        # If using game scores as an additional feature, increase the classifier input dimension by 1.
        classifier_in = conv_hidden + 1 if self.use_game_scores else conv_hidden
        
        # Final classifier: a linear layer mapping the conversation encoder output (optionally with game scores concatenated)
        # to 2 output classes.
        self.classifier = nn.Linear(classifier_in, 2)

    def forward(self, tokens, mask, game_scores=None):
       
        B, M, T = tokens.shape
        # Reshape tokens from [B, M, T] to [B*M, T] so that each message is treated individually.
        tokens = tokens.view(B * M, T)
        
        # Embed the tokens: output shape becomes [B*M, T, embed_dim].
        emb = self.embedding(tokens)
        emb = self.dropout(emb)  # Apply dropout to embeddings.
        
        # Process each message with the bidirectional LSTM.
        # out has shape [B*M, T, 2*msg_hidden] (concatenation of forward and backward hidden states).
        out, _ = self.msg_encoder(emb)
        
        # Apply max pooling over the token dimension (T) to obtain a fixed-length vector per message.
        # The result has shape [B*M, 2*msg_hidden].
        out, _ = torch.max(out, dim=1)
        
        # Reshape back to [B, M, 2*msg_hidden] to reassemble messages into conversations.
        msg_vecs = out.view(B, M, -1)
        msg_vecs = self.dropout(msg_vecs)  # Apply dropout to message representations.
        
        # Process the sequence of message vectors with the conversation encoder (unidirectional LSTM).
        # conv_out has shape [B, M, conv_hidden].
        conv_out, _ = self.conv_encoder(msg_vecs)
        conv_out = self.dropout(conv_out)  # Apply dropout to conversation-level representations.
        
        # If the power (game scores) feature is used, concatenate it as an extra feature to conv_out.
        # First, unsqueeze game_scores to shape [B, M, 1] then concatenate along the feature dimension.
        if self.use_game_scores and game_scores is not None:
            game_scores = game_scores.unsqueeze(-1)
            conv_out = torch.cat([conv_out, game_scores], dim=2)
        
        # Pass the conversation-level representations (optionally with game scores) through the classifier.
        # The output logits have shape [B, M, 2] (for 2 classes).
        logits = self.classifier(conv_out)
        return logits

def sequence_cross_entropy_with_logits(logits, targets, mask):
   
    B, M, C = logits.shape  # B: batch size, M: number of messages, C: number of classes (2)
    # Flatten logits and targets so that each valid message is considered individually.
    logits_flat = logits.view(B * M, C)
    targets_flat = targets.view(B * M)
    mask_flat = mask.view(B * M).float()  # Convert mask to float for multiplication.
    
    # Compute the cross-entropy loss for each message.
    # The weight tensor assigns higher weight (POS_WEIGHT) to the positive class.
    ce = F.cross_entropy(logits_flat, targets_flat, reduction='none', weight=torch.tensor([1.0, POS_WEIGHT]).to(DEVICE))
    # Zero out loss for padded messages by multiplying with the mask.
    ce = ce * mask_flat
    # Return the average loss over all valid messages.
    return ce.sum() / (mask_flat.sum() + 1e-8)


def compute_accuracy(logits, targets, mask):
   
    # Get predictions by taking the index of the maximum logit for each message.
    preds = logits.argmax(dim=-1)
    # Compare predictions with targets and apply the mask to consider only valid messages.
    correct = (preds == targets) * (mask == 1)
    total = mask.sum()  # Total number of valid messages.
    # Compute and return accuracy.
    return (correct.sum().float() / (total.float() + 1e-8)).item()


def train_one_epoch(model, loader, optimizer):
   
    model.train()  # Set the model to training mode.
    total_loss = 0.0
    total_acc = 0.0
    all_preds = []  # List to store predictions for all valid messages.
    all_labels = []  # List to store corresponding ground truth labels.
    count = 0  # Number of batches processed.
    
    for tokens_batch, labels_batch, mask_batch, scores_batch in loader:
        # Move tensors to the specified device.
        tokens_batch = tokens_batch.to(DEVICE)
        labels_batch = labels_batch.to(DEVICE)
        mask_batch = mask_batch.to(DEVICE)
        scores_batch = scores_batch.to(DEVICE)
        
        optimizer.zero_grad()  # Reset gradients.
        logits = model(tokens_batch, mask_batch, scores_batch)  # Forward pass.
        loss = sequence_cross_entropy_with_logits(logits, labels_batch, mask_batch)  # Compute loss.
        loss.backward()  # Backpropagation.
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # Gradient clipping.
        optimizer.step()  # Update parameters.
        
        # Compute batch accuracy.
        acc = compute_accuracy(logits, labels_batch, mask_batch)
        total_loss += loss.item()
        total_acc += acc
        
        # Gather predictions and labels for computing macro F1 score later.
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels_batch.cpu().numpy()
        mask_np = mask_batch.cpu().numpy()
        for p_row, l_row, m_row in zip(preds, labels_np, mask_np):
            for p, l, m in zip(p_row, l_row, m_row):
                if m == 1:  # Consider only valid (non-padded) messages.
                    all_preds.append(p)
                    all_labels.append(l)
        count += 1

    # Compute macro F1 score using scikit-learn's f1_score function.
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / count, total_acc / count, macro_f1


def eval_model(model, loader):
    
    model.eval()  # Set the model to evaluation mode.
    total_loss = 0.0
    total_acc = 0.0
    all_preds = []  # Store predictions over valid messages.
    all_labels = []  # Store corresponding ground truth labels.
    count = 0
    
    with torch.no_grad():  # Disable gradient computation.
        for tokens_batch, labels_batch, mask_batch, scores_batch in loader:
            tokens_batch = tokens_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)
            scores_batch = scores_batch.to(DEVICE)
            
            logits = model(tokens_batch, mask_batch, scores_batch)  # Forward pass.
            loss = sequence_cross_entropy_with_logits(logits, labels_batch, mask_batch)  # Compute loss.
            acc = compute_accuracy(logits, labels_batch, mask_batch)  # Compute accuracy.
            
            total_loss += loss.item()
            total_acc += acc
            
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels_np = labels_batch.cpu().numpy()
            mask_np = mask_batch.cpu().numpy()
            for p_row, l_row, m_row in zip(preds, labels_np, mask_np):
                for p, l, m in zip(p_row, l_row, m_row):
                    if m == 1:  # Only consider valid messages.
                        all_preds.append(p)
                        all_labels.append(l)
            count += 1
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / count, total_acc / count, macro_f1


# Loading the  datasets
train_dataset = DiplomacyDataset(TRAIN_PATH, use_game_scores=True)
val_dataset = DiplomacyDataset(VAL_PATH, use_game_scores=True) if VAL_PATH else None
test_dataset = DiplomacyDataset(TEST_PATH, use_game_scores=True)

vocab_size = len(train_dataset.tok2ix)
print("Vocabulary size:", vocab_size)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) if val_dataset else None
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Instantiate the model with use_game_scores=True
model = HierarchicalLSTM(vocab_size=vocab_size, embed_dim=EMBED_DIM, msg_hidden=MSG_HIDDEN, conv_hidden=CONV_HIDDEN, dropout=DROPOUT, num_layers=1, use_game_scores=True).to(DEVICE)

# Load pretrained GloVe embeddings if desired (not used in this version, so we skip this step)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)

# Learning Rate Scheduler (ReduceLROnPlateau) based on validation loss
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training loop
best_val_loss = float('inf')
best_val_f1 = 0.0
best_epoch = -1

for epoch in range(EPOCHS):
    train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer)
    if val_loader:
        val_loss, val_acc, val_f1 = eval_model(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Macro F1: {train_f1:.4f}\n",
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}")
        scheduler.step(val_loss)
        
        # Save model if validation loss is better
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save the best model
            model_path = os.path.join(MODEL_DIR, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'vocab': train_dataset.tok2ix
            }, model_path)
            print(f"Saved best model at epoch {epoch+1} with val F1: {val_f1:.4f}")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Macro F1: {train_f1:.4f}")
        
        # If no validation set, save model periodically
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(MODEL_DIR, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'vocab': train_dataset.tok2ix
            }, model_path)

# Save final model
final_model_path = os.path.join(MODEL_DIR, 'final_model.pt')
torch.save({
    'epoch': EPOCHS - 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab': train_dataset.tok2ix
}, final_model_path)
print(f"Saved final model after {EPOCHS} epochs")

# Evaluate on test set
test_loss, test_acc, test_f1 = eval_model(model, test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}")

# If there was a validation set, load and evaluate the best model
if val_loader and os.path.exists(os.path.join(MODEL_DIR, 'best_model.pt')):
    print(f"Loading best model from epoch {best_epoch+1}...")
    checkpoint = torch.load(os.path.join(MODEL_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    best_test_loss, best_test_acc, best_test_f1 = eval_model(model, test_loader)
    print(f"Best Model (Epoch {best_epoch+1}) | "
          f"Test Loss: {best_test_loss:.4f}, Test Acc: {best_test_acc:.4f}, Test Macro F1: {best_test_f1:.4f}")

# Inference on one conversation from the test set
model.eval()
sample_conv, sample_lbls, sample_scores = test_dataset[0]

tokens_batch, labels_batch, mask_batch, scores_batch = collate_fn([(sample_conv, sample_lbls, sample_scores)])
tokens_batch = tokens_batch.to(DEVICE)
labels_batch = labels_batch.to(DEVICE)
mask_batch = mask_batch.to(DEVICE)
scores_batch = scores_batch.to(DEVICE)

logits = model(tokens_batch, mask_batch, scores_batch)  # [1, M, 2]
preds = logits.argmax(dim=-1).squeeze(0)  # [M]

print("Conversation has", len(sample_conv), "messages.")
for i, (msg, gold_label) in enumerate(zip(sample_conv, sample_lbls)):
    predicted_label = preds[i].item()
    predicted_bool = (predicted_label == 1)
    gold_bool = bool(gold_label)
    print(f"Msg {i} -> Pred: {predicted_bool}, Gold: {gold_bool}")