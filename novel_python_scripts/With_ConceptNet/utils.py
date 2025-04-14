import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report

class FocalWeightedLoss(nn.Module):
    def __init__(self, class_weights, truth_focal_weight=4.0):
        super(FocalWeightedLoss, self).__init__()
        self.class_weights = class_weights
        self.truth_focal_weight = truth_focal_weight
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        probs = F.softmax(logits, dim=1)
        truth_probs = probs[:, 0]  # probability for truth class
        truth_mask = (targets == 0).float()
        focal_weight = (1 - truth_probs) ** self.truth_focal_weight
        focal_loss = truth_mask * focal_weight * ce_loss + (1 - truth_mask) * ce_loss
        return focal_loss.mean()


def prepare_batch_for_model(batch, device):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    context_input_ids = batch['context_input_ids'].to(device) if 'context_input_ids' in batch else None
    context_attention_mask = batch['context_attention_mask'].to(device) if 'context_attention_mask' in batch else None
    labels = batch['label'].to(device)
    scores = batch['score'].to(device) if 'score' in batch else None
    conceptnet_features = batch['conceptnet_features'].to(device)
    batch_size = input_ids.size(0)
    
    # Build a simple adjacency matrix for messages in the same conversation
    batch_adj_matrix = torch.zeros((batch_size, batch_size), device=device)
    conv_ids = batch['conv_id'] if isinstance(batch['conv_id'], list) else batch['conv_id'].tolist()
    positions = batch['position'] if isinstance(batch['position'], list) else batch['position'].tolist()
    for i in range(batch_size):
        for j in range(batch_size):
            if conv_ids[i] == conv_ids[j]:
                if i == j:
                    batch_adj_matrix[i, j] = 1.0
                else:
                    distance = abs(positions[i] - positions[j])
                    batch_adj_matrix[i, j] = 1.0 / (distance + 1)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask,
        'labels': labels,
        'scores': scores,
        'batch_adj_matrix': batch_adj_matrix,
        'conceptnet_features': conceptnet_features
    }


def train(model, dataloader, optimizer, scheduler, device, class_weights, truth_focal_weight=4.0, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0
    all_labels = []
    all_preds = []
    loss_fn = FocalWeightedLoss(class_weights, truth_focal_weight)
    optimizer.zero_grad()
    accumulated_steps = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch_data = prepare_batch_for_model(batch, device)
        outputs = model(
            input_ids=batch_data['input_ids'], 
            attention_mask=batch_data['attention_mask'],
            context_input_ids=batch_data['context_input_ids'],
            context_attention_mask=batch_data['context_attention_mask'],
            game_scores=batch_data['scores'],
            batch_adj_matrix=batch_data['batch_adj_matrix'],
            conceptnet_features=batch_data['conceptnet_features']
        )
        loss = loss_fn(outputs, batch_data['labels'])
        loss = loss / gradient_accumulation_steps
        loss.backward()
        total_loss += loss.item() * gradient_accumulation_steps
        _, preds = torch.max(outputs, dim=1)
        all_labels.extend(batch_data['labels'].cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        accumulated_steps += 1
        if accumulated_steps % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
    
    if accumulated_steps % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
    
    avg_loss = total_loss / len(dataloader)
    try:
        truth_f1 = f1_score(all_labels, all_preds, pos_label=0, average='binary', zero_division=0)
        lie_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    except Exception:
        truth_f1 = lie_f1 = macro_f1 = 0
    
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, truth_f1, lie_f1, macro_f1, cm


def evaluate(model, dataloader, device, class_weights, truth_focal_weight=4.0):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    loss_fn = FocalWeightedLoss(class_weights, truth_focal_weight)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_data = prepare_batch_for_model(batch, device)
            outputs = model(
                input_ids=batch_data['input_ids'], 
                attention_mask=batch_data['attention_mask'],
                context_input_ids=batch_data['context_input_ids'],
                context_attention_mask=batch_data['context_attention_mask'],
                game_scores=batch_data['scores'],
                batch_adj_matrix=batch_data['batch_adj_matrix'],
                conceptnet_features=batch_data['conceptnet_features']
            )
            loss = loss_fn(outputs, batch_data['labels'])
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            all_labels.extend(batch_data['labels'].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    try:
        truth_f1 = f1_score(all_labels, all_preds, pos_label=0, average='binary', zero_division=0)
        lie_f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary', zero_division=0)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        print(classification_report(all_labels, all_preds, target_names=['Truth', 'Lie'], digits=4, zero_division=0))
    except Exception:
        truth_f1 = lie_f1 = macro_f1 = 0
    
    cm = confusion_matrix(all_labels, all_preds)
    return avg_loss, truth_f1, lie_f1, macro_f1, cm


def plot_confusion_matrix(cm, epoch, split='val'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Truth', 'Lie'], yticklabels=['Truth', 'Lie'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {split.capitalize()} (Epoch {epoch+1})')
    plt.savefig(f'confusion_matrix_{split}_epoch_{epoch+1}.png')
    plt.close()


def plot_metrics(train_metric, val_metric, metric_name):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_metric) + 1)
    plt.plot(epochs, train_metric, 'b-', label=f'Train {metric_name}')
    plt.plot(epochs, val_metric, 'r-', label=f'Val {metric_name}')
    plt.title(f'{metric_name} over Training')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{metric_name.lower()}_plot.png')
    plt.close()