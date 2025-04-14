import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict

class EnhancedDeceptionDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=128, use_game_scores=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_game_scores = use_game_scores
        self.texts = []
        self.labels = []
        self.scores = []
        self.conversation_ids = []  # Track conversation id
        self.message_positions = []  # Message order in conversation
        self.prior_context = []  # Prior context (concatenated previous messages)
        
        with open(path, 'r', encoding='utf-8') as f:
            conv_id = 0
            for line in f:
                data = json.loads(line.strip())
                messages = data.get('messages', [])
                labels = data.get('sender_labels', [])
                game_scores = data.get('game_score_delta', None) if use_game_scores else None
                if game_scores is None:
                    game_scores = [0] * len(messages)
                
                for pos, (msg, label, score) in enumerate(zip(messages, labels, game_scores)):
                    if label in [True, False, "true", "false", "True", "False"]:
                        # Convert string labels to boolean if needed
                        if isinstance(label, str):
                            is_lie = label.lower() == "true"
                        else:
                            is_lie = label
                        # Convention: 1 = lie, 0 = truth
                        self.texts.append(msg)
                        self.labels.append(1 if is_lie else 0)
                        self.scores.append(float(score))
                        self.conversation_ids.append(conv_id)
                        self.message_positions.append(pos)
                        # Build prior context: use up to two previous messages
                        context = ""
                        if pos > 0:
                            context_msgs = messages[max(0, pos-2):pos]
                            context = " [SEP] ".join(context_msgs)
                        self.prior_context.append(context)
                conv_id += 1
        
        self.class_counts = Counter(self.labels)
        total = len(self.labels)
        self.truth_indices = [i for i, label in enumerate(self.labels) if label == 0]
        self.lie_indices = [i for i, label in enumerate(self.labels) if label == 1]
        
        # Group messages by conversation for potential graph construction
        self.conv_to_msgs = defaultdict(list)
        for i, cid in enumerate(self.conversation_ids):
            self.conv_to_msgs[cid].append(i)
        
        print(f"Dataset loaded from {path}")
        print(f"Total messages: {total}")
        for label, count in sorted(self.class_counts.items()):
            label_name = "Truth" if label == 0 else "Lie"
            print(f"{label_name}: {count} ({count/total*100:.2f}%)")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        score = self.scores[idx]
        conv_id = self.conversation_ids[idx]
        position = self.message_positions[idx]
        context = self.prior_context[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if context:
            context_encoding = self.tokenizer(
                context,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        else:
            context_encoding = {
                'input_ids': torch.zeros((1, self.max_len), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.max_len), dtype=torch.long)
            }
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'context_input_ids': context_encoding['input_ids'].flatten(),
            'context_attention_mask': context_encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float),
            'conv_id': conv_id,
            'position': position,
            'relative_positions': []  # Reserved for legacy
        }


class EnhancedBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, oversample_factor=15):
        self.dataset = dataset
        self.oversample_factor = oversample_factor
        self.truth_indices = dataset.truth_indices * oversample_factor  # Oversample truths
        self.lie_indices = dataset.lie_indices
        
        min_samples = max(1000, len(self.truth_indices))
        target_size = min(len(self.truth_indices), len(self.lie_indices))
        target_size = max(target_size, min_samples)
        
        if len(self.truth_indices) < target_size:
            self.truth_indices = self.truth_indices * (target_size // len(self.truth_indices) + 1)
        if len(self.lie_indices) < target_size:
            self.lie_indices = self.lie_indices * (target_size // len(self.lie_indices) + 1)
        
        self.truth_indices = random.sample(self.truth_indices, target_size)
        self.lie_indices = random.sample(self.lie_indices, target_size)
        
        self.indices = self.truth_indices + self.lie_indices
        random.shuffle(self.indices)
    
    def __iter__(self):
        return iter(self.indices)
    
    def __len__(self):
        return len(self.indices)


def custom_collate_fn(batch):
    # Collate all keys except 'relative_positions'
    batch_without_relative = [{k: v for k, v in item.items() if k != 'relative_positions'} for item in batch]
    collated = torch.utils.data.dataloader.default_collate(batch_without_relative)
    collated['relative_positions'] = [item['relative_positions'] for item in batch]
    return collated