import torch
import numpy as np
import random

# Data paths
TRAIN_PATH = "data/train.jsonl"
VAL_PATH = "data/validation.jsonl"
TEST_PATH = "data/test.jsonl"

# Model and training parameters
TRANSFORMER_MODEL = "roberta-base"  # Using RoBERTa base
BATCH_SIZE = 32
EPOCHS = 5
LR = 5e-6
USE_GAME_SCORES = True
OVERSAMPLING_FACTOR = 30  # Oversampling factor (tune if necessary)
TRUTH_FOCAL_WEIGHT = 4.0  # Additional weight for truth class during loss calculation
EARLY_STOPPING_PATIENCE = 5
GRADIENT_ACCUMULATION_STEPS = 2

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"