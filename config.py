"""
config.py — Central configuration for all hyperparameters and paths.
All training scripts and notebooks import from here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR         = "data/"
RAW_DIR          = os.path.join(DATA_DIR, "raw/")
CACHED_DIR       = os.path.join(DATA_DIR, "cached/")
SPLITS_DIR       = os.path.join(DATA_DIR, "splits/")

IMAGES_DIR       = os.path.join(RAW_DIR, "flickr30k-images/")
CAPTIONS_FILE    = os.path.join(RAW_DIR, "results.csv")

TRAIN_FEATURES   = os.path.join(CACHED_DIR, "train_image_features.pt")
VAL_FEATURES     = os.path.join(CACHED_DIR, "val_image_features.pt")
TEST_FEATURES    = os.path.join(CACHED_DIR, "test_image_features.pt")

CHECKPOINTS_DIR  = "experiments/checkpoints/"

# ─── Model Architecture ───────────────────────────────────────────────────────

EMBEDDING_DIM    = 256        # Shared embedding space dimensionality
HIDDEN_DIM       = 512        # Projection MLP hidden layer size
VISION_INPUT_DIM = 2048       # ResNet-50 pool5 output
TEXT_INPUT_DIM   = 768        # DistilBERT CLS token output
DROPOUT          = 0.1

# ─── Training ────────────────────────────────────────────────────────────────

BATCH_SIZE       = 128
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
NUM_EPOCHS       = 20
SEED             = 42

# Temperature for InfoNCE loss
# Set to None to use learnable temperature (recommended)
# Set to a float (e.g., 0.07) for fixed temperature ablation runs
TEMPERATURE      = None       # None = learnable, initialized at log(1/0.07)
INIT_TEMPERATURE = 0.07       # Initial value when using learnable temperature

# ─── DistilBERT ──────────────────────────────────────────────────────────────

TEXT_MODEL_NAME  = "distilbert-base-uncased"
MAX_TOKEN_LENGTH = 64         # Max caption token length (Flickr30k captions are short)

# ─── Evaluation ──────────────────────────────────────────────────────────────

RECALL_AT_K      = [1, 5, 10]

# ─── Hardware ────────────────────────────────────────────────────────────────

NUM_WORKERS      = 2          # DataLoader workers (keep low on Colab)
PIN_MEMORY       = True       # Speed up CPU→GPU transfers

# ─── Image Transforms ────────────────────────────────────────────────────────

IMAGE_SIZE       = 224
# ImageNet normalization stats (used by pretrained ResNet-50)
NORMALIZE_MEAN   = [0.485, 0.456, 0.406]
NORMALIZE_STD    = [0.229, 0.224, 0.225]
