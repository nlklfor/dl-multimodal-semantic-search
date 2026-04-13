"""
precompute_features.py

Run ONCE to extract and cache ResNet-50 features for all Flickr30k images.
Since the vision backbone is frozen during training, features never change —
caching them reduces epoch time from ~40 min → ~2 min on Colab/Kaggle T4.

Usage:
    python scripts/precompute_features.py --split train
    python scripts/precompute_features.py --split val
    python scripts/precompute_features.py --split test
    python scripts/precompute_features.py --split all   (default)
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMAGES_DIR, CAPTIONS_FILE, CACHED_DIR, SPLITS_DIR,
    IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD, NUM_WORKERS
)


# ─── Dataset for raw images ───────────────────────────────────────────────────

class RawImageDataset(Dataset):
    """Loads raw images for feature extraction. No captions needed."""

    def __init__(self, image_ids, transform):
        self.image_ids = image_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(IMAGES_DIR, image_id)
        image = Image.open(path).convert("RGB")
        return self.transform(image), image_id


# ─── ResNet-50 backbone (no final FC) ─────────────────────────────────────────

def build_backbone():
    """Returns frozen ResNet-50 with FC layer removed."""
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


# ─── Image transforms ─────────────────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


# ─── Load split IDs ───────────────────────────────────────────────────────────

def load_split_ids(split):
    path = os.path.join(SPLITS_DIR, f"{split}_ids.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Split file not found: {path}\n"
            f"Run scripts/create_splits.py first, or download Karpathy splits."
        )
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


# ─── Main extraction loop ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(split, device):
    print(f"\n{'='*50}")
    print(f"Extracting features for split: {split}")
    print(f"{'='*50}")

    image_ids = load_split_ids(split)
    print(f"Found {len(image_ids)} images in {split} split")

    dataset    = RawImageDataset(image_ids, get_transform())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    backbone = build_backbone().to(device)

    all_features = []
    all_ids      = []

    for images, ids in tqdm(dataloader, desc=f"Extracting {split}"):
        images = images.to(device)
        feats  = backbone(images)                    # (B, 2048, 1, 1)
        feats  = feats.squeeze(-1).squeeze(-1)       # (B, 2048)
        all_features.append(feats.cpu())
        all_ids.extend(ids)

    features_tensor = torch.cat(all_features, dim=0)   # (N, 2048)

    # Save features
    os.makedirs(CACHED_DIR, exist_ok=True)
    feat_path = os.path.join(CACHED_DIR, f"{split}_image_features.pt")
    torch.save(features_tensor, feat_path)
    print(f"Saved features: {feat_path}  shape={features_tensor.shape}")

    # Save ID ordering map (filename → index)
    id_map = {img_id: idx for idx, img_id in enumerate(all_ids)}
    map_path = os.path.join(CACHED_DIR, f"{split}_id_to_idx.json")
    with open(map_path, "w") as f:
        json.dump(id_map, f)
    print(f"Saved ID map:   {map_path}  ({len(id_map)} entries)")

    return features_tensor


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split to process")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        extract_features(split, device)

    print("\n✅ Feature extraction complete. Training can now use cached features.")
    print(f"   Cache directory: {CACHED_DIR}")
