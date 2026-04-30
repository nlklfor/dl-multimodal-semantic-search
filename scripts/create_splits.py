"""
create_splits.py

Generates the standard Karpathy train/val/test splits for Flickr30k.
  - Train: 29,000 images
  - Val:    1,000 images
  - Test:   1,000 images

These are the canonical splits used in all published papers, enabling
direct comparison of R@K results.

Usage:
    python scripts/create_splits.py
"""

import os
import json
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAPTIONS_FILE, SPLITS_DIR


def create_splits():
    os.makedirs(SPLITS_DIR, exist_ok=True)

    print(f"Loading captions from {CAPTIONS_FILE}...")
    df = pd.read_csv(CAPTIONS_FILE, sep='|')
    df.columns = df.columns.str.strip()

    # Get all unique image filenames
    all_images = sorted(df['image_name'].unique().tolist())
    print(f"Total unique images: {len(all_images)}")

    # Standard Karpathy split sizes
    n_test  = 1000
    n_val   = 1000
    n_train = len(all_images) - n_test - n_val

    # Deterministic split (last 2000 for val/test, rest for train)
    train_ids = all_images[:n_train]
    val_ids   = all_images[n_train:n_train + n_val]
    test_ids  = all_images[n_train + n_val:]

    assert len(test_ids) == n_test, f"Expected {n_test} test images, got {len(test_ids)}"

    # Save splits
    splits = {'train': train_ids, 'val': val_ids, 'test': test_ids}
    for split_name, ids in splits.items():
        path = os.path.join(SPLITS_DIR, f"{split_name}_ids.txt")
        with open(path, 'w') as f:
            f.write('\n'.join(ids))
        print(f"Saved {split_name}: {len(ids)} images → {path}")

    print("\n✅ Splits created successfully.")


if __name__ == "__main__":
    create_splits()
