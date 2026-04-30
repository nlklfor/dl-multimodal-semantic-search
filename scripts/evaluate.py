"""
evaluate.py

Standalone evaluation script. Loads a trained checkpoint and computes
Recall@1, R@5, R@10 on the Flickr30k test split (1,000 images, 5,000 captions).

Usage:
    python scripts/evaluate.py --checkpoint experiments/checkpoints/checkpoint_epoch20_learnable.pt
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CACHED_DIR, CAPTIONS_FILE, SPLITS_DIR, BATCH_SIZE, NUM_WORKERS
)
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder
from src.dataset.flickr_dataset import FlickrDataset
from src.evaluation.recall_at_k import evaluate_recall, print_results


def load_captions(captions_file):
    df = pd.read_csv(captions_file, sep='|')
    df.columns = df.columns.str.strip()
    df['comment'] = df['comment'].astype(str).str.strip()
    captions = {}
    for _, row in df.iterrows():
        img_id = row['image_name']
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(row['comment'])
    return captions


def load_split_ids(split):
    path = os.path.join(SPLITS_DIR, f"{split}_ids.txt")
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


@torch.no_grad()
def build_embeddings(dataset, vision_enc, text_enc, device):
    """Encode all images and captions in the dataset."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS)

    all_image_embs = []
    all_text_embs  = []

    for image_feats, captions in tqdm(loader, desc="Encoding"):
        image_feats = image_feats.to(device)
        img_emb = vision_enc(image_feats)       # (B, 256)
        txt_emb = text_enc(captions, device)    # (B, 256)
        all_image_embs.append(img_emb.cpu())
        all_text_embs.append(txt_emb.cpu())

    return torch.cat(all_image_embs), torch.cat(all_text_embs)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading checkpoint: {args.checkpoint}")
    vision_enc = VisionEncoder().to(device)
    text_enc   = TextEncoder().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vision_enc.load_state_dict(ckpt['vision_encoder'])
    text_enc.load_state_dict(ckpt['text_encoder'])
    print(f"Loaded from epoch {ckpt['epoch']}")

    vision_enc.eval()
    text_enc.eval()

    # Load test data
    captions = load_captions(CAPTIONS_FILE)
    test_ids  = load_split_ids('test')

    test_dataset = FlickrDataset(
        features_path = os.path.join(CACHED_DIR, "test_image_features.pt"),
        captions_dict = captions,
        image_ids     = test_ids,
        split         = 'test',
    )

    # Build embeddings
    print("\nBuilding embeddings for test set...")
    image_embs, text_embs = build_embeddings(test_dataset, vision_enc, text_enc, device)
    print(f"Image embeddings: {image_embs.shape}")
    print(f"Text embeddings:  {text_embs.shape}")

    # Evaluate
    print("\nComputing Recall@K...")
    results = evaluate_recall(image_embs, text_embs)
    print_results(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    args = parser.parse_args()
    main(args)
