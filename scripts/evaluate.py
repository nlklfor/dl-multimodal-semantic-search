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
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CACHED_DIR, CAPTIONS_FILE, SPLITS_DIR, BATCH_SIZE
)
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder
from src.evaluation.recall_at_k import evaluate_recall, print_results


N_CAPTIONS_PER_IMAGE = 5   # Flickr30k canonical setup


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
def build_eval_embeddings(image_features, image_ids, captions_dict,
                          vision_enc, text_enc, device, batch_size=BATCH_SIZE):
    """
    Build embeddings in the canonical Flickr30k retrieval-eval layout:

        image_embs  shape (N_images, D)        — one row per image
        text_embs   shape (N_images * 5, D)    — 5 captions per image, flattened
                                                 in the order
                                                 [img0_cap0..cap4, img1_cap0..cap4, …]

    This matches what `evaluate_recall` in src/evaluation/recall_at_k.py expects
    (it derives correct_image_idx = caption_idx // 5).
    """
    # ── Images ───────────────────────────────────────────────────────────────
    image_embs = []
    for i in tqdm(range(0, len(image_ids), batch_size), desc="Encoding images"):
        feats = image_features[i:i + batch_size].to(device)
        image_embs.append(vision_enc(feats).cpu())
    image_embs = torch.cat(image_embs, dim=0)

    # ── Captions, flattened in canonical order ───────────────────────────────
    flat_captions = []
    for img_id in image_ids:
        caps = captions_dict[img_id]
        assert len(caps) == N_CAPTIONS_PER_IMAGE, \
            f"{img_id} has {len(caps)} captions, expected {N_CAPTIONS_PER_IMAGE}"
        flat_captions.extend(caps[:N_CAPTIONS_PER_IMAGE])

    text_embs = []
    for i in tqdm(range(0, len(flat_captions), batch_size), desc="Encoding captions"):
        batch = flat_captions[i:i + batch_size]
        text_embs.append(text_enc(batch, device).cpu())
    text_embs = torch.cat(text_embs, dim=0)

    return image_embs, text_embs


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
    captions       = load_captions(CAPTIONS_FILE)
    test_ids       = load_split_ids('test')
    test_features  = torch.load(os.path.join(CACHED_DIR, "test_image_features.pt"),
                                 weights_only=True)

    # Build embeddings in canonical Flickr30k eval layout (1k images × 5 captions)
    print("\nBuilding embeddings for test set...")
    image_embs, text_embs = build_eval_embeddings(
        test_features, test_ids, captions,
        vision_enc, text_enc, device,
    )
    print(f"Image embeddings: {tuple(image_embs.shape)}  (expected (1000, 256))")
    print(f"Text embeddings:  {tuple(text_embs.shape)}   (expected (5000, 256))")

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
