"""
build_gallery_index.py

Run ONCE after training is complete. Encodes the cached ResNet-50 features for
all 31,014 Flickr30k images (train + val + test) through the trained
VisionEncoder projection head and saves a single gallery index for the
Gradio demo.

Output:
    data/cached/gallery_embs.pt    — torch.float32 tensor, shape (31014, 256)
    data/cached/gallery_ids.json   — list of filenames; index i ↔ row i

Usage:
    python scripts/build_gallery_index.py \\
        --checkpoint experiments/checkpoints/checkpoint_epoch20_learnable.pt
"""

import os
import json
import argparse
import torch
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CACHED_DIR, SPLITS_DIR
from src.encoders.vision_encoder import VisionEncoder


GALLERY_EMBS_PATH = os.path.join(CACHED_DIR, "gallery_embs.pt")
GALLERY_IDS_PATH  = os.path.join(CACHED_DIR, "gallery_ids.json")
SPLITS            = ["train", "val", "test"]


def load_split(split):
    feats_path = os.path.join(CACHED_DIR, f"{split}_image_features.pt")
    ids_path   = os.path.join(SPLITS_DIR,  f"{split}_ids.txt")

    feats = torch.load(feats_path, weights_only=True)        # (N, 2048)
    with open(ids_path) as f:
        ids = [line.strip() for line in f if line.strip()]

    assert len(ids) == feats.shape[0], (
        f"{split}: id-count {len(ids)} != feature-rows {feats.shape[0]} — "
        f"split files and cached features are out of sync."
    )
    return feats, ids


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load trained vision encoder ──────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    vision_enc = VisionEncoder().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vision_enc.load_state_dict(ckpt["vision_encoder"])
    vision_enc.eval()
    print(f"  Loaded VisionEncoder from epoch {ckpt['epoch']}")

    # ── Concatenate cached features from all splits ──────────────────────────
    print("\nLoading cached ResNet-50 features:")
    all_feats = []
    all_ids   = []
    for split in SPLITS:
        feats, ids = load_split(split)
        print(f"  {split:5s}: {feats.shape[0]:>5,} images")
        all_feats.append(feats)
        all_ids.extend(ids)

    feats = torch.cat(all_feats, dim=0)                   # (31014, 2048)
    print(f"  total: {feats.shape[0]:,} images, {feats.shape[1]}-d cached features")

    # ── Project to 256-d shared embedding space ──────────────────────────────
    print("\nEncoding through trained projection head...")
    embs_chunks = []
    for i in tqdm(range(0, len(feats), args.batch_size), desc="Encoding"):
        batch = feats[i:i + args.batch_size].to(device)
        embs_chunks.append(vision_enc(batch).cpu())

    gallery_embs = torch.cat(embs_chunks, dim=0)          # (31014, 256)
    norms        = gallery_embs.norm(dim=-1)
    print(f"\nGallery embeddings: shape={tuple(gallery_embs.shape)}  dtype={gallery_embs.dtype}")
    print(f"  L2 norms:  mean={norms.mean():.4f}  min={norms.min():.4f}  max={norms.max():.4f}  (expect ≈1)")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        "Gallery embeddings are not L2-normalized — VisionEncoder output is broken."

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(CACHED_DIR, exist_ok=True)
    torch.save(gallery_embs, GALLERY_EMBS_PATH)
    with open(GALLERY_IDS_PATH, "w") as f:
        json.dump(all_ids, f)

    print(f"\nSaved {GALLERY_EMBS_PATH}  ({os.path.getsize(GALLERY_EMBS_PATH)/1e6:.1f} MB)")
    print(f"Saved {GALLERY_IDS_PATH}   ({os.path.getsize(GALLERY_IDS_PATH)/1e3:.0f} KB)")
    print("\nGallery index ready. Demo can now load it for instant startup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Encoding batch size (default 512)")
    args = parser.parse_args()
    main(args)
