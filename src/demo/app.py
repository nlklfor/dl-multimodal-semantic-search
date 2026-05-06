"""
app.py

Interactive Gradio demo for the Multimodal Semantic Search Engine.
Type a natural language query and retrieve the top-5 most relevant images
from the Flickr30k gallery using vector similarity search.

Usage:
    python src/demo/app.py --checkpoint experiments/checkpoints/checkpoint_epoch20_learnable.pt
    python src/demo/app.py --checkpoint <path> --share          # public Gradio link
    python src/demo/app.py --checkpoint <path> --split test     # 1k ablation gallery
"""

import os
import json
import argparse
import torch
import gradio as gr
from PIL import Image
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import (
    IMAGES_DIR, CACHED_DIR, CAPTIONS_FILE, SPLITS_DIR
)
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder


GALLERY_EMBS_PATH = os.path.join(CACHED_DIR, "gallery_embs.pt")
GALLERY_IDS_PATH  = os.path.join(CACHED_DIR, "gallery_ids.json")


# ─── Build image gallery index ───────────────────────────────────────────────

def load_precomputed_gallery():
    """Fast path: load the 31k gallery built by scripts/build_gallery_index.py."""
    print(f"Loading pre-computed gallery: {GALLERY_EMBS_PATH}")
    gallery_embs = torch.load(GALLERY_EMBS_PATH, weights_only=True)
    with open(GALLERY_IDS_PATH) as f:
        image_ids = json.load(f)
    assert len(image_ids) == gallery_embs.shape[0], \
        "gallery_embs.pt and gallery_ids.json are out of sync."
    print(f"Gallery loaded: {gallery_embs.shape[0]:,} images, dim={gallery_embs.shape[1]}")
    return gallery_embs, image_ids


def build_gallery_index(vision_enc, device, split='test'):
    """Fallback path: encode a single split on-the-fly. Used for ablation."""
    print(f"Building gallery index for split: {split}...")

    features_path = os.path.join(CACHED_DIR, f"{split}_image_features.pt")
    features = torch.load(features_path).to(device)     # (N, 2048)

    # Load corresponding image IDs
    splits_path = os.path.join(SPLITS_DIR, f"{split}_ids.txt")
    with open(splits_path) as f:
        image_ids = [line.strip() for line in f.readlines()]

    # Encode in batches
    vision_enc.eval()
    all_embs = []
    batch_size = 256

    with torch.no_grad():
        for i in tqdm(range(0, len(features), batch_size), desc="Encoding gallery"):
            batch = features[i:i + batch_size]
            embs  = vision_enc(batch)            # (B, 256)
            all_embs.append(embs.cpu())

    gallery_embs = torch.cat(all_embs, dim=0)    # (N, 256)
    print(f"Gallery index built: {gallery_embs.shape[0]} images")

    return gallery_embs, image_ids


# ─── Search function ──────────────────────────────────────────────────────────

def search(query, gallery_embs, image_ids, text_enc, device, top_k=5):
    """Encode query text and retrieve top-k images by cosine similarity."""
    text_enc.eval()

    with torch.no_grad():
        query_emb = text_enc([query], device)              # (1, 256)

    scores  = (query_emb.cpu() @ gallery_embs.T).squeeze(0)   # (N,)
    top_idx = scores.topk(top_k).indices.tolist()

    results = []
    for idx in top_idx:
        img_id = image_ids[idx]
        img_path = os.path.join(IMAGES_DIR, img_id)
        score = scores[idx].item()

        try:
            img = Image.open(img_path).convert("RGB")
            results.append((img, f"Score: {score:.3f}"))
        except FileNotFoundError:
            print(f"Warning: image not found: {img_path}")

    return results


# ─── Gradio interface ─────────────────────────────────────────────────────────

def build_interface(vision_enc, text_enc, gallery_embs, image_ids, device):

    def query_fn(text_query):
        if not text_query.strip():
            return []
        results = search(text_query, gallery_embs, image_ids, text_enc, device)
        return results

    with gr.Blocks(title="Multimodal Semantic Search", theme=gr.themes.Soft()) as demo:

        gr.Markdown("""
        # 🔍 Multimodal Semantic Image Search
        **Dual-Encoder Contrastive Learning — University of Bern, MSc Deep Learning**

        Type any natural language description to search the image gallery
        using learned visual-textual embeddings.
        """)

        with gr.Row():
            with gr.Column(scale=4):
                query_box = gr.Textbox(
                    label="Search Query",
                    placeholder='Try: "a dog running on the beach" or "two people sharing an umbrella"',
                    lines=2,
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

        gallery = gr.Gallery(
            label="Top 5 Results",
            columns=5,
            height=300,
            object_fit="cover",
        )

        gr.Examples(
            examples=[
                ["a dog playing in the snow"],
                ["a child blowing out birthday candles"],
                ["two people sharing an umbrella in the rain"],
                ["a street performer entertaining a crowd"],
                ["cyclists racing on a mountain road"],
                ["a woman reading a book outdoors"],
            ],
            inputs=query_box,
        )

        gr.Markdown("""
        ---
        **How it works:** Your query is encoded by a frozen DistilBERT + projection head
        into a 256-dim embedding. This is compared against pre-computed embeddings for all
        gallery images (ResNet-50 + projection head) using cosine similarity.
        Results are the top-5 nearest neighbours in the shared embedding space.
        """)

        search_btn.click(fn=query_fn, inputs=query_box, outputs=gallery)
        query_box.submit(fn=query_fn, inputs=query_box, outputs=gallery)

    return demo


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"],
                        help="Override: encode a single split on-the-fly instead of "
                             "loading the pre-computed 31k gallery.")
    parser.add_argument("--share", action="store_true",
                        help="Create public Gradio sharing link")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load trained models
    print(f"\nLoading checkpoint: {args.checkpoint}")
    vision_enc = VisionEncoder().to(device)
    text_enc   = TextEncoder().to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    vision_enc.load_state_dict(ckpt['vision_encoder'])
    text_enc.load_state_dict(ckpt['text_encoder'])
    print(f"Loaded from epoch {ckpt['epoch']}")

    # Gallery: prefer pre-computed 31k index when available; fall back to per-split
    # encoding when the user passes --split or the pre-computed file is missing.
    use_precomputed = (args.split is None and os.path.exists(GALLERY_EMBS_PATH))
    if use_precomputed:
        gallery_embs, image_ids = load_precomputed_gallery()
    else:
        split = args.split or "test"
        gallery_embs, image_ids = build_gallery_index(vision_enc, device, split=split)

    # Launch demo
    demo = build_interface(vision_enc, text_enc, gallery_embs, image_ids, device)
    demo.launch(share=args.share)
