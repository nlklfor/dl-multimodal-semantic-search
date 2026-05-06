"""
make_report_figures.py

Generates the plots and diagrams used in the project report. Outputs are
written to `docs/figures/` and tracked in git.

Run after a training run finishes:

    python scripts/make_report_figures.py \\
        --history    experiments/checkpoints/training_history.json \\
        --checkpoint experiments/checkpoints/checkpoint_epoch50_learnable.pt

Static figures (always produced):
  - fig_loss_curves.png         train + val loss across epochs
  - fig_recall_comparison.png   R@K bar chart, 20-epoch vs 50-epoch
  - fig_literature_comparison.png   T→I R@1 vs VSE++ / CLIP variants
  - fig_overfitting.png         train-val gap over training

Checkpoint-dependent (skipped if --checkpoint is missing):
  - fig_umap.png                UMAP of the test set embeddings
  - fig_sample_retrievals.png   top-5 retrievals for 4 demo queries
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import IMAGES_DIR, CACHED_DIR, CAPTIONS_FILE, SPLITS_DIR  # noqa: E402

FIG_DIR = "docs/figures"

# Hard-coded run results — kept in code so the figures regenerate
# deterministically without parsing eval logs.
RESULTS = {
    "20_epoch": {
        "T→I": {"R@1": 16.08, "R@5": 41.56, "R@10": 54.88},
        "I→T": {"R@1": 20.50, "R@5": 46.30, "R@10": 61.30},
    },
    "50_epoch": {
        "T→I": {"R@1": 17.48, "R@5": 44.44, "R@10": 57.00},
        "I→T": {"R@1": 23.70, "R@5": 49.40, "R@10": 60.90},
    },
}

LITERATURE = {
    "Random (1k)":              0.1,
    "Ours (20 ep, 1.7M params)": 16.08,
    "Ours (50 ep, 1.7M params)": 17.48,
    "VSE++ (10M params)":       39.6,
    "CLIP zero-shot (151M)":    65.1,
    "CLIP fine-tuned (151M)":   88.0,
}


# ─── Static plots (data hard-coded above) ────────────────────────────────────

def fig_loss_curves(history_path, out_path):
    h = json.load(open(history_path))
    epochs = np.arange(1, len(h["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, h["train_loss"], label="train", linewidth=2)
    ax.plot(epochs, h["val_loss"],   label="val",   linewidth=2)
    ax.axhline(np.log(128), color="grey", linestyle=":", linewidth=1,
               label="log(128) — random batch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("symmetric InfoNCE loss")
    ax.set_title(f"Training curves ({len(epochs)}-epoch run, learnable τ)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_overfitting(history_path, out_path):
    h = json.load(open(history_path))
    epochs   = np.arange(1, len(h["train_loss"]) + 1)
    train    = np.array(h["train_loss"])
    val      = np.array(h["val_loss"])
    gap      = val - train

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 2]})

    ax1.plot(epochs, train, label="train", linewidth=2)
    ax1.plot(epochs, val,   label="val",   linewidth=2)
    ax1.set_ylabel("loss")
    ax1.set_title("Train vs val — overfitting analysis")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.fill_between(epochs, 0, gap, color="tab:red", alpha=0.4,
                     label="val − train")
    ax2.plot(epochs, gap, color="tab:red", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val − train")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_recall_comparison(out_path):
    metrics = ["R@1", "R@5", "R@10"]
    directions = ["T→I", "I→T"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    width = 0.35
    x = np.arange(len(metrics))

    for ax, direction in zip(axes, directions):
        vals_20 = [RESULTS["20_epoch"][direction][m] for m in metrics]
        vals_50 = [RESULTS["50_epoch"][direction][m] for m in metrics]
        b1 = ax.bar(x - width / 2, vals_20, width, label="20 epochs", color="#7BAFD4")
        b2 = ax.bar(x + width / 2, vals_50, width, label="50 epochs", color="#1F4E79")
        for b in [*b1, *b2]:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7,
                    f"{b.get_height():.1f}", ha="center", fontsize=9)
        ax.set_title(f"{direction} retrieval")
        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylim(0, 70)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()

    axes[0].set_ylabel("Recall (%)")
    fig.suptitle("Recall@K — 20-epoch baseline vs 50-epoch run", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_literature_comparison(out_path):
    labels = list(LITERATURE.keys())
    values = list(LITERATURE.values())
    colors = ["#9E9E9E", "#7BAFD4", "#1F4E79", "#E07B00", "#5C8A2D", "#2D5C2D"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(v + 1, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Text → Image Recall@1 (%)")
    ax.set_title("Where our model sits — Flickr30k T→I R@1")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─── Checkpoint-dependent plots ──────────────────────────────────────────────

@torch.no_grad()
def _build_test_embeddings(checkpoint_path, device):
    """Encode all 1000 test images and 5000 captions through the trained encoders."""
    import pandas as pd
    from src.encoders.vision_encoder import VisionEncoder
    from src.encoders.text_encoder import TextEncoder

    vision = VisionEncoder().to(device)
    text   = TextEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vision.load_state_dict(ckpt["vision_encoder"])
    text.load_state_dict(ckpt["text_encoder"])
    vision.eval(); text.eval()

    feats = torch.load(os.path.join(CACHED_DIR, "test_image_features.pt"),
                       weights_only=True)
    with open(os.path.join(SPLITS_DIR, "test_ids.txt")) as f:
        ids = [l.strip() for l in f if l.strip()]

    df = pd.read_csv(CAPTIONS_FILE, sep="|")
    df.columns = df.columns.str.strip()
    df["comment"] = df["comment"].astype(str).str.strip()
    caps = {}
    for _, r in df.iterrows():
        caps.setdefault(r["image_name"], []).append(r["comment"])

    img_emb = vision(feats.to(device)).cpu()
    flat = [c for img_id in ids for c in caps[img_id][:5]]
    txt_emb = []
    for i in range(0, len(flat), 256):
        txt_emb.append(text(flat[i:i + 256], device).cpu())
    txt_emb = torch.cat(txt_emb)

    return img_emb, txt_emb, ids


def fig_umap(checkpoint_path, out_path):
    import umap

    device = torch.device("cpu")
    img_emb, txt_emb, _ = _build_test_embeddings(checkpoint_path, device)

    # Sample 300 matched pairs to keep plot readable
    rng = np.random.default_rng(42)
    n = 300
    idx = rng.choice(img_emb.shape[0], n, replace=False)
    img_sample = img_emb[idx].numpy()
    txt_sample = txt_emb[idx * 5].numpy()                # caption 0 of each

    combined = np.concatenate([img_sample, txt_sample], axis=0)
    reducer  = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    points   = reducer.fit_transform(combined)
    img_pts  = points[:n]
    txt_pts  = points[n:]

    fig, ax = plt.subplots(figsize=(8, 7))
    # Connect matched pairs with thin grey lines
    for i in range(n):
        ax.plot([img_pts[i, 0], txt_pts[i, 0]],
                [img_pts[i, 1], txt_pts[i, 1]],
                color="lightgrey", linewidth=0.4, alpha=0.5, zorder=1)
    ax.scatter(img_pts[:, 0], img_pts[:, 1], s=18, c="#1F4E79",
               label="image", zorder=2, alpha=0.8)
    ax.scatter(txt_pts[:, 0], txt_pts[:, 1], s=18, c="#E07B00",
               label="text",  zorder=2, alpha=0.8, marker="^")
    ax.set_title(f"Shared embedding space (UMAP, n={n} matched pairs)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


def fig_sample_retrievals(checkpoint_path, gallery_path, gallery_ids_path, out_path):
    if not (os.path.exists(gallery_path) and os.path.exists(gallery_ids_path)):
        print(f"  ✗ skip sample retrievals — missing {gallery_path}")
        return
    if not os.path.exists(IMAGES_DIR):
        print(f"  ✗ skip sample retrievals — missing {IMAGES_DIR}")
        return

    device = torch.device("cpu")
    from src.encoders.text_encoder import TextEncoder
    text = TextEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    text.load_state_dict(ckpt["text_encoder"])
    text.eval()

    gallery = torch.load(gallery_path, weights_only=True)
    ids     = json.load(open(gallery_ids_path))

    queries = [
        "a dog playing in the snow",
        "two people sharing an umbrella in the rain",
        "a child blowing out birthday candles",
        "cyclists racing on a mountain road",
    ]

    fig, axes = plt.subplots(len(queries), 5, figsize=(13, 2.5 * len(queries)))
    with torch.no_grad():
        for r, q in enumerate(queries):
            q_emb = text([q], device).cpu()
            scores = (q_emb @ gallery.T).squeeze(0)
            top5 = scores.topk(5).indices.tolist()
            for c, idx in enumerate(top5):
                ax = axes[r, c]
                img_path = os.path.join(IMAGES_DIR, ids[idx])
                if os.path.exists(img_path):
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                ax.set_title(f"{scores[idx].item():.3f}", fontsize=9)
                ax.axis("off")
            axes[r, 0].set_ylabel(f"{q!r}", rotation=0, ha="right",
                                   va="center", fontsize=9)
    fig.suptitle("Top-5 retrievals for sample queries", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_path}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history",      default="experiments/checkpoints/training_history.json")
    parser.add_argument("--checkpoint",   default="experiments/checkpoints/checkpoint_epoch50_learnable.pt")
    parser.add_argument("--gallery",      default=os.path.join(CACHED_DIR, "gallery_embs.pt"))
    parser.add_argument("--gallery-ids",  default=os.path.join(CACHED_DIR, "gallery_ids.json"))
    parser.add_argument("--out-dir",      default=FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Writing figures to {args.out_dir}/\n")

    # Static plots
    if os.path.exists(args.history):
        fig_loss_curves(args.history,  os.path.join(args.out_dir, "fig_loss_curves.png"))
        fig_overfitting(args.history,  os.path.join(args.out_dir, "fig_overfitting.png"))
    else:
        print(f"  ✗ skip loss/overfitting — missing {args.history}")
    fig_recall_comparison(   os.path.join(args.out_dir, "fig_recall_comparison.png"))
    fig_literature_comparison(os.path.join(args.out_dir, "fig_literature_comparison.png"))

    # Dynamic plots
    if os.path.exists(args.checkpoint):
        fig_umap(args.checkpoint, os.path.join(args.out_dir, "fig_umap.png"))
        fig_sample_retrievals(
            args.checkpoint, args.gallery, args.gallery_ids,
            os.path.join(args.out_dir, "fig_sample_retrievals.png"),
        )
    else:
        print(f"  ✗ skip UMAP / sample retrievals — missing {args.checkpoint}")

    print("\nDone.")


if __name__ == "__main__":
    main()
