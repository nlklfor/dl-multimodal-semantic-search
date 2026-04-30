# Results

## Final Model Performance

Evaluated on the **Flickr30k test split** (1,000 images, 5,000 captions).
Standard Karpathy split. All numbers are Recall@K (higher is better).

### Retrieval Results

| Direction       | R@1   | R@5   | R@10  |
|-----------------|-------|-------|-------|
| Text → Image    | —     | —     | —     |
| Image → Text    | —     | —     | —     |

> *To be filled after training*

---

## Comparison with Literature

For context, here are published R@1 (Text→Image) scores on Flickr30k:

| Model | Training Data | T→I R@1 |
|-------|--------------|---------|
| VSE++ (2018) | Flickr30k | 39.6% |
| CLIP (zero-shot) | 400M pairs | 65.1% |
| CLIP (fine-tuned) | 400M + Flickr30k | 88.0% |
| **Ours** | Flickr30k (29k) | — |

Our model trains only **1.6M parameters** on **29,000 image-caption pairs** with **frozen backbones**. A R@1 score in the 30–45% range would be a strong result under these constraints.

---

## Temperature Ablation Summary

| Temperature | T→I R@1 | T→I R@5 | Notes |
|-------------|---------|---------|-------|
| τ = 0.05    | —       | —       | |
| τ = 0.07    | —       | —       | CLIP default |
| τ = 0.10    | —       | —       | |
| τ = learnable | —     | —       | **Our default** |

---

## Training Curves

> *(Plots to be added after training)*

Loss curves (train vs. val) for the baseline run.

## Embedding Space Visualization

> *(UMAP plot to be added after training)*

UMAP of 500 test pairs (blue = image embeddings, orange = text embeddings,
lines = matched pairs).
