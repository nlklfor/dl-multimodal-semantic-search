# Architecture

## Overview

We implement a **dual-encoder contrastive learning** architecture inspired by OpenAI's CLIP. The system maps images and natural language captions into a shared 256-dimensional embedding space, where semantically matching pairs are geometrically close.

---

## Design Philosophy

Standard image classification maps inputs to a fixed set of discrete labels. Our approach instead learns a **continuous embedding space** — meaning the model can retrieve images for any free-form text query, even queries never seen during training (zero-shot retrieval).

The key insight of contrastive learning: rather than telling the model what an image *is*, we tell it which text *matches* it and which does not.

---

## Dual Encoder Pipeline

```
IMAGE INPUT                          TEXT INPUT
(3, 224, 224)                        "a dog in the snow"
     │                                      │
     ▼                                      ▼
┌─────────────────┐              ┌──────────────────────┐
│   ResNet-50     │              │     DistilBERT       │
│  (frozen, IN)   │              │  (frozen, HuggingFace│
│                 │              │   distilbert-base-   │
│  Pool → 2048    │              │   uncased)           │
│                 │              │  CLS token → 768     │
└────────┬────────┘              └──────────┬───────────┘
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌──────────────────────┐
│ Vision Proj MLP │              │  Text Proj MLP       │
│  2048 → 512     │              │  768 → 512           │
│  ReLU           │              │  ReLU                │
│  Dropout(0.1)   │              │  Dropout(0.1)        │
│  512 → 256      │              │  512 → 256           │
└────────┬────────┘              └──────────┬───────────┘
         │                                  │
         ▼                                  ▼
   L2 Normalize                       L2 Normalize
   (B, 256)                           (B, 256)
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
             Cosine Similarity Matrix
                   (B, B)
                        │
                        ▼
                  InfoNCE Loss
```

---

## Component Details

### Vision Encoder

| Component | Detail |
|-----------|--------|
| Backbone | ResNet-50, pretrained on ImageNet-1k |
| Modification | Remove final FC layer (keep average pooled features) |
| Output | (B, 2048) |
| Trainable? | **No — fully frozen** |
| Projection | Linear(2048, 512) → ReLU → Dropout(0.1) → Linear(512, 256) |
| Final output | (B, 256), L2-normalized |

**Why freeze the backbone?** ResNet-50 already extracts rich visual features from ImageNet. Fine-tuning it on 31k images would overfit badly and is computationally prohibitive on free-tier GPUs. The projection MLP is where alignment learning actually occurs.

**Pre-caching optimization:** Since the backbone is frozen, image features never change. We compute all 31,000 feature vectors once and save them to disk (`data/cached/image_features.pt`). This reduces each training epoch from ~40 minutes to ~2 minutes on a Colab T4.

---

### Text Encoder

| Component | Detail |
|-----------|--------|
| Backbone | `distilbert-base-uncased` (66M params) |
| Token used | `[CLS]` token hidden state |
| Output | (B, 768) |
| Trainable? | **No — fully frozen** |
| Projection | Linear(768, 512) → ReLU → Dropout(0.1) → Linear(512, 256) |
| Final output | (B, 256), L2-normalized |

**Why DistilBERT over BERT-base?** DistilBERT is 40% smaller and 60% faster with only ~3% performance drop. On free-tier GPUs, this matters significantly when processing captions in every training batch.

**Why use CLS token?** The `[CLS]` token aggregates the full sequence representation during BERT's pretraining. It is the standard choice for sentence-level tasks.

---

### Projection Heads

Both encoders output to the same 256-dimensional space through a two-layer MLP:

```python
class ProjectionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        projected = self.net(x)
        return F.normalize(projected, dim=-1)  # L2 normalize — critical!
```

**Why L2 normalize?** Without normalization, the loss can be gamed by scaling embeddings to large magnitudes rather than learning meaningful directions. Normalization constrains all embeddings to a unit hypersphere, forcing the model to encode meaning in *direction*, not *magnitude*.

---

### InfoNCE Loss

Given a batch of B matched (image, text) pairs, we construct a B×B similarity matrix:

```
          text_0   text_1   text_2  ...
image_0  [ sim_00  sim_01  sim_02  ... ]   ← correct match on diagonal
image_1  [ sim_10  sim_11  sim_12  ... ]
image_2  [ sim_20  sim_21  sim_22  ... ]
  ...
```

The loss treats each row as a classification problem: for image_i, which of the B texts is the correct match? The ground truth is always the diagonal (label = i).

```python
def infonce_loss(image_emb, text_emb, temperature):
    logits = (image_emb @ text_emb.T) / temperature   # (B, B)
    labels = torch.arange(B).to(device)               # [0, 1, 2, ..., B-1]
    loss_i2t = F.cross_entropy(logits,   labels)      # image → text
    loss_t2i = F.cross_entropy(logits.T, labels)      # text → image
    return (loss_i2t + loss_t2i) / 2                  # symmetric
```

**Learnable temperature τ:** Temperature controls the sharpness of the distribution. We initialize `τ = log(1/0.07)` and make it a learnable `nn.Parameter`. Small τ → sharper (more confident) distributions. This is what CLIP does.

---

## Inference / Retrieval

At inference time (after training), we:

1. Pre-compute embeddings for all N images in the gallery → matrix of shape (N, 256)
2. Encode the user's query text → vector of shape (1, 256)
3. Compute cosine similarity between query and all gallery embeddings
4. Return the top-K indices

```python
# Pre-built gallery (done once)
gallery_embs = vision_encoder(all_cached_features)   # (31000, 256)

# At query time
query_emb = text_encoder(tokenize(query))            # (1, 256)
scores = query_emb @ gallery_embs.T                  # (1, 31000)
top_k  = scores.topk(5).indices                      # Top 5 image indices
```

This runs in milliseconds — cosine similarity over 31k 256-dim vectors is trivial.

---

## Parameter Count

| Component | Parameters | Trainable? |
|-----------|-----------|------------|
| ResNet-50 backbone | 23.5M | ❌ |
| Vision Projection MLP | ~1.1M | ✅ |
| DistilBERT backbone | 66.4M | ❌ |
| Text Projection MLP | ~0.5M | ✅ |
| Temperature (τ) | 1 | ✅ |
| **Total trainable** | **~1.6M** | ✅ |

Training only 1.6M parameters on a frozen feature space is why this project is feasible on free-tier compute.
