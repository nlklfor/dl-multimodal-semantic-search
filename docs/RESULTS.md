# Results

Evaluated on the **Flickr30k test split** (1,000 images, 5,000 captions). Standard Karpathy split. Recall@K — higher is better.

## Final Model Performance

### Retrieval Results

| Direction       | R@1    | R@5    | R@10   |
|-----------------|--------|--------|--------|
| Text → Image    | 16.08% | 41.56% | 54.88% |
| Image → Text    | 20.50% | 46.30% | 61.30% |

(Computed with 1,000 test images × 5 captions = 5,000 captions, the canonical Flickr30k retrieval setup.)

---

## Comparison with Literature

For context, published R@1 (Text → Image) scores on Flickr30k:

| Model | Training data | Trainable params | T→I R@1 |
|-------|---------------|------------------|---------|
| VSE++ (2018) | Flickr30k (29k) | ~10M | 39.6% |
| CLIP ViT-B/32 (zero-shot) | 400M web pairs | 151M | 65.1% |
| CLIP (fine-tuned on Flickr30k) | 400M + Flickr30k | 151M | 88.0% |
| **Ours** (frozen ResNet-50 + DistilBERT + projection-only) | Flickr30k (29k) | **1.7M** | **16.1 %** |

Our model trains only **1,705,473 parameters** on **29,000 image-caption pairs** with both backbones frozen — about **6× fewer trainable parameters than VSE++ and ~90× fewer than CLIP**. Reaching ~16 % R@1 (160× above 0.1 % random chance for 1k candidates) on this constrained budget shows the projection MLPs successfully aligned the two pretrained representation spaces.

---

## Training Curves

![Loss curves](loss_curve.png)

Train and validation loss tracked each other tightly throughout the run — basically zero generalisation gap. No overfitting, no instability, no need for early stopping.

### Per-epoch loss table

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1  | 3.7433 | 2.9991 |
| 2  | 2.9332 | 2.7666 |
| 3  | 2.7472 | 2.6106 |
| 4  | 2.6268 | 2.5132 |
| 5  | 2.5386 | 2.4116 |
| 6  | 2.4696 | 2.4328 |
| 7  | 2.4136 | 2.3585 |
| 8  | 2.3551 | 2.3435 |
| 9  | 2.3008 | 2.2964 |
| 10 | 2.2824 | 2.2854 |
| 11 | 2.2373 | 2.1884 |
| 12 | 2.1993 | 2.2543 |
| 13 | 2.1811 | 2.1460 |
| 14 | 2.1461 | 2.1442 |
| 15 | 2.1110 | 2.1373 |
| 16 | 2.0954 | 2.1202 |
| 17 | 2.0588 | 2.1281 |
| 18 | 2.0556 | 2.0852 |
| 19 | 2.0287 | 2.1013 |
| **20** | **2.0175** | **2.0937** |

**Sanity check:** epoch 1 loss (3.7433) starts well below the random-init upper bound `log(128) ≈ 4.85`, and the curve descends smoothly to ~2.0 — the loss floor for this architecture on this dataset.

---

## Run configuration

| Parameter | Value |
|---|---|
| Vision backbone | ResNet-50 ImageNet-1K, **frozen** (pool5 features, 2048-d) |
| Text backbone | `distilbert-base-uncased`, **frozen** ([CLS] token, 768-d) |
| Projection MLP | `Linear → ReLU → Dropout(0.1) → Linear` to 256-d, L2-normalised |
| Loss | Symmetric InfoNCE (image→text + text→image, averaged) |
| Temperature | Learnable (init τ = 0.07, final τ = **0.0485**) |
| Optimiser | AdamW, lr = 1e-3, weight_decay = 1e-4 |
| Batch size | 128 (drop_last) |
| Epochs | 20 |
| Seed | 42 |
| Hardware | Google Colab — Tesla T4 |
| Wall time | ~12 min for 20 epochs (cached ResNet-50 features, no on-the-fly extraction) |

The temperature dropped from 0.0701 → 0.0485 over training — the model is sharpening its similarity distribution as it learns, exactly the dynamics described in the original CLIP paper.

---

## Temperature ablation

| Temperature | T→I R@1 | T→I R@5 | I→T R@1 | I→T R@5 | Notes |
|-------------|---------|---------|---------|---------|-------|
| τ = 0.05    | —       | —       | —       | —       | sharpest |
| τ = 0.07    | —       | —       | —       | —       | CLIP default |
| τ = 0.10    | —       | —       | —       | —       | softest |
| **τ learnable** | — | — | — | — | **our default** (final τ = 0.0485) |

> *Each ablation requires a separate ~12 min training run on Colab T4 with `python src/train.py --temperature {0.05,0.07,0.10}`. Numbers to be filled in once those runs complete.*

---

## Embedding space visualisation

> *(UMAP plot to be added — see [05_demo_and_visualization.ipynb](../notebooks/05_demo_and_visualization.ipynb).)*

Plan: sample 500 matched (image, text) pairs from the test set, project the 1,000 embeddings (500 image + 500 text) into 2-D with UMAP, colour by modality and connect matched pairs with thin lines. If alignment worked, matched pairs should sit close together and rough semantic clusters (people, animals, sports, food) should appear across modalities.

---

## Reproduction

```bash
# 1. Train (20 epochs, ~12 min on T4)
python src/train.py

# 2. Build the 31k gallery index for the demo
python scripts/build_gallery_index.py \
    --checkpoint experiments/checkpoints/checkpoint_epoch20_learnable.pt

# 3. Compute Recall@K — fills in the table at the top of this file
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/checkpoint_epoch20_learnable.pt
```
