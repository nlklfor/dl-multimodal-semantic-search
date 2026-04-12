# 🔍 Multimodal Semantic Search Engine

**Aligning Visual and Textual Representations via Contrastive Learning for Semantic Image Retrieval**

> Master's Project — Deep Learning | University of Bern, Switzerland
> 
> Team: Mykyta Slieptsov & Kobiljon Muhammadov | Spring 2026

---

## Overview

This project implements a **CLIP-style dual-encoder architecture** trained with **InfoNCE contrastive loss** to align image and text representations in a shared embedding space. Given a natural language query (e.g., *"a dog playing in the snow"*), the system retrieves the most semantically relevant images from the Flickr30k dataset using vector similarity search.

![Architecture Diagram](assets/architecture.png)

---

## Key Features

- 🖼️ **Vision Encoder:** Frozen ResNet-50 backbone + trainable projection MLP (2048 → 512 → 256)
- 📝 **Text Encoder:** Frozen DistilBERT backbone + trainable projection MLP (768 → 512 → 256)
- ⚡ **Pre-cached image features** for fast Colab/Kaggle training (~2 min/epoch vs ~40 min)
- 📐 **InfoNCE loss** implemented from scratch with learnable temperature
- 📊 **Standard evaluation:** Recall@1, R@5, R@10 (both text→image and image→text)
- 🎮 **Interactive Gradio demo** for free-text image retrieval

---

## Results

| Metric | Text → Image | Image → Text |
|--------|-------------|-------------|
| R@1    | —           | —           |
| R@5    | —           | —           |
| R@10   | —           | —           |

> Results will be populated after training is complete.

---

## Repository Structure

```
multimodal-semantic-search/
│
├── data/
│   ├── raw/                  # Raw Flickr30k images & captions (not tracked by git)
│   ├── cached/               # Pre-computed ResNet-50 features (.pt files)
│   └── splits/               # Train / val / test split indices
│
├── src/
│   ├── dataset/
│   │   └── flickr_dataset.py     # Custom PyTorch Dataset with caption sampling
│   ├── encoders/
│   │   ├── vision_encoder.py     # ResNet-50 backbone + projection MLP
│   │   └── text_encoder.py       # DistilBERT backbone + projection MLP
│   ├── loss/
│   │   └── infonce.py            # InfoNCE loss from scratch
│   ├── evaluation/
│   │   └── recall_at_k.py        # R@1, R@5, R@10 evaluation
│   └── demo/
│       └── app.py                # Gradio search interface
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_caching.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_demo_and_visualization.ipynb
│
├── experiments/
│   └── ablations/                # Temperature ablation results
│
├── scripts/
│   ├── precompute_features.py    # Standalone script to cache ResNet features
│   └── evaluate.py               # Standalone evaluation runner
│
├── tests/
│   ├── test_dataset.py
│   ├── test_loss.py
│   └── test_encoders.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXPERIMENTS.md
│   └── RESULTS.md
│
├── assets/
│   └── architecture.png          # Architecture diagram (for README)
│
├── .gitignore
├── requirements.txt
├── environment.yml
├── config.py                     # Central config (hyperparameters, paths)
└── README.md
```

---

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/nlklfor/dl-multimodal-semantic-search.git
cd dl-multimodal-semantic-search
pip install -r requirements.txt
```

### 2. Download Flickr30k

```bash
# See docs/DATA_SETUP.md for full instructions
# Place images in data/raw/flickr30k-images/
# Place captions in data/raw/results.csv
```

### 3. Pre-compute Image Features

```bash
python scripts/precompute_features.py --output_dir data/cached/
```

### 4. Train

Open and run `notebooks/03_training.ipynb` on Colab/Kaggle, or:

```bash
python src/train.py --config config.py
```

### 5. Run the Demo

```bash
python src/demo/app.py
```

---

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a full explanation of the dual-encoder design, projection heads, and contrastive learning objective.

---

## Experiments & Ablations

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for our temperature ablation study (τ ∈ {0.05, 0.07, 0.10}).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | Model training |
| `torchvision` | ≥0.15 | ResNet-50 backbone |
| `transformers` | ≥4.30 | DistilBERT tokenizer + model |
| `gradio` | ≥3.40 | Demo UI |
| `scikit-learn` | ≥1.3 | UMAP / t-SNE visualization |
| `pandas` | ≥2.0 | Caption CSV loading |
| `Pillow` | ≥9.0 | Image loading |
| `tqdm` | ≥4.65 | Training progress bars |
| `matplotlib` | ≥3.7 | Plotting |

---

## Team & Contributions

| Task | Owner |
|------|-------|
| Data pipeline & pre-caching | Person A |
| Dual encoder architecture | Person B |
| InfoNCE loss | Person A |
| Training loop & checkpointing | Person B |
| Evaluation (R@K) | Person A |
| Gradio demo UI | Person B |
| Ablation experiments | Both |
| Report & slides | Both |

---

## References

1. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). OpenAI.
2. Oord et al. (2018). *Representation Learning with Contrastive Predictive Coding* (InfoNCE).
3. Young et al. (2014). *From image descriptions to visual denotations* (Flickr30k).
4. Sanh et al. (2019). *DistilBERT, a distilled version of BERT.*

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.
