# Data Setup

## Dataset: Flickr30k

Flickr30k contains 31,783 images collected from Flickr, each annotated with **5 independent human-written captions**, giving 158,915 captions total.

### Why Flickr30k?

- Optimal size for a 2-week training pipeline (vs. MS COCO which is ~5x larger)
- 5 captions per image enables **text augmentation** during training (randomly sample 1 caption per epoch)
- Standard benchmark — our R@K results can be compared directly against published models
- Freely available for academic use

---

## Download Instructions

Flickr30k requires a brief registration. Follow these steps:

### Step 1 — Request Access

1. Go to: [https://shannon.cs.illinois.edu/DenotationGraph/](https://shannon.cs.illinois.edu/DenotationGraph/)
2. Fill out the dataset request form with your University of Bern email
3. You will receive a download link via email (usually within a few hours)

### Step 2 — Download Files

You need two things:
- `flickr30k-images.tar.gz` (~11GB) — the raw images
- `results.csv` — the captions file (included in the dataset package)

### Step 3 — Upload to Kaggle Dataset (Recommended)

Since we train on Kaggle, upload the data as a private Kaggle dataset to avoid re-uploading every session:

```bash
# Install Kaggle CLI
pip install kaggle

# Create a new dataset on kaggle.com first, then push:
kaggle datasets init -p /path/to/flickr30k
kaggle datasets create -p /path/to/flickr30k
```

Then in your Kaggle notebook, add it as a dataset input and it will appear at `/kaggle/input/flickr30k/`.

### Step 4 — Verify Structure

After download, your `data/raw/` should look like:

```
data/raw/
├── flickr30k-images/
│   ├── 1000092795.jpg
│   ├── 1000268201.jpg
│   └── ... (31,783 images)
└── results.csv
```

### Step 5 — Verify the CSV format

```python
import pandas as pd
df = pd.read_csv('data/raw/results.csv', sep='|')
print(df.head())
# Expected columns: image_name | comment_number | comment
print(f"Total rows: {len(df)}")        # Should be ~158,915
print(f"Unique images: {df['image_name'].nunique()}")  # Should be ~31,783
```

---

## Dataset Splits

We use the standard Karpathy split for Flickr30k, which is used by virtually all papers in the field (enabling direct comparison):

| Split | Images | Captions |
|-------|--------|----------|
| Train | 29,000 | 145,000 |
| Val   | 1,000  | 5,000   |
| Test  | 1,000  | 5,000   |

The split indices are saved in `data/splits/` as `train_ids.txt`, `val_ids.txt`, `test_ids.txt`.

**To generate these splits**, run:
```bash
python scripts/create_splits.py
```

Or download the pre-made Karpathy split JSON from [Andrej Karpathy's website](https://cs.stanford.edu/people/karpathy/deepimagesent/).

---

## Pre-computed Features

After running `scripts/precompute_features.py`, the following files will be generated in `data/cached/`:

| File | Size | Description |
|------|------|-------------|
| `train_image_features.pt` | ~450MB | ResNet-50 features for 29,000 train images, shape (29000, 2048) |
| `val_image_features.pt`   | ~16MB  | ResNet-50 features for 1,000 val images |
| `test_image_features.pt`  | ~16MB  | ResNet-50 features for 1,000 test images |
| `image_id_to_idx.json`    | ~2MB   | Mapping from image filename to feature matrix index |

> ⚠️ These files are excluded from Git (see `.gitignore`). Each team member must generate them locally or share via Kaggle dataset.

---

## Caption Augmentation

During training, for each image we randomly sample **1 of its 5 captions** per epoch. This means:

- Epoch 1: image_42 is paired with caption #3
- Epoch 2: image_42 is paired with caption #1
- Epoch 3: image_42 is paired with caption #5

This acts as **text-space data augmentation**, exposing the model to the full diversity of human language descriptions and preventing the text encoder from overfitting to specific phrasings.

Implementation in `src/dataset/flickr_dataset.py`:

```python
def __getitem__(self, idx):
    image_id = self.image_ids[idx]
    captions = self.captions[image_id]          # list of 5 captions
    caption  = random.choice(captions)          # random sample per epoch
    features = self.cached_features[idx]        # pre-computed tensor
    return features, caption
```
