"""
train.py

Main training script for the dual-encoder contrastive learning model.
Designed to run on Kaggle (30hr/week free T4) or Google Colab.

Usage:
    python src/train.py
    python src/train.py --temperature 0.07   # fixed temperature for ablation
    python src/train.py --epochs 15
"""

import os
import json
import math
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS,
    SEED, TEMPERATURE, CHECKPOINTS_DIR, NUM_WORKERS, PIN_MEMORY,
    CACHED_DIR, CAPTIONS_FILE, SPLITS_DIR, TEXT_MODEL_NAME, MAX_TOKEN_LENGTH,
    SAVE_EVERY_N_EPOCHS,
)
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder
from src.loss.infonce import InfoNCELoss
from src.dataset.flickr_dataset import FlickrDataset
from src.evaluation.recall_at_k import evaluate_recall, print_results


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─── Load captions ────────────────────────────────────────────────────────────

def load_captions(captions_file):
    """Returns dict: image_id → list of 5 caption strings."""
    import pandas as pd
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


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(epoch, vision_enc, text_enc, optimizer, loss, tag=""):
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    name = f"checkpoint_epoch{epoch:02d}{('_' + tag) if tag else ''}.pt"
    path = os.path.join(CHECKPOINTS_DIR, name)
    torch.save({
        'epoch':              epoch,
        'vision_encoder':     vision_enc.state_dict(),
        'text_encoder':       text_enc.state_dict(),
        'optimizer':          optimizer.state_dict(),
        'loss':               loss,
    }, path)
    print(f"  💾 Checkpoint saved: {path}")
    return path


def load_checkpoint(path, vision_enc, text_enc, optimizer):
    ckpt = torch.load(path)
    vision_enc.load_state_dict(ckpt['vision_encoder'])
    text_enc.load_state_dict(ckpt['text_encoder'])
    optimizer.load_state_dict(ckpt['optimizer'])
    print(f"  ✅ Resumed from epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}")
    return ckpt['epoch']


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n📂 Loading data...")
    captions    = load_captions(CAPTIONS_FILE)
    train_ids   = load_split_ids('train')
    val_ids     = load_split_ids('val')

    train_dataset = FlickrDataset(
        features_path = os.path.join(CACHED_DIR, "train_image_features.pt"),
        captions_dict = captions,
        image_ids     = train_ids,
        split         = 'train',
    )
    val_dataset = FlickrDataset(
        features_path = os.path.join(CACHED_DIR, "val_image_features.pt"),
        captions_dict = captions,
        image_ids     = val_ids,
        split         = 'val',
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=PIN_MEMORY)

    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples")

    # ── Models ───────────────────────────────────────────────────────────────
    print("\n🏗️  Building encoders...")
    vision_enc = VisionEncoder().to(device)
    text_enc   = TextEncoder().to(device)

    # Only projection MLP params are trainable
    trainable_params = (
        list(vision_enc.projection.parameters()) +
        list(text_enc.projection.parameters())
    )

    temperature = args.temperature if args.temperature else TEMPERATURE
    criterion   = InfoNCELoss(temperature=temperature).to(device)
    if criterion.fixed is False:
        trainable_params.append(criterion.log_temperature)

    n_params = sum(p.numel() for p in trainable_params)
    print(f"   Trainable parameters: {n_params:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)

    # ── Resume from checkpoint if specified ───────────────────────────────────
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, vision_enc, text_enc, optimizer)

    # ── Training ──────────────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {args.epochs} epochs...\n")
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(start_epoch, args.epochs):

        # ── Train epoch ───────────────────────────────────────────────────────
        vision_enc.train()
        text_enc.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [Train]")
        for image_feats, captions_batch in pbar:
            image_feats = image_feats.to(device)

            image_emb = vision_enc(image_feats)
            text_emb  = text_enc(captions_batch, device)

            loss = criterion(image_emb, text_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'τ': f'{criterion.temperature:.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # ── Validation ────────────────────────────────────────────────────────
        vision_enc.eval()
        text_enc.eval()
        val_loss = 0.0

        with torch.no_grad():
            for image_feats, captions_batch in tqdm(val_loader,
                                                    desc=f"Epoch {epoch+1:02d}/{args.epochs} [Val]"):
                image_feats = image_feats.to(device)
                image_emb = vision_enc(image_feats)
                text_emb  = text_enc(captions_batch, device)
                loss = criterion(image_emb, text_emb)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"\n  Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | τ: {criterion.temperature:.4f}\n")

        # ── Save checkpoint every N epochs + final (keeps disk usage sane) ────
        tag         = f"temp{args.temperature}" if args.temperature else "learnable"
        is_milestone = (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0
        is_final     = (epoch + 1) == args.epochs
        if is_milestone or is_final:
            save_checkpoint(epoch + 1, vision_enc, text_enc, optimizer,
                            avg_val_loss, tag=tag)

    # ── Save training history ─────────────────────────────────────────────────
    hist_path = os.path.join(CHECKPOINTS_DIR, "training_history.json")
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n📊 Training history saved: {hist_path}")
    print("✅ Training complete.")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Fixed temperature. Omit for learnable.")
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint .pt file to resume from")
    args = parser.parse_args()
    train(args)
