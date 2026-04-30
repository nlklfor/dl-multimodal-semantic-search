"""
recall_at_k.py

Standard retrieval evaluation metrics for image-text matching.
Computes Recall@K for both directions:
  - Text → Image: Given a caption, retrieve the correct image
  - Image → Text: Given an image, retrieve any of its correct captions
"""

import torch
from config import RECALL_AT_K


@torch.no_grad()
def evaluate_recall(image_embs, text_embs, k_values=RECALL_AT_K, n_captions_per_image=5):
    """
    Compute Recall@K for Flickr30k test set (1000 images, 5000 captions).

    Args:
        image_embs: Tensor (N_images, D) — embeddings for all test images
        text_embs:  Tensor (N_captions, D) — embeddings for all test captions
                    Assumed ordered: [img0_cap0, img0_cap1, ..., img0_cap4, img1_cap0, ...]
        k_values:   list of K values, e.g. [1, 5, 10]
        n_captions_per_image: 5 for Flickr30k

    Returns:
        dict with keys like 't2i_R@1', 't2i_R@5', 'i2t_R@1', etc.
    """
    N_images   = image_embs.shape[0]
    N_captions = text_embs.shape[0]

    # Compute full similarity matrix: (N_captions, N_images)
    sim_matrix = text_embs @ image_embs.T          # cosine sim (embeddings are L2 normalized)

    results = {}

    # ── Text → Image ──────────────────────────────────────────────────────────
    # For each caption, rank all images. Correct image = caption_idx // n_captions_per_image
    t2i_correct = 0
    t2i_correct_at_k = {k: 0 for k in k_values}

    for cap_idx in range(N_captions):
        correct_image_idx = cap_idx // n_captions_per_image
        scores = sim_matrix[cap_idx]                         # (N_images,)
        ranked = scores.argsort(descending=True)

        for k in k_values:
            if correct_image_idx in ranked[:k]:
                t2i_correct_at_k[k] += 1

    for k in k_values:
        results[f't2i_R@{k}'] = round(100.0 * t2i_correct_at_k[k] / N_captions, 2)

    # ── Image → Text ──────────────────────────────────────────────────────────
    # For each image, rank all captions. Correct captions = [img_idx*5 .. img_idx*5+4]
    i2t_correct_at_k = {k: 0 for k in k_values}
    sim_matrix_T = sim_matrix.T                              # (N_images, N_captions)

    for img_idx in range(N_images):
        correct_cap_indices = set(range(
            img_idx * n_captions_per_image,
            (img_idx + 1) * n_captions_per_image
        ))
        scores = sim_matrix_T[img_idx]                       # (N_captions,)
        ranked = scores.argsort(descending=True)

        for k in k_values:
            top_k_set = set(ranked[:k].tolist())
            if correct_cap_indices & top_k_set:              # any correct caption in top-k?
                i2t_correct_at_k[k] += 1

    for k in k_values:
        results[f'i2t_R@{k}'] = round(100.0 * i2t_correct_at_k[k] / N_images, 2)

    return results


def print_results(results):
    """Pretty-print evaluation results."""
    print("\n" + "="*45)
    print(f"{'Retrieval Evaluation Results':^45}")
    print("="*45)
    print(f"{'Metric':<20} {'Score':>10}")
    print("-"*45)
    for k in RECALL_AT_K:
        print(f"  Text → Image R@{k:<3}  {results[f't2i_R@{k}']:>9.2f}%")
    print("-"*45)
    for k in RECALL_AT_K:
        print(f"  Image → Text R@{k:<3}  {results[f'i2t_R@{k}']:>9.2f}%")
    print("="*45 + "\n")
