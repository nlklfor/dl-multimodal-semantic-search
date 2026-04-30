"""
flickr_dataset.py

Custom PyTorch Dataset for Flickr30k.
Loads pre-computed ResNet-50 image features from cache and
randomly samples 1 of 5 captions per image per epoch.
"""

import random
import torch
from torch.utils.data import Dataset


class FlickrDataset(Dataset):
    """
    Flickr30k dataset using pre-cached image features.

    Args:
        features_path (str): Path to .pt file of shape (N, 2048)
        captions_dict (dict): Maps image_id -> list of 5 caption strings
        image_ids (list): Ordered list of image IDs (index matches features matrix)
        split (str): 'train', 'val', or 'test'
    """

    def __init__(self, features_path, captions_dict, image_ids, split='train'):
        self.features   = torch.load(features_path)   # (N, 2048)
        self.captions   = captions_dict
        self.image_ids  = image_ids
        self.split      = split

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        features = self.features[idx]                 # (2048,)

        captions = self.captions[image_id]            # list of 5 strings
        if self.split == 'train':
            caption = random.choice(captions)         # augmentation: random sample
        else:
            caption = captions[0]                     # deterministic for val/test

        return features, caption
