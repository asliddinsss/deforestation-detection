"""
dataset.py
----------
Custom PyTorch Dataset for satellite image patches.
Supports loading from a CSV manifest or directly from a folder structure.

Folder structure expected:
    data/processed/
        train/
            forest/
            deforested/
        val/
            forest/
            deforested/
        test/
            forest/
            deforested/
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Callable

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────
# Default transforms
# ─────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),  # RandomErasing equivalent
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────

CLASS_TO_IDX = {"forest": 0, "deforested": 1}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class DeforestationDataset(Dataset):
    """
    Loads satellite image patches with binary labels.

    Args:
        root_dir:    Path to split folder (e.g., data/processed/train/)
        transform:   Albumentations transform pipeline
        return_path: If True, also return the image file path (useful for debugging)
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        return_path: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_path = return_path
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        samples = []
        for class_name, label in CLASS_TO_IDX.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"[Warning] Class folder not found: {class_dir}")
                continue
            for img_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                samples.append((img_path, label))
        print(f"[Dataset] Loaded {len(samples)} samples from {self.root_dir}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return image, label, str(img_path)
        return image, label

    def class_counts(self) -> dict:
        """Returns count per class — useful for checking class imbalance."""
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return {IDX_TO_CLASS[k]: v for k, v in counts.items()}


# ─────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────

def get_dataloaders(
    data_dir: str = "data/processed",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).
    """
    train_ds = DeforestationDataset(
        os.path.join(data_dir, "train"),
        transform=get_train_transforms(image_size),
    )
    val_ds = DeforestationDataset(
        os.path.join(data_dir, "val"),
        transform=get_val_transforms(image_size),
    )
    test_ds = DeforestationDataset(
        os.path.join(data_dir, "test"),
        transform=get_val_transforms(image_size),
        return_path=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"\n[DataLoaders Ready]")
    print(f"  Train: {len(train_ds)} samples | Class dist: {train_ds.class_counts()}")
    print(f"  Val:   {len(val_ds)} samples")
    print(f"  Test:  {len(test_ds)} samples\n")

    return train_loader, val_loader, test_loader
