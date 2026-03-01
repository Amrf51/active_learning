"""Data loading and preprocessing utilities."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
from typing import Dict, Tuple, List
import logging
import os

logger = logging.getLogger(__name__)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    Find class folders in directory, filtering out hidden/system folders.

    Filters out:
    - Hidden directories (starting with '.')
    - System directories like __pycache__

    Args:
        directory: Root directory path

    Returns:
        Tuple of (class_names, class_to_idx)
    """
    classes = []
    for entry in os.scandir(directory):
        # Skip non-directories
        if not entry.is_dir():
            continue
        # Skip hidden directories (., .ipynb_checkpoints, etc.)
        if entry.name.startswith('.'):
            continue
        # Skip system directories
        if entry.name in ['__pycache__', '__MACOSX']:
            continue
        classes.append(entry.name)

    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class FilteredImageFolder(ImageFolder):
    """
    ImageFolder that filters out hidden and system directories.

    This prevents errors when directories like .ipynb_checkpoints, .DS_Store,
    or __pycache__ exist in the data directory.
    """

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Override find_classes to filter unwanted directories."""
        return find_classes(directory)


def get_transforms(augmentation: bool = True, phase: str = "train"):
    """Get data transforms.
    
    Args:
        augmentation: Whether to apply augmentation
        phase: "train", "val", or "test"
        
    Returns:
        torchvision.transforms.Compose object
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if phase == "train" and augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


class ImageFolderWithIndex(Dataset):
    """
    Wrapper around ImageFolder that applies different transforms based on index sets.
    
    This avoids the overhead of custom __getitem__ implementations while
    allowing different transforms for train/val/test splits.
    """
    
    def __init__(
        self,
        root: str,
        train_indices: List[int],
        val_indices: List[int],
        test_indices: List[int],
        train_transform,
        eval_transform
    ):
        self.dataset = FilteredImageFolder(root=root)
        self.train_indices_set = set(train_indices)
        self.val_indices_set = set(val_indices)
        self.test_indices_set = set(test_indices)
        self.train_transform = train_transform
        self.eval_transform = eval_transform

        # Store all indices for full dataset access
        self.all_indices = train_indices + val_indices + test_indices
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load image and label using ImageFolder's method
        image, label = self.dataset[idx]
        
        # Image is already a PIL Image from ImageFolder (no transform set)
        # Apply appropriate transform based on which split this index belongs to
        if idx in self.train_indices_set:
            image = self.train_transform(image)
        else:
            image = self.eval_transform(image)
        
        return image, label
    
    @property
    def classes(self):
        return self.dataset.classes
    
    @property
    def samples(self):
        return self.dataset.samples


class SplitSubset(Dataset):
    """
    Subset that references a parent dataset with precomputed transforms.
    """
    
    def __init__(self, parent_dataset: ImageFolderWithIndex, indices: List[int]):
        self.parent = parent_dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.parent[actual_idx]


def get_datasets(
    data_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    augmentation: bool = True,
    seed: int = 42
) -> Dict:
    """Load dataset and create train/val/test splits.
    
    Args:
        data_dir: Path to data directory (ImageFolder structure)
        val_split: Fraction for validation
        test_split: Fraction for test
        augmentation: Whether to apply augmentation to training data
        seed: Random seed for reproducibility
        
    Returns:
        Dict with train_dataset, val_dataset, test_dataset, class_names, etc.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # First, get dataset size
    temp_dataset = FilteredImageFolder(root=str(data_path))
    total_size = len(temp_dataset)
    class_names = temp_dataset.classes
    
    logger.info(f"Loaded dataset: {total_size} images, {len(class_names)} classes")
    logger.info(f"Classes: {class_names}")
    
    # Set seed and create shuffled indices
    np.random.seed(seed)
    all_indices = np.arange(total_size)
    np.random.shuffle(all_indices)
    
    # Calculate split sizes
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    # Split indices
    train_indices = all_indices[:train_size].tolist()
    val_indices = all_indices[train_size:train_size + val_size].tolist()
    test_indices = all_indices[train_size + val_size:].tolist()
    
    # Create transforms
    train_transform = get_transforms(augmentation=augmentation, phase="train")
    eval_transform = get_transforms(augmentation=False, phase="val")
    
    # Create the smart dataset that knows which transform to apply
    full_dataset = ImageFolderWithIndex(
        root=str(data_path),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    
    # Create subsets
    train_dataset = SplitSubset(full_dataset, train_indices)
    val_dataset = SplitSubset(full_dataset, val_indices)
    test_dataset = SplitSubset(full_dataset, test_indices)
    
    splits_info = {
        "total_samples": total_size,
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "test_samples": len(test_indices),
        "train_percentage": f"{100 * len(train_indices) / total_size:.1f}%",
        "val_percentage": f"{100 * len(val_indices) / total_size:.1f}%",
        "test_percentage": f"{100 * len(test_indices) / total_size:.1f}%",
        "seed": seed,
    }
    
    logger.info(f"Splits created - Train: {splits_info['train_samples']}, "
                f"Val: {splits_info['val_samples']}, Test: {splits_info['test_samples']}")
    
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "full_dataset": full_dataset,  # For AL to access
        "train_indices": train_indices,
        "class_names": class_names,
        "num_classes": len(class_names),
        "splits_info": splits_info,
    }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    augmentation: bool = True,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Convenience function that returns DataLoaders directly.
    
    Args:
        data_dir: Path to data directory
        batch_size: Batch size for all loaders
        val_split: Fraction for validation
        test_split: Fraction for test
        augmentation: Whether to apply augmentation
        num_workers: Number of DataLoader workers
        seed: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    datasets = get_datasets(
        data_dir=data_dir,
        val_split=val_split,
        test_split=test_split,
        augmentation=augmentation,
        seed=seed
    )
    
    pin = torch.cuda.is_available()
    
    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin
    )
    
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )
    
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )
    
    dataset_info = {
        "class_names": datasets["class_names"],
        "num_classes": datasets["num_classes"],
        "splits_info": datasets["splits_info"],
    }
    
    return train_loader, val_loader, test_loader, dataset_info


def get_class_names(data_dir: str) -> List[str]:
    """Get class names from data directory."""
    dataset = FilteredImageFolder(root=data_dir)
    return dataset.classes


def get_dataset_info(data_dir: str) -> Dict:
    """Get information about dataset."""
    dataset = FilteredImageFolder(root=data_dir)
    
    class_counts = {}
    for _, label in dataset.samples:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    return {
        "total_images": len(dataset),
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "class_counts": class_counts,
    }