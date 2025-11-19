"""Data loading and preprocessing utilities."""

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_transforms(augmentation: bool = True, phase: str = "train"):
    """Get data transforms (augmentation for training, normalization for all).
    
    Args:
        augmentation: Whether to apply augmentation
        phase: "train", "val", or "test"
        
    Returns:
        torchvision.transforms.Compose object
    """
    # ImageNet normalization (standard for pretrained models)
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
        # No augmentation for val/test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(config, batch_size: int = 32, seed: int = 42):
    """Load data and return train, val, test dataloaders.
    
    Args:
        config: DataConfig object
        batch_size: Batch size for dataloaders (default 32)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(seed)
    
    # Load full dataset
    data_path = Path(config.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    # Load dataset with training transforms (will split afterwards)
    dataset = ImageFolder(
        root=str(data_path),
        transform=get_transforms(augmentation=False, phase="train")  # Will reapply later
    )
    
    logger.info(f"Loaded dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    logger.info(f"Classes: {dataset.classes}")
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * config.train_split)
    val_size = int(total_size * config.val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    logger.info(f"Split: train={train_size}, val={val_size}, test={test_size}")
    
    # Re-apply appropriate transforms to each split
    train_dataset.dataset.transform = get_transforms(
        augmentation=config.augmentation,
        phase="train"
    )
    val_dataset.dataset.transform = get_transforms(augmentation=False, phase="val")
    test_dataset.dataset.transform = get_transforms(augmentation=False, phase="test")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_class_names(data_dir: str):
    """Get class names from data directory.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        List of class names
    """
    dataset = ImageFolder(root=data_dir)
    return dataset.classes


def get_dataset_info(data_dir: str):
    """Get information about dataset.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary with dataset stats
    """
    dataset = ImageFolder(root=data_dir)
    
    class_counts = {}
    for class_idx in range(len(dataset.classes)):
        class_counts[dataset.classes[class_idx]] = sum(
            1 for img, label in dataset if label == class_idx
        )
    
    return {
        "total_images": len(dataset),
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "class_counts": class_counts,
    }