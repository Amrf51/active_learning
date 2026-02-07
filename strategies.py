"""
Active Learning sampling strategies.

All strategies follow the same interface:
    strategy_fn(model, unlabeled_loader, n_samples, device) -> np.ndarray

Returns indices INTO the unlabeled loader (0 to len-1), not absolute dataset indices.
The ALDataManager handles conversion to absolute indices.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


def uncertainty_least_confidence(
    model: torch.nn.Module,
    unlabeled_loader: DataLoader,
    n_samples: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Query instances with lowest prediction confidence.
    
    Uncertainty = 1 - max(softmax)
    High uncertainty = model is unsure about its top prediction.
    
    Args:
        model: PyTorch model
        unlabeled_loader: DataLoader for unlabeled pool
        n_samples: Number of samples to query
        device: Device to run inference on
        
    Returns:
        Indices of most uncertain samples (into unlabeled_loader)
    """
    model.eval()
    confidences = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            max_probs, _ = probs.max(dim=1)
            confidences.extend(max_probs.cpu().numpy())
    
    confidences = np.array(confidences)
    
    # Lower confidence = higher uncertainty = query first
    query_indices = np.argsort(confidences)[:n_samples]
    
    logger.info(f"Least confidence sampling: selected {len(query_indices)} samples")
    
    return query_indices


def uncertainty_entropy(
    model: torch.nn.Module,
    unlabeled_loader: DataLoader,
    n_samples: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Query instances with highest prediction entropy.
    
    Entropy = -sum(p * log(p))
    High entropy = uncertainty spread across multiple classes.
    
    Args:
        model: PyTorch model
        unlabeled_loader: DataLoader for unlabeled pool
        n_samples: Number of samples to query
        device: Device to run inference on
        
    Returns:
        Indices of highest entropy samples (into unlabeled_loader)
    """
    model.eval()
    entropies = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            # Add small epsilon to avoid log(0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
            entropies.extend(entropy.cpu().numpy())
    
    entropies = np.array(entropies)
    
    # Higher entropy = more uncertain = query first
    query_indices = np.argsort(-entropies)[:n_samples]
    
    logger.info(f"Entropy sampling: selected {len(query_indices)} samples")
    
    return query_indices


def margin_sampling(
    model: torch.nn.Module,
    unlabeled_loader: DataLoader,
    n_samples: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Query instances with smallest margin between top-2 predictions.
    
    Margin = P(top1) - P(top2)
    Small margin = model can't decide between two classes.
    
    Args:
        model: PyTorch model
        unlabeled_loader: DataLoader for unlabeled pool
        n_samples: Number of samples to query
        device: Device to run inference on
        
    Returns:
        Indices of smallest margin samples (into unlabeled_loader)
    """
    model.eval()
    margins = []
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            # Get top 2 probabilities
            top2_probs, _ = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            
            if top2_probs.shape[1] >= 2:
                margin = top2_probs[:, 0] - top2_probs[:, 1]
            else:
                margin = top2_probs[:, 0]
            
            margins.extend(margin.cpu().numpy())
    
    margins = np.array(margins)
    
    # Smaller margin = more uncertain = query first
    query_indices = np.argsort(margins)[:n_samples]
    
    logger.info(f"Margin sampling: selected {len(query_indices)} samples")
    
    return query_indices


def random_sampling(
    model: torch.nn.Module,
    unlabeled_loader: DataLoader,
    n_samples: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Query random instances (baseline strategy).
    
    Args:
        model: PyTorch model (unused, for interface consistency)
        unlabeled_loader: DataLoader for unlabeled pool
        n_samples: Number of samples to query
        device: Device (unused, for interface consistency)
        
    Returns:
        Random indices (into unlabeled_loader)
    """
    n_total = len(unlabeled_loader.dataset)
    n_query = min(n_samples, n_total)
    
    query_indices = np.random.choice(n_total, size=n_query, replace=False)
    
    logger.info(f"Random sampling: selected {len(query_indices)} samples")
    
    return query_indices


# Strategy registry
STRATEGIES = {
    "uncertainty": uncertainty_least_confidence,
    "least_confidence": uncertainty_least_confidence,
    "entropy": uncertainty_entropy,
    "margin": margin_sampling,
    "random": random_sampling,
}


def get_strategy(strategy_name: str, uncertainty_method: str = "least_confidence"):
    """
    Get sampling strategy function by name.
    
    Args:
        strategy_name: Name of strategy (uncertainty, margin, entropy, random)
        uncertainty_method: For "uncertainty", which variant (least_confidence or entropy)
        
    Returns:
        Strategy function with signature (model, loader, n_samples, device) -> indices
    """
    strategy_name = strategy_name.lower()
    
    if strategy_name == "uncertainty":
        if uncertainty_method == "entropy":
            return uncertainty_entropy
        return uncertainty_least_confidence
    
    if strategy_name not in STRATEGIES:
        available = list(STRATEGIES.keys())
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {available}")
    
    return STRATEGIES[strategy_name]


def list_available_strategies() -> list:
    """List all available sampling strategies."""
    return list(STRATEGIES.keys())