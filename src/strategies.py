"""Active Learning sampling strategies."""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class UncertaintySampling:
    """Uncertainty-based sampling strategies."""
    
    @staticmethod
    def least_confidence(model, unlabeled_pool, n_samples: int, device: str = "cuda") -> np.ndarray:
        """Query instances with least confidence (max softmax < threshold).
        
        Args:
            model: PyTorch model
            unlabeled_pool: DataLoader of unlabeled data
            n_samples: Number of samples to query
            device: Device to run on
            
        Returns:
            Indices of most uncertain samples
        """
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for images, _ in unlabeled_pool:
                images = images.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                max_probs, _ = probabilities.max(dim=1)
                # Uncertainty is 1 - max probability
                uncertainty = 1 - max_probs
                uncertainties.extend(uncertainty.cpu().numpy())
        
        uncertainties = np.array(uncertainties)
        # Get indices of top n_samples most uncertain instances
        query_indices = np.argsort(-uncertainties)[:n_samples]
        
        return query_indices
    
    @staticmethod
    def entropy(model, unlabeled_pool, n_samples: int, device: str = "cuda") -> np.ndarray:
        """Query instances with highest entropy.
        
        Args:
            model: PyTorch model
            unlabeled_pool: DataLoader of unlabeled data
            n_samples: Number of samples to query
            device: Device to run on
            
        Returns:
            Indices of most uncertain samples (highest entropy)
        """
        model.eval()
        entropies = []
        
        with torch.no_grad():
            for images, _ in unlabeled_pool:
                images = images.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                # Entropy = -sum(p * log(p))
                entropy = -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        entropies = np.array(entropies)
        # Get indices of top n_samples highest entropy instances
        query_indices = np.argsort(-entropies)[:n_samples]
        
        return query_indices


class MarginSampling:
    """Margin-based sampling (difference between top-2 predictions)."""
    
    @staticmethod
    def margin(model, unlabeled_pool, n_samples: int, device: str = "cuda") -> np.ndarray:
        """Query instances with smallest margin (least confident about top-2 classes).
        
        Args:
            model: PyTorch model
            unlabeled_pool: DataLoader of unlabeled data
            n_samples: Number of samples to query
            device: Device to run on
            
        Returns:
            Indices of samples with smallest margins
        """
        model.eval()
        margins = []
        
        with torch.no_grad():
            for images, _ in unlabeled_pool:
                images = images.to(device)
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                # Get top 2 probabilities
                top_probs, _ = torch.topk(probabilities, 2, dim=1)
                # Margin = difference between top 2
                margin = top_probs[:, 0] - top_probs[:, 1]
                margins.extend(margin.cpu().numpy())
        
        margins = np.array(margins)
        # Get indices of smallest margins (most uncertain)
        query_indices = np.argsort(margins)[:n_samples]
        
        return query_indices


class DiversitySampling:
    """Diversity-based sampling (feature-space distance)."""
    
    @staticmethod
    def kmeans_sampling(model, unlabeled_pool, n_samples: int, 
                       device: str = "cuda", layer: str = "penultimate") -> np.ndarray:
        """Query instances that are diverse in feature space using k-means.
        
        Args:
            model: PyTorch model
            unlabeled_pool: DataLoader of unlabeled data
            n_samples: Number of samples to query
            device: Device to run on
            layer: Which layer to extract features from
            
        Returns:
            Indices of diverse samples
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            logger.error("scikit-learn required for KMeans sampling")
            raise
        
        model.eval()
        features = []
        
        # Extract features from penultimate layer
        with torch.no_grad():
            for images, _ in unlabeled_pool:
                images = images.to(device)
                # Get features from second-to-last layer
                feature_vec = model.forward_features(images) if hasattr(model, 'forward_features') else \
                             torch.flatten(model(images, features=True), 1)
                features.extend(feature_vec.cpu().numpy())
        
        features = np.array(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=min(n_samples, len(features)), random_state=42)
        kmeans.fit(features)
        
        # Query samples closest to cluster centers
        distances = kmeans.transform(features)
        query_indices = np.argmin(distances, axis=0)
        
        return query_indices


class RandomSampling:
    """Random baseline sampling."""
    
    @staticmethod
    def random(n_total: int, n_samples: int) -> np.ndarray:
        """Query random instances.
        
        Args:
            n_total: Total number of instances
            n_samples: Number of samples to query
            
        Returns:
            Indices of random samples
        """
        query_indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
        return query_indices


def get_strategy(strategy_name: str, uncertainty_method: str = "least_confidence"):
    """Get sampling strategy function.
    
    Args:
        strategy_name: Name of strategy (uncertainty, margin, entropy, random, kmeans)
        uncertainty_method: For uncertainty strategy, which method to use
        
    Returns:
        Strategy function
    """
    strategy_name = strategy_name.lower()
    
    if strategy_name == "uncertainty":
        if uncertainty_method == "least_confidence":
            return UncertaintySampling.least_confidence
        elif uncertainty_method == "entropy":
            return UncertaintySampling.entropy
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
    
    elif strategy_name == "margin":
        return MarginSampling.margin
    
    elif strategy_name == "entropy":
        return UncertaintySampling.entropy
    
    elif strategy_name == "random":
        return RandomSampling.random
    
    elif strategy_name == "kmeans":
        return DiversitySampling.kmeans_sampling
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def list_available_strategies() -> list:
    """List all available sampling strategies.
    
    Returns:
        List of strategy names
    """
    return ["uncertainty", "margin", "entropy", "random", "kmeans"]