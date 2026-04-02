"""
losses.py — Custom loss functions for active learning experiments.

Provides:
    SupConLoss      — Supervised Contrastive Loss (Khosla et al., 2020)
    ProjectionHead  — MLP projection head used with SupCon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls same-class embeddings together, pushes different-class
    embeddings apart using temperature-scaled cosine similarity.

    Args:
        temperature: Scaling factor for logits (default 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [N, D] L2-normalised projected embeddings
            labels:   [N]    integer class labels

        Returns:
            Scalar loss (0.0 if no sample has a positive pair in the batch)
        """
        device = features.device
        features = F.normalize(features, dim=1)

        # Similarity matrix scaled by temperature
        similarity = torch.matmul(features, features.T) / self.temperature

        # Positive mask: same class, excluding self
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask.fill_diagonal_(0.0)

        # Numerically stable log-sum-exp
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        # Exclude self from denominator
        self_mask = torch.ones_like(mask) - torch.eye(features.size(0), device=device)
        exp_logits = torch.exp(logits) * self_mask

        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_count = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (positive_count + 1e-12)

        # Only average over samples that have at least one positive pair
        valid = positive_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        return -mean_log_prob[valid].mean()


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head for contrastive learning.

    Maps backbone embeddings to a lower-dimensional space where the
    contrastive loss is computed. Not used at inference time.

    Args:
        in_dim:     Input dimension (= backbone feature dim)
        hidden_dim: Hidden layer width (default 512)
        out_dim:    Output projection dimension (default 128)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projected embeddings."""
        return F.normalize(self.net(x), dim=1)
