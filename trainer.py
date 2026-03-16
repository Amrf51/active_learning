"""
Trainer - Handles model training and evaluation.

This trainer supports two modes:
1. Batch mode: train() runs all epochs (for CLI/batch execution)
2. Step mode: train_single_epoch() for interactive dashboard

The ActiveLearningLoop orchestrator commands the Trainer when needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models import get_model, extract_features, get_feature_dim
from losses import SupConLoss, ProjectionHead
from state import EpochMetrics

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        exp_dir: Path,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Config object with training parameters
            exp_dir: Experiment directory for saving results
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.device = device
        
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.training.label_smoothing
        )

        # SupCon / combined loss: create projection head alongside the model
        loss_fn = self.config.training.loss_fn
        if loss_fn in ("supcon", "combined"):
            feat_dim = get_feature_dim(self.model)
            self.projection_head = ProjectionHead(in_dim=feat_dim).to(device)
            self.supcon_criterion = SupConLoss(
                temperature=self.config.training.supcon_temperature
            )
        else:
            self.projection_head = None
            self.supcon_criterion = None

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch": []
        }
        
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self._backbone_frozen = False

        logger.info(f"Trainer initialized | Device: {device}")
    
    def _get_head_module(self):
        """Return the classification head module (fc / classifier / head)."""
        return (
            getattr(self.model, 'fc', None)
            or getattr(self.model, 'classifier', None)
            or getattr(self.model, 'head', None)
        )

    def _create_optimizer(self):
        """Create optimizer with discriminative LR: backbone at reduced rate, head at full rate."""
        name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        backbone_lr = lr * self.config.training.backbone_lr_factor

        head_module = self._get_head_module()
        if head_module is None:
            params = [{"params": self.model.parameters(), "lr": lr}]
        else:
            head_ids = {id(p) for p in head_module.parameters()}
            backbone_params = [p for p in self.model.parameters() if id(p) not in head_ids]
            head_params = [p for p in self.model.parameters() if id(p) in head_ids]
            params = [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": lr},
            ]

        # Projection head (SupCon) trained at the same rate as the classifier head
        if self.projection_head is not None:
            params.append({"params": self.projection_head.parameters(), "lr": lr})

        if name == "adam":
            return optim.Adam(params, weight_decay=wd)
        elif name == "sgd":
            return optim.SGD(params, momentum=0.9, weight_decay=wd)
        elif name == "adamw":
            return optim.AdamW(params, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def freeze_backbone(self):
        """Freeze all backbone parameters; head stays trainable."""
        head_module = self._get_head_module()
        head_ids = {id(p) for p in head_module.parameters()} if head_module else set()
        for param in self.model.parameters():
            if id(param) not in head_ids:
                param.requires_grad = False
        logger.info("Backbone frozen — head-only training active")

    def unfreeze_backbone(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — discriminative LR fine-tuning active")

    def _create_scheduler(self, skip_warmup: bool = False):
        """
        Create LR scheduler with optional linear warmup.

        Args:
            skip_warmup: If True, skip the warmup phase (used when rebuilding
                         the scheduler after backbone unfreeze, since warmup is done).
        """
        sched = self.config.training.scheduler
        warmup = 0 if skip_warmup else self.config.training.warmup_epochs
        total_epochs = self.config.training.epochs

        if sched == "none":
            return None

        if sched == "cosine":
            main = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max(1, total_epochs - warmup)
            )
            if warmup > 0:
                warmup_sched = optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup
                )
                return optim.lr_scheduler.SequentialLR(
                    self.optimizer, schedulers=[warmup_sched, main], milestones=[warmup]
                )
            return main
        elif sched == "plateau":
            # ReduceLROnPlateau requires the metric value at each step, which is
            # incompatible with SequentialLR. Skip warmup for plateau.
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=2
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched}")

    def reset_model_weights(self, mode: str = "continue", cycle: int = 1):
        """
        Reset model for a new AL cycle.

        Modes:
            "continue"  — Preferred. Cycle 1: freeze backbone so the random head
                          can warm up without corrupting pretrained features.
                          Cycle 2+: keep all weights, reset optimizer/tracking only.
            "pretrained" — Reload ImageNet weights every cycle (independent experiments).
            "head_only"  — Keep backbone, reset head. Freeze backbone for warmup.
            "none"       — Keep everything, reset optimizer + tracking.
        """
        if mode == "continue":
            if cycle == 1:
                # Backbone is already pretrained; freeze it so the random head
                # has safe warmup epochs before full fine-tuning begins.
                self.freeze_backbone()
                self._backbone_frozen = True
            else:
                # Carry forward all learned weights; just reset optimizer momentum
                # and tracking so the new cycle starts with a clean LR schedule.
                self.unfreeze_backbone()
                self._backbone_frozen = False
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self._reset_tracking()
            return

        if mode == "pretrained":
            logger.info("Reset mode: pretrained — reloading ImageNet weights")
            self.model = get_model(
                name=self.config.model.name,
                num_classes=self.config.model.num_classes,
                pretrained=self.config.model.pretrained,
                device=self.device
            )
            if self.projection_head is not None:
                feat_dim = get_feature_dim(self.model)
                self.projection_head = ProjectionHead(in_dim=feat_dim).to(self.device)
            self.freeze_backbone()
            self._backbone_frozen = True
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self._reset_tracking()
            return

        if mode == "head_only":
            logger.info("Reset mode: head_only — resetting classification head")
            head_module = self._get_head_module()
            if head_module is not None:
                if isinstance(head_module, nn.Sequential):
                    for layer in head_module:
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                else:
                    head_module.reset_parameters()
            self.freeze_backbone()
            self._backbone_frozen = True
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self._reset_tracking()
            return

        if mode == "none":
            logger.info("Reset mode: none — keeping weights, resetting optimizer + tracking")
            self.unfreeze_backbone()
            self._backbone_frozen = False
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            self._reset_tracking()
            return

        raise ValueError(f"Unknown reset mode: {mode}")
    
    def _reset_tracking(self):
        """Reset training history and tracking variables."""
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch": []
        }
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
            
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.train()
        if self.projection_head is not None:
            self.projection_head.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        loss_fn = self.config.training.loss_fn
        alpha = self.config.training.supcon_weight

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            if loss_fn != "cross_entropy" and self.projection_head is not None:
                # Register hook before the single forward pass so both CE and
                # SupCon share the same computation graph (no double pass).
                hook_out = {}
                def _hook(m, i, o):
                    hook_out['feat'] = o
                handle = self.model.global_pool.register_forward_hook(_hook)
                outputs = self.model(images)
                handle.remove()
                feats = hook_out['feat'].view(hook_out['feat'].size(0), -1)
                proj = self.projection_head(feats)
                sc_loss = self.supcon_criterion(proj, labels)
                ce_loss = self.criterion(outputs, labels)
                if loss_fn == "supcon":
                    loss = sc_loss
                else:  # combined
                    loss = (1 - alpha) * ce_loss + alpha * sc_loss
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            if self.config.training.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip_norm
                )
            self.optimizer.step()
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model.
        
        Args:
            val_loader: Validation DataLoader
            
        Returns:
            Tuple of (avg_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def train_single_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        epoch_num: int
    ) -> EpochMetrics:
        """
        Train exactly one epoch and return metrics.
        
        This method is used by the interactive dashboard for step-by-step
        training with state updates between epochs.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            epoch_num: Current epoch number (1-indexed)
            
        Returns:
            EpochMetrics with training results
        """
        # Unfreeze backbone once the warmup/freeze period is over.
        # Also rebuild optimizer + scheduler so backbone params get proper Adam
        # state initialized from scratch (frozen params have no optimizer state).
        freeze_epochs = self.config.training.freeze_backbone_epochs
        if self._backbone_frozen and epoch_num > freeze_epochs:
            self.unfreeze_backbone()
            self._backbone_frozen = False
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler(skip_warmup=True)
            logger.info(f"Epoch {epoch_num}: backbone unfrozen, optimizer rebuilt with discriminative LR")

        train_loss, train_acc = self.train_epoch(train_loader)

        val_loss, val_acc = None, None
        if val_loader is not None:
            val_loss, val_acc = self.validate(val_loader)
        
        self.history["epoch"].append(epoch_num)
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_acc)
        
        if val_acc is not None:
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch_num
                self.patience_counter = 0

                if self.config.checkpoint.save_best_model:
                    self._save_checkpoint(epoch_num, is_best=True)
            else:
                self.patience_counter += 1

        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_acc is not None:
                    self.scheduler.step(val_acc)
            else:
                self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]['lr']
        
        return EpochMetrics(
            epoch=epoch_num,
            train_loss=train_loss,
            train_accuracy=train_acc,
            val_loss=val_loss,
            val_accuracy=val_acc,
            learning_rate=current_lr
        )
    
    def should_stop_early(self) -> bool:
        """
        Check if early stopping criteria is met.
        
        Returns:
            True if training should stop
        """
        return self.patience_counter >= self.config.training.early_stopping_patience
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Train model for configured number of epochs (batch mode).
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional)
            
        Returns:
            Dict with training summary
        """
        epochs = self.config.training.epochs
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            metrics = self.train_single_epoch(train_loader, val_loader, epoch)
            
            if val_loader is not None:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {metrics.train_loss:.4f}, Train Acc: {metrics.train_accuracy:.4f} | "
                    f"Val Loss: {metrics.val_loss:.4f}, Val Acc: {metrics.val_accuracy:.4f}"
                )
                
                if self.should_stop_early():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                logger.info(
                    f"Epoch {epoch}/{epochs} | "
                    f"Train Loss: {metrics.train_loss:.4f}, Train Acc: {metrics.train_accuracy:.4f}"
                )
            
            if epoch % self.config.checkpoint.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, is_best=False)
        
        logger.info(
            f"Training complete | Best Val Acc: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}"
        )
        
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "final_train_accuracy": self.history["train_accuracy"][-1] if self.history["train_accuracy"] else 0,
            "epochs_trained": len(self.history["epoch"]),
        }
    
    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        save_cm_path: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test DataLoader
            class_names: Optional list of class names
            save_cm_path: Optional path to save confusion matrix as .npy file
            
        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                _, preds = probs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_probs = np.vstack(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        
        # Compute and optionally save confusion matrix
        if save_cm_path is not None:
            cm = confusion_matrix(all_labels, all_preds)
            # Ensure parent directory exists
            save_cm_path.parent.mkdir(parents=True, exist_ok=True)
            # Save as numpy array (not in JSON - too large)
            np.save(save_cm_path, cm)
            logger.info(f"Confusion matrix saved to {save_cm_path}")
        
        # Expected Calibration Error (ECE) — 15 equal-width confidence bins
        confidences = all_probs.max(axis=1)
        correct = (np.array(all_preds) == np.array(all_labels)).astype(float)
        n_bins = 15
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_conf = confidences[mask].mean()
                bin_acc = correct[mask].mean()
                ece += mask.sum() * abs(bin_conf - bin_acc)
        ece = float(ece / len(all_labels))

        metrics = {
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "ece": ece,
        }
        if save_cm_path is not None:
            metrics["confusion_matrix_path"] = str(save_cm_path)
        
        if class_names is not None:
            p_per_class, r_per_class, f1_per_class, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=range(len(class_names)), zero_division=0
            )
            
            per_class = {}
            for i, name in enumerate(class_names):
                per_class[name] = {
                    "precision": float(p_per_class[i]),
                    "recall": float(r_per_class[i]),
                    "f1": float(f1_per_class[i]),
                }
            metrics["per_class"] = per_class
        
        logger.info(
            f"Test Results | Acc: {accuracy:.4f}, P: {precision:.4f}, "
            f"R: {recall:.4f}, F1: {f1:.4f}, ECE: {ece:.4f}"
        )
        
        return metrics
    
    def get_predictions_for_indices(
        self,
        indices: List[int],
        dataset: Dataset,
        class_names: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Get model predictions for specific dataset indices.
        
        Used for probe tracking and query visualization.
        
        Args:
            indices: List of dataset indices to predict
            dataset: Dataset to get images from
            class_names: Optional class names for readable output
            
        Returns:
            List of prediction dicts with probabilities
        """
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for idx in indices:
                image, label = dataset[idx]
                
                if image.dim() == 3:
                    image = image.unsqueeze(0)
                
                image = image.to(self.device)
                output = self.model(image)
                probs = F.softmax(output, dim=1)[0]
                
                prob_list = probs.cpu().numpy().tolist()
                predicted_idx = int(probs.argmax())
                confidence = float(probs.max())
                
                result = {
                    "index": idx,
                    "true_label": int(label),
                    "predicted_label": predicted_idx,
                    "confidence": confidence,
                    "probabilities": prob_list,
                }
                
                if class_names is not None:
                    result["true_class"] = class_names[label]
                    result["predicted_class"] = class_names[predicted_idx]
                    result["probabilities_dict"] = {
                        class_names[i]: prob_list[i]
                        for i in range(len(class_names))
                    }
                
                results.append(result)
        
        return results
    
    def get_predictions_for_loader(
        self,
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Get predictions for all samples in a DataLoader.
        
        Used for uncertainty computation in AL strategies.
        
        Args:
            data_loader: DataLoader to predict on
            class_names: Optional class names
            
        Returns:
            Tuple of (list of prediction dicts, probability matrix)
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_probs.append(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        prob_matrix = np.vstack(all_probs)
        
        results = []
        for i in range(len(all_labels)):
            probs = prob_matrix[i]
            predicted_idx = int(np.argmax(probs))
            
            result = {
                "index": i,
                "true_label": int(all_labels[i]),
                "predicted_label": predicted_idx,
                "confidence": float(probs.max()),
                "probabilities": probs.tolist(),
            }
            
            if class_names is not None:
                result["true_class"] = class_names[all_labels[i]]
                result["predicted_class"] = class_names[predicted_idx]
            
            results.append(result)
        
        return results, prob_matrix
    
    def compute_uncertainty_scores(
        self,
        prob_matrix: np.ndarray,
        method: str = "least_confidence"
    ) -> np.ndarray:
        """
        Compute uncertainty scores from probability matrix.
        
        Args:
            prob_matrix: (N, C) array of probabilities
            method: Uncertainty method (least_confidence, entropy, margin)
            
        Returns:
            (N,) array of uncertainty scores (higher = more uncertain)
        """
        if method == "least_confidence":
            return 1.0 - prob_matrix.max(axis=1)
        
        elif method == "entropy":
            return -np.sum(prob_matrix * np.log(prob_matrix + 1e-10), axis=1)
        
        elif method == "margin":
            sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
            return sorted_probs[:, 0] - sorted_probs[:, 1]
        
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "history": self.history,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
    
    def save_cycle_checkpoint(self, cycle_num: int):
        """
        Save best model for a specific AL cycle.
        
        Args:
            cycle_num: Current cycle number
        """
        path = self.checkpoint_dir / f"best_model_cycle_{cycle_num}.pth"
        
        checkpoint = {
            "cycle": cycle_num,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Cycle {cycle_num} checkpoint saved")
    
    def load_checkpoint(self, path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "best_val_accuracy" in checkpoint:
            self.best_val_accuracy = checkpoint["best_val_accuracy"]

        logger.info(f"Checkpoint loaded: {path}")

    def restore_best_model(self):
        """Reload best-checkpoint weights before evaluation (A3 fix)."""
        path = self.checkpoint_dir / "best_model.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Restored best model from {path}")
        else:
            logger.warning("No best_model.pth found — evaluating with final model state")
    
    def get_embeddings(self, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract penultimate-layer embeddings for all samples in dataloader.

        Returns:
            Tuple of (embeddings [N, D], labels [N]) as numpy arrays
        """
        return extract_features(self.model, dataloader, self.device)

    def save_training_log(self):
        """Save training history to files."""
        history_path = self.exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        log_path = self.exp_dir / "training_log.txt"
        with open(log_path, "w") as f:
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc\n")
            f.write("-" * 55 + "\n")
            
            for i, epoch in enumerate(self.history["epoch"]):
                val_loss = self.history["val_loss"][i]
                val_acc = self.history["val_accuracy"][i]
                
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
                val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "N/A"
                
                f.write(
                    f"{epoch:5d} | {self.history['train_loss'][i]:10.4f} | "
                    f"{self.history['train_accuracy'][i]:9.4f} | "
                    f"{val_loss_str:>8} | {val_acc_str:>7}\n"
                )
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history.copy()
    
    def get_training_summary(self) -> Dict:
        """Get summary of current training state."""
        return {
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
            "epochs_trained": len(self.history["epoch"]),
            "patience_counter": self.patience_counter,
            "current_lr": self.optimizer.param_groups[0]['lr'],
        }
