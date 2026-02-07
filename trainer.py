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
from .models import get_model
from .state import EpochMetrics

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
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        
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
        
        logger.info(f"Trainer initialized | Device: {device}")
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        
        if name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def reset_model_weights(self, mode: str = "pretrained"):
        """
        Reset model weights based on specified mode.
        
        Args:
            mode: Reset strategy
                - "pretrained": Reload fresh ImageNet weights
                - "head_only": Keep backbone, reset classification head only
                - "none": No reset, continue from current state
        """
        if mode == "none":
            logger.info("Reset mode: none - keeping current weights")
            self.optimizer = self._create_optimizer()
            self._reset_tracking()
            return
        
        if mode == "head_only":
            logger.info("Reset mode: head_only - resetting classification head")
            if hasattr(self.model, 'fc'):
                self.model.fc.reset_parameters()
            elif hasattr(self.model, 'classifier'):
                if isinstance(self.model.classifier, nn.Sequential):
                    for layer in self.model.classifier:
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                else:
                    self.model.classifier.reset_parameters()
            elif hasattr(self.model, 'head'):
                if isinstance(self.model.head, nn.Sequential):
                    for layer in self.model.head:
                        if hasattr(layer, 'reset_parameters'):
                            layer.reset_parameters()
                else:
                    self.model.head.reset_parameters()
            
            self.optimizer = self._create_optimizer()
            self._reset_tracking()
            return
        
        if mode == "pretrained":
            logger.info("Reset mode: pretrained - reloading ImageNet weights")
            model_name = self.config.model.name
            num_classes = self.config.model.num_classes
            pretrained = self.config.model.pretrained
            self.model = get_model(
                name=model_name,
                num_classes=num_classes,
                pretrained=pretrained,
                device=self.device
            )
            self.optimizer = self._create_optimizer()
            self._reset_tracking()
            logger.info("Model reset to pretrained state complete")
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
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
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
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
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
        
        metrics = {
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }
        
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
            f"R: {recall:.4f}, F1: {f1:.4f}"
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