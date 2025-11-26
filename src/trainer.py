"""
Trainer - Handles model training and evaluation.

This is a "dumb" trainer that only knows how to:
- Train on a given DataLoader
- Validate on a given DataLoader
- Evaluate on a test set
- Save/load checkpoints

It does NOT know about Active Learning, pools, or sampling strategies.
The ActiveLearningLoop orchestrator commands the Trainer when needed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
from typing import Dict, Tuple, Optional, List
import logging

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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
    
    def reset_model_weights(self):
        """
        Reset model to initial state (random weights for non-pretrained layers).
        
        Called by ActiveLearningLoop at the start of each cycle for
        from-scratch training comparison.
        """
        logger.info("Resetting model weights...")
        
        for module in self.model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        
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
        
        logger.info("Model weights reset complete")
    
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
            
            if (batch_idx + 1) % self.config.checkpoint.log_every_n_batches == 0:
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Train model for configured number of epochs.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader (optional, enables early stopping)
            
        Returns:
            Dict with training summary
        """
        epochs = self.config.training.epochs
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
                # Early stopping check
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    
                    if self.config.checkpoint.save_best_model:
                        self._save_checkpoint(epoch + 1, is_best=True)
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                self.history["val_loss"].append(None)
                self.history["val_accuracy"].append(None)
                
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.checkpoint.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1, is_best=False)
        
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
        class_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test DataLoader
            class_names: Optional list of class names for per-class metrics
            
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
        logger.debug(f"Checkpoint saved: {path}")
    
    def save_cycle_checkpoint(self, cycle_num: int):
        """
        Save best model for a specific AL cycle.
        
        Called by ActiveLearningLoop after each cycle completes.
        
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
        logger.info(f"Cycle {cycle_num} best model saved: {path}")
    
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
        # JSON format
        history_path = self.exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Text format
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
        
        logger.info(f"Training logs saved to {self.exp_dir}")
    
    def get_history(self) -> Dict:
        """Get training history."""
        return self.history.copy()