"""Training loop and model management."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for model training and evaluation."""
    
    def __init__(self, model: nn.Module, config, exp_dir: Path, device: str = "cuda"):
        """Initialize trainer.
        
        Args:
            model: PyTorch model
            config: Config object
            exp_dir: Experiment directory to save results
            device: Device to train on
        """
        self.model = model
        self.config = config
        self.exp_dir = Path(exp_dir)
        self.device = device
        
        # Setup directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Setup optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.scheduler = None
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "epoch": []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized. Device: {device}, Exp dir: {exp_dir}")
    
    def _get_optimizer(self):
        """Get optimizer based on config."""
        optimizer_name = self.config.training.optimizer.lower()
        lr = self.config.training.learning_rate
        wd = self.config.training.weight_decay
        
        if optimizer_name == "adam":
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        elif optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training dataloader
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Log progress
            if (batch_idx + 1) % self.config.checkpoint.log_every_n_batches == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model.
        
        Args:
            val_loader: Validation dataloader
            
        Returns:
            Tuple of (loss, accuracy)
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
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train model for specified epochs.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
        """
        logger.info(f"Starting training for {self.config.training.epochs} epochs...")
        
        for epoch in range(self.config.training.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Log to history
            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_acc)
            
            logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs} | "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint.save_every_n_epochs == 0:
                self._save_checkpoint(epoch + 1)
            
            # Save best model
            if self.config.checkpoint.save_best_model and val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self._save_checkpoint(epoch + 1, is_best=True)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info(f"Training completed. Best val accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
    
    def evaluate(self, test_loader: DataLoader, class_names: list = None) -> Dict:
        """Evaluate model on test set.
        
        Args:
            test_loader: Test dataloader
            class_names: List of class names for per-class metrics
            
        Returns:
            Dictionary with metrics
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
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        
        metrics = {
            "test_accuracy": float(accuracy),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
            "best_val_accuracy": float(self.best_val_accuracy),
            "best_epoch": int(self.best_epoch),
        }
        
        # Per-class metrics
        if class_names is not None:
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                all_labels, all_preds, labels=range(len(class_names)), zero_division=0
            )
            
            per_class_metrics = {}
            for i, class_name in enumerate(class_names):
                per_class_metrics[class_name] = {
                    "precision": float(precision_per_class[i]),
                    "recall": float(recall_per_class[i]),
                    "f1": float(f1_per_class[i]),
                }
            
            metrics["per_class"] = per_class_metrics
        
        logger.info(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Epoch number
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def save_training_log(self):
        """Save training history as JSON and text."""
        # Save as JSON
        history_path = self.exp_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        
        # Save as text log
        log_path = self.exp_dir / "training_log.txt"
        with open(log_path, "w") as f:
            f.write("Epoch | Train Loss | Train Acc | Val Loss | Val Acc\n")
            f.write("-" * 60 + "\n")
            for i, epoch in enumerate(self.history["epoch"]):
                f.write(f"{epoch:5d} | {self.history['train_loss'][i]:10.4f} | "
                       f"{self.history['train_accuracy'][i]:9.4f} | "
                       f"{self.history['val_loss'][i]:8.4f} | "
                       f"{self.history['val_accuracy'][i]:7.4f}\n")
        
        logger.info(f"Training logs saved to {log_path}")