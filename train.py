"""
Training script - supports both standard and Active Learning modes.

Usage:
    Standard training:
        python train.py --config config/base_config.yaml --exp-name baseline
    
    Active Learning:
        python train.py --config config/al_config.yaml --exp-name al_uncertainty
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataloader import get_datasets, get_dataset_info
from src.models import get_model, get_model_info
from src.trainer import Trainer
from src.data_manager import ALDataManager
from src.active_loop import ActiveLearningLoop
from src.strategies import get_strategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_standard_training(
    config: Config,
    exp_dir: Path,
    device: str
) -> Dict:
    """
    Run standard training (no Active Learning).
    
    Args:
        config: Configuration object
        exp_dir: Experiment directory
        device: Device to train on
        
    Returns:
        Dict with training results
    """
    logger.info("="*60)
    logger.info("MODE: STANDARD TRAINING")
    logger.info("="*60)
    
    # Load datasets
    datasets = get_datasets(
        data_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        augmentation=config.data.augmentation,
        seed=config.training.seed
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train: {len(datasets['train_dataset'])} samples")
    logger.info(f"Val: {len(datasets['val_dataset'])} samples")
    logger.info(f"Test: {len(datasets['test_dataset'])} samples")
    
    # Load model
    model = get_model(config.model, device=device)
    model_info = get_model_info(model)
    logger.info(f"Model: {config.model.name} ({model_info['trainable_parameters']:,} params)")
    
    # Initialize trainer
    trainer = Trainer(model, config, exp_dir, device=device)
    
    # Train
    train_summary = trainer.train(train_loader, val_loader)
    trainer.save_training_log()
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, class_names=datasets["class_names"])
    
    # Save metrics
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    # Build results summary
    results = {
        "mode": "standard",
        "model": config.model.name,
        "train_summary": train_summary,
        "test_metrics": test_metrics,
        "splits_info": datasets["splits_info"],
        "model_info": model_info,
    }
    
    return results


def run_active_learning(
    config: Config,
    exp_dir: Path,
    device: str
) -> Dict:
    """
    Run Active Learning training.
    
    Args:
        config: Configuration object
        exp_dir: Experiment directory
        device: Device to train on
        
    Returns:
        Dict with AL results
    """
    al_config = config.active_learning
    
    logger.info("="*60)
    logger.info("MODE: ACTIVE LEARNING")
    logger.info(f"Strategy: {al_config.sampling_strategy}")
    logger.info(f"Cycles: {al_config.num_cycles}")
    logger.info(f"Initial pool: {al_config.initial_pool_size}")
    logger.info(f"Query batch: {al_config.batch_size_al}")
    logger.info("="*60)
    
    # Load datasets
    datasets = get_datasets(
        data_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        augmentation=config.data.augmentation,
        seed=config.training.seed
    )
    
    # Save splits info
    splits_file = exp_dir / "data_splits.json"
    with open(splits_file, "w") as f:
        json.dump(datasets["splits_info"], f, indent=2)
    
    # Create fixed val and test loaders (never change during AL)
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    logger.info(f"Train pool: {len(datasets['train_dataset'])} samples (for AL)")
    logger.info(f"Val: {len(datasets['val_dataset'])} samples (fixed)")
    logger.info(f"Test: {len(datasets['test_dataset'])} samples (fixed)")
    
    # Initialize AL Data Manager (only training data, not val/test)
    data_manager = ALDataManager(
        dataset=datasets["train_dataset"],
        initial_pool_size=al_config.initial_pool_size,
        seed=config.training.seed,
        exp_dir=exp_dir
    )
    
    # Load model
    model = get_model(config.model, device=device)
    model_info = get_model_info(model)
    logger.info(f"Model: {config.model.name} ({model_info['trainable_parameters']:,} params)")
    
    # Initialize trainer
    trainer = Trainer(model, config, exp_dir, device=device)
    
    # Get sampling strategy
    strategy = get_strategy(
        al_config.sampling_strategy,
        al_config.uncertainty_method
    )
    logger.info(f"Strategy function: {strategy.__name__}")
    
    # Initialize AL Loop (the orchestrator)
    al_loop = ActiveLearningLoop(
        trainer=trainer,
        data_manager=data_manager,
        strategy=strategy,
        val_loader=val_loader,
        test_loader=test_loader,
        exp_dir=exp_dir,
        config=config
    )
    
    # Run all cycles
    cycle_results = al_loop.run_all_cycles()
    
    # Build results summary
    results = {
        "mode": "active_learning",
        "strategy": al_config.sampling_strategy,
        "model": config.model.name,
        "num_cycles": al_config.num_cycles,
        "initial_pool_size": al_config.initial_pool_size,
        "batch_size_al": al_config.batch_size_al,
        "cycle_results": cycle_results,
        "best_cycle": al_loop.get_best_cycle(),
        "splits_info": datasets["splits_info"],
        "model_info": model_info,
    }
    
    return results


def main(args):
    """Main entry point."""
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    logger.info(f"Loading config: {config_path}")
    config = Config.from_yaml(str(config_path))
    
    # Determine experiment directory
    if Path(args.config).parent.name.startswith(("exp_", "experiment")):
        # Config is already in an experiment directory
        exp_dir = Path(args.config).parent
        logger.info(f"Using existing experiment dir: {exp_dir}")
    else:
        # Create new experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if config.active_learning.enabled:
            exp_name = f"{args.exp_name}_{config.active_learning.sampling_strategy}_{timestamp}"
        else:
            exp_name = f"{args.exp_name}_{timestamp}"
        
        exp_dir = Path("experiments") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created experiment dir: {exp_dir}")
        
        # Save config copy with metadata
        config.save_to(str(exp_dir / "config.yaml"), include_metadata=True)
    
    # Add file handler for logging
    file_handler = logging.FileHandler(exp_dir / "training.log")
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(file_handler)
    
    # Log full config
    logger.info(f"Config:\n{json.dumps(config.to_dict(), indent=2)}")
    
    # Get dataset info
    dataset_info = get_dataset_info(config.data.data_dir)
    logger.info(f"Dataset: {dataset_info['total_images']} images, {dataset_info['num_classes']} classes")
    logger.info(f"Classes: {dataset_info['class_names']}")
    logger.info(f"Distribution: {dataset_info['class_counts']}")
    
    try:
        # Run appropriate mode
        if config.active_learning.enabled:
            results = run_active_learning(config, exp_dir, device)
        else:
            results = run_standard_training(config, exp_dir, device)
        
        # Add metadata
        results["experiment_name"] = exp_name
        results["timestamp"] = timestamp
        results["config"] = config.to_dict()
        results["dataset_info"] = dataset_info
        
        # Save final results
        results_file = exp_dir / "results_summary.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nExperiment complete!")
        logger.info(f"Results saved to: {exp_dir}")
        
        # Print key metrics
        if config.active_learning.enabled:
            best = results.get("best_cycle", {})
            logger.info(f"Best test accuracy: {best.get('test_accuracy', 0):.4f} (cycle {best.get('cycle', 0)})")
        else:
            test_acc = results.get("test_metrics", {}).get("test_accuracy", 0)
            logger.info(f"Test accuracy: {test_acc:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train vehicle classification model (standard or active learning)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--exp-name",
        default="experiment",
        help="Experiment name prefix"
    )
    
    args = parser.parse_args()
    main(args)