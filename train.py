"""Main training script - simple runner that delegates to trainer.py."""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import torch

from src.config import Config
from src.dataloader import get_dataloaders, get_dataset_info
from src.models import get_model, get_model_info
from src.trainer import Trainer
from src.active_learning import ActiveLearningOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    """Main training function.

    Args:
        args: Command line arguments
    """
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Save config to experiment directory (reproducibility)
    config.save_to(exp_dir / "config.yaml")
    logger.info("Config saved to experiment directory")
    
    # Setup file handler for experiment logs
    log_file = exp_dir / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    # Log config
    logger.info(f"Configuration:\n{json.dumps(config.to_dict(), indent=2)}")
    
    try:
        # Get dataset info
        logger.info("Loading dataset information...")
        dataset_info = get_dataset_info(config.data.data_dir)
        logger.info(f"Dataset info: {dataset_info}")

        # Load model
        logger.info(f"Loading model: {config.model.name}")
        model = get_model(config.model, device=device)
        model_info = get_model_info(model)
        logger.info(f"Model info: {model_info}")

        if config.active_learning.enabled:
            logger.info("Active Learning enabled. Initializing orchestrator...")
            orchestrator = ActiveLearningOrchestrator(config, device, exp_dir)
            orchestrator.initialize_pools(config.active_learning.initial_pool_size)

            val_loader = orchestrator.get_val_loader(config.training.batch_size)
            test_loader = orchestrator.get_test_loader(config.training.batch_size)

            cycle_metrics = {}
            for cycle in range(1, config.active_learning.num_cycles + 1):
                if not orchestrator.labeled_indices:
                    raise RuntimeError("Labeled pool is empty. Cannot start training cycle.")

                cycle_dir = exp_dir / f"cycle_{cycle}"
                logger.info(f"Starting cycle {cycle} -> {cycle_dir}")
                trainer = Trainer(model, config, cycle_dir, device=device)

                resume_ckpt = None
                if cycle > 1:
                    prev_best = exp_dir / f"cycle_{cycle - 1}" / "checkpoints" / "best_model.pth"
                    if prev_best.exists():
                        resume_ckpt = prev_best

                train_loader = orchestrator.get_labeled_loader(config.training.batch_size)
                artifacts = trainer.train(
                    train_loader,
                    val_loader,
                    resume_from=resume_ckpt,
                    reset_history=True,
                )
                trainer.save_training_log()

                metrics = trainer.evaluate(test_loader, class_names=dataset_info["class_names"])
                metrics_file = cycle_dir / "metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)

                summary = {
                    "experiment_name": exp_name,
                    "timestamp": timestamp,
                    "config": config.to_dict(),
                    "dataset_info": dataset_info,
                    "model_info": model_info,
                    "metrics": metrics,
                    "training_artifacts": artifacts,
                    "pool_summary": orchestrator.summarize(),
                }

                summary_file = cycle_dir / "results_summary.json"
                with open(summary_file, "w") as f:
                    json.dump(summary, f, indent=2)

                orchestrator.save_cycle_state(cycle_dir, cycle)
                cycle_metrics[f"cycle_{cycle}"] = metrics

                logger.info(f"Cycle {cycle} complete. Metrics saved to {metrics_file}")

                if cycle < config.active_learning.num_cycles and orchestrator.unlabeled_indices:
                    queried = orchestrator.query_samples(model, config.active_learning.batch_size_al)
                    orchestrator.move_to_labeled(queried)
                else:
                    logger.info("No further queries or cycles remaining.")

            with open(exp_dir / "cycle_metrics.json", "w") as f:
                json.dump(cycle_metrics, f, indent=2)
            logger.info(f"All cycles complete. Cycle metrics saved to {exp_dir / 'cycle_metrics.json'}")
        else:
            # Load dataloaders
            logger.info("Creating dataloaders...")
            train_loader, val_loader, test_loader = get_dataloaders(
                config.data,
                batch_size=config.training.batch_size
            )

            # Initialize trainer
            logger.info("Initializing trainer...")
            trainer = Trainer(model, config, exp_dir, device=device)

            # Train model
            logger.info("Starting training...")
            trainer.train(train_loader, val_loader)

            # Save training logs
            trainer.save_training_log()

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            metrics = trainer.evaluate(test_loader, class_names=dataset_info["class_names"])

            # Save metrics
            metrics_file = exp_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_file}")

            # Prepare final results summary
            results_summary = {
                "experiment_name": exp_name,
                "timestamp": timestamp,
                "config": config.to_dict(),
                "dataset_info": dataset_info,
                "model_info": model_info,
                "metrics": metrics,
            }

            summary_file = exp_dir / "results_summary.json"
            with open(summary_file, "w") as f:
                json.dump(results_summary, f, indent=2)

            logger.info(f"✅ Training completed successfully!")
            logger.info(f"📊 Results saved to {exp_dir}")
            logger.info(f"   - metrics.json")
            logger.info(f"   - results_summary.json")
            logger.info(f"   - training_history.json")
            logger.info(f"   - training_log.txt")
            logger.info(f"   - checkpoints/")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model on vehicle classification dataset"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--exp-name",
        default="experiment",
        help="Experiment name (prefix for results folder)"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    main(args)