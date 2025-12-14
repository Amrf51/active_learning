#!/usr/bin/env python3
"""
Active Learning Worker Process

Background worker that executes AL cycles based on commands from the dashboard.
Implements the Controller-Worker pattern where the dashboard (controller) sends
commands and the worker executes the heavy computation.

Usage:
    python run_worker.py --experiment-id exp_001

The worker will:
1. Initialize AL components from experiment state
2. Poll for commands from dashboard
3. Execute training, evaluation, and querying
4. Update state with progress and results
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.state import (
    StateManager, 
    ExperimentPhase, 
    Command,
    EpochMetrics,
    CycleMetrics
)
from src.config import Config
from src.active_loop import ActiveLearningLoop
from src.trainer import Trainer
from src.data_manager import ALDataManager
from src.dataloader import get_dataloaders
from src.models import get_model
from src.strategies import get_strategy


def setup_logging(experiment_id: str, log_level: str = "INFO") -> None:
    """
    Set up logging for the worker process.
    
    Args:
        experiment_id: Experiment ID for log file naming
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create logs directory
    log_dir = Path("experiments") / experiment_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"worker_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Worker logging initialized for experiment {experiment_id}")
    logger.info(f"Log file: {log_file}")


def initialize_from_state(state_manager: StateManager) -> tuple:
    """
    Initialize AL components from experiment state.
    
    Args:
        state_manager: StateManager instance
        
    Returns:
        Tuple of (al_loop, trainer, data_manager, config)
        
    Raises:
        RuntimeError: If state is not properly configured
    """
    logger = logging.getLogger(__name__)
    
    # Read current state
    state = state_manager.read_state()
    
    if state.config is None:
        raise RuntimeError("Experiment state does not contain configuration")
    
    # Create config object from state
    config = Config()
    
    # Map state config to Config object
    config.model.name = state.config.model_name
    config.model.pretrained = state.config.pretrained
    config.model.num_classes = state.config.num_classes
    
    config.training.epochs = state.config.epochs_per_cycle
    config.training.batch_size = state.config.batch_size
    config.training.learning_rate = state.config.learning_rate
    config.training.optimizer = state.config.optimizer
    config.training.weight_decay = state.config.weight_decay
    config.training.early_stopping_patience = state.config.early_stopping_patience
    config.training.seed = state.config.seed
    
    config.active_learning.num_cycles = state.config.num_cycles
    config.active_learning.sampling_strategy = state.config.sampling_strategy
    config.active_learning.uncertainty_method = state.config.uncertainty_method
    config.active_learning.initial_pool_size = state.config.initial_pool_size
    config.active_learning.batch_size_al = state.config.batch_size_al
    config.active_learning.reset_mode = state.config.reset_mode
    
    config.data.data_dir = state.config.data_dir
    config.data.val_split = state.config.val_split
    config.data.test_split = state.config.test_split
    config.data.augmentation = state.config.augmentation
    
    logger.info("Configuration loaded from state:")
    logger.info(f"  Model: {config.model.name}")
    logger.info(f"  Strategy: {config.active_learning.sampling_strategy}")
    logger.info(f"  Cycles: {config.active_learning.num_cycles}")
    logger.info(f"  Data dir: {config.data.data_dir}")
    
    # Initialize data loaders
    train_loader, val_loader, test_loader, class_names = get_dataloaders(config)
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(config.model, device=device)
    
    # Initialize trainer
    exp_dir = state_manager.experiment_dir
    trainer = Trainer(model, config, exp_dir, device=device)
    
    # Initialize data manager
    data_manager = ALDataManager(
        train_loader.dataset,
        initial_pool_size=config.active_learning.initial_pool_size,
        seed=config.training.seed
    )
    
    # Load existing pool state if available
    pool_state_file = exp_dir / "al_pool_state.json"
    if pool_state_file.exists():
        data_manager.load_state(pool_state_file)
        logger.info("Loaded existing pool state")
    
    # Initialize AL strategy
    strategy = get_strategy(config.active_learning.sampling_strategy)
    
    # Initialize ActiveLearningLoop
    al_loop = ActiveLearningLoop(
        trainer=trainer,
        data_manager=data_manager,
        strategy=strategy,
        val_loader=val_loader,
        test_loader=test_loader,
        exp_dir=exp_dir,
        config=config,
        class_names=class_names
    )
    
    logger.info("AL components initialized successfully")
    
    return al_loop, trainer, data_manager, config


def main():
    """Main worker process entry point."""
    parser = argparse.ArgumentParser(
        description="Active Learning Worker Process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_worker.py --experiment-id exp_001
    python run_worker.py --experiment-id uncertainty_test --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment ID to process"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.experiment_id, args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate experiment directory
    exp_dir = Path(args.experiments_dir) / args.experiment_id
    if not exp_dir.exists():
        logger.error(f"Experiment directory not found: {exp_dir}")
        sys.exit(1)
    
    # Initialize state manager
    try:
        state_manager = StateManager(exp_dir)
        
        if not state_manager.state_exists():
            logger.error(f"Experiment state file not found in {exp_dir}")
            sys.exit(1)
        
        logger.info(f"Worker started for experiment: {args.experiment_id}")
        logger.info(f"Experiment directory: {exp_dir}")
        
        # Initialize AL components
        al_loop, trainer, data_manager, config = initialize_from_state(state_manager)
        
        # Update worker PID in state
        state_manager.update_state(
            worker_pid=os.getpid(),
            last_heartbeat=datetime.now()
        )
        
        logger.info(f"Worker process PID: {os.getpid()}")
        logger.info("Worker initialization complete - entering command loop")
        
        # Import the command loop implementation (will be implemented in next subtask)
        from worker_command_loop import run_command_loop
        
        # Start command loop
        run_command_loop(state_manager, al_loop, trainer, data_manager, config)
        
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Worker failed with error: {e}", exc_info=True)
        
        # Try to update state with error
        try:
            state_manager.update_state(
                phase=ExperimentPhase.ERROR,
                error_message=str(e)
            )
        except:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    # Import torch here to avoid import issues
    import torch
    main()