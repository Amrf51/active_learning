"""
worker.py â€” Worker process for Active Learning backend.

This module runs in a separate process and handles:
- Model initialization
- Training cycles
- Query sample selection
- Annotation processing

The worker communicates with the main process via:
- task_queue: Receives commands from Controller
- result_queue: Sends results and progress updates back
- events: Multiprocessing events for signaling
"""

import traceback
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from torch.utils.data import DataLoader

from config import Config, load_config
from protocol import (
    # Message types
    INIT_MODEL, RUN_CYCLE, QUERY, ANNOTATE, SHUTDOWN,
    PROGRESS_UPDATE, TRAIN_COMPLETE, QUERY_COMPLETE,
    ANNOTATE_COMPLETE, CYCLE_COMPLETE, ERROR,
    # Event names
    MODEL_READY, WORKER_INITIALIZED, TRAINING_STARTED, TRAINING_DONE,
    QUERY_STARTED, QUERY_DONE, ANNOTATION_STARTED, ANNOTATION_DONE,
    STOP_REQUESTED, WORKER_ERROR, SHUTDOWN_COMPLETE,
    # Message builders
    build_message, build_progress_update_message, build_train_complete_message,
    build_query_complete_message, build_annotate_complete_message,
    build_cycle_complete_message, build_error_message
)
from models import get_model
from data_manager import ALDataManager
from trainer import Trainer
from active_loop import ActiveLearningLoop
from dataloader import get_datasets
from strategies import get_strategy
from state import QueriedImage

logger = logging.getLogger(__name__)


def worker_loop(task_queue, result_queue, events: Dict, config_dict: Dict[str, Any]) -> None:
    """
    Main worker process loop.
    
    Runs in a separate process and handles all backend operations.
    Listens for commands from the Controller via task_queue and sends
    results back via result_queue.
    
    Args:
        task_queue: Multiprocessing queue for receiving commands
        result_queue: Multiprocessing queue for sending results
        events: Dictionary of multiprocessing.Event objects
        config_dict: Configuration dictionary (serialized Config object)
    """
    # Initialize worker state
    al_loop: Optional[ActiveLearningLoop] = None
    config: Optional[Config] = None
    
    logger.info("Worker process started")
    events[WORKER_INITIALIZED].set()
    
    try:
        while True:
            # Check for stop request
            if events[STOP_REQUESTED].is_set():
                logger.info("Stop requested, worker shutting down")
                break
            
            # Get next task (blocking with timeout)
            try:
                message = task_queue.get(timeout=1.0)
            except:
                # Timeout, check stop event again
                continue
            
            msg_type = message.get("type")
            payload = message.get("payload", {})
            
            logger.info(f"Worker received message: {msg_type}")
            
            try:
                # Handle different message types
                if msg_type == INIT_MODEL:
                    al_loop, config = _handle_init_model(
                        payload, result_queue, events
                    )
                
                elif msg_type == RUN_CYCLE:
                    _handle_run_cycle(
                        al_loop, payload, result_queue, events
                    )
                
                elif msg_type == QUERY:
                    _handle_query(
                        al_loop, payload, result_queue, events
                    )
                
                elif msg_type == ANNOTATE:
                    _handle_annotate(
                        al_loop, payload, result_queue, events
                    )
                
                elif msg_type == SHUTDOWN:
                    _handle_shutdown(result_queue, events)
                    break
                
                else:
                    logger.warning(f"Unknown message type: {msg_type}")
            
            except Exception as e:
                # Handle errors and send to main process
                error_msg = f"Error handling {msg_type}: {str(e)}"
                tb = traceback.format_exc()
                logger.error(f"{error_msg}\n{tb}")
                
                events[WORKER_ERROR].set()
                result_queue.put(build_error_message(
                    error_type=msg_type + "_error",
                    error_msg=error_msg,
                    traceback=tb
                ))
    
    except Exception as e:
        # Fatal error in worker loop
        error_msg = f"Fatal worker error: {str(e)}"
        tb = traceback.format_exc()
        logger.error(f"{error_msg}\n{tb}")
        
        events[WORKER_ERROR].set()
        result_queue.put(build_error_message(
            error_type="worker_fatal_error",
            error_msg=error_msg,
            traceback=tb
        ))
    
    finally:
        logger.info("Worker process exiting")
        events[SHUTDOWN_COMPLETE].set()


def _handle_init_model(
    payload: Dict[str, Any],
    result_queue,
    events: Dict
) -> tuple:
    """
    Handle INIT_MODEL message.
    
    Builds model, data_manager, trainer, and ActiveLearningLoop.
    
    Args:
        payload: Message payload with config dict
        result_queue: Queue for sending results
        events: Event dictionary
        
    Returns:
        Tuple of (al_loop, config)
    """
    logger.info("Initializing model and AL loop...")
    
    # Load configuration
    config_overrides = payload.get("config", {})
    config = load_config(overrides=config_overrides)
    
    # Build AL loop
    al_loop = _build_al_loop(payload, config)
    
    # Signal completion
    events[MODEL_READY].set()
    
    result_queue.put(build_message(
        "init_complete",
        {
            "model_name": config.model.name,
            "num_classes": config.model.num_classes,
            "device": config.experiment.device,
            "pool_info": al_loop.get_current_pool_info()
        }
    ))
    
    logger.info("Model initialization complete")
    
    return al_loop, config


def _build_al_loop(payload: Dict[str, Any], config: Config) -> ActiveLearningLoop:
    """
    Build ActiveLearningLoop with all dependencies.
    
    Args:
        payload: Message payload (may contain additional config)
        config: Config object
        
    Returns:
        Initialized ActiveLearningLoop
    """
    # Set up experiment directory
    exp_dir = Path(config.experiment.exp_dir) / config.experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Get datasets (returns dict with train_dataset, val_dataset, test_dataset, class_names, etc.)
    datasets = get_datasets(
        data_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        augmentation=config.data.augmentation,
        seed=config.experiment.seed
    )
    
    # Extract what we need
    train_dataset = datasets["train_dataset"]
    class_names = datasets["class_names"]
    
    # Create val/test DataLoaders from datasets
    pin = torch.cuda.is_available()
    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin
    )
    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin
    )
    
    # Set num_classes if not specified
    if config.model.num_classes is None:
        config.model.num_classes = len(class_names)
    
    # Create model
    model = get_model(
        name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        device=config.experiment.device
    )
    
    # Create data manager
    data_manager = ALDataManager(
        dataset=train_dataset,
        initial_pool_size=config.active_learning.initial_pool_size,
        seed=config.experiment.seed,
        exp_dir=exp_dir
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        exp_dir=exp_dir,
        device=config.experiment.device
    )
    
    # Get strategy function
    strategy = get_strategy(
        config.active_learning.sampling_strategy,
        config.active_learning.uncertainty_method
    )
    
    # Create ActiveLearningLoop
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
    
    logger.info(f"AL loop built: {len(class_names)} classes, "
                f"{len(train_dataset)} training samples")
    
    return al_loop


def _handle_run_cycle(
    al_loop: Optional[ActiveLearningLoop],
    payload: Dict[str, Any],
    result_queue,
    events: Dict
) -> None:
    """
    Handle RUN_CYCLE message.
    
    Executes one complete AL cycle with progress reporting.
    
    Args:
        al_loop: ActiveLearningLoop instance
        payload: Message payload with cycle_num
        result_queue: Queue for sending results
        events: Event dictionary
    """
    if al_loop is None:
        raise RuntimeError("AL loop not initialized")
    
    cycle_num = payload.get("cycle_num", 1)
    
    logger.info(f"Starting cycle {cycle_num}")
    events[TRAINING_STARTED].set()
    
    # Prepare cycle
    prep_info = al_loop.prepare_cycle(cycle_num)
    result_queue.put(build_message("cycle_prepared", prep_info))
    
    # Training loop
    epochs = al_loop.config.training.epochs
    for epoch in range(1, epochs + 1):
        # Check for stop request
        if events[STOP_REQUESTED].is_set():
            logger.info("Stop requested during training")
            result_queue.put(build_message("training_stopped", {
                "cycle": cycle_num,
                "epoch": epoch
            }))
            return
        
        # Train one epoch
        metrics = al_loop.train_single_epoch(epoch)
        
        # Send progress update
        result_queue.put(build_progress_update_message(
            stage="training",
            current=epoch,
            total=epochs,
            details=metrics.to_dict()
        ))
        
        # Check early stopping
        if al_loop.should_stop_early():
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Training complete
    events[TRAINING_DONE].set()
    
    # Run evaluation
    test_metrics = al_loop.run_evaluation()
    
    result_queue.put(build_train_complete_message(test_metrics))
    
    # Finalize cycle
    cycle_metrics = al_loop.finalize_cycle(test_metrics)
    
    result_queue.put(build_cycle_complete_message(
        cycle_num=cycle_num,
        metrics=cycle_metrics.model_dump()
    ))
    
    logger.info(f"Cycle {cycle_num} complete")


def _handle_query(
    al_loop: Optional[ActiveLearningLoop],
    payload: Dict[str, Any],
    result_queue,
    events: Dict
) -> None:
    """
    Handle QUERY message.
    
    Applies AL strategy to select samples for annotation.
    
    Args:
        al_loop: ActiveLearningLoop instance
        payload: Message payload (may contain query_size override)
        result_queue: Queue for sending results
        events: Event dictionary
    """
    if al_loop is None:
        raise RuntimeError("AL loop not initialized")
    
    logger.info("Starting query")
    events[QUERY_STARTED].set()
    
    # Query samples
    queried_images = al_loop.query_samples()
    
    # Convert to dicts for serialization
    queried_dicts = [_queried_to_dict(img) for img in queried_images]
    
    # Send results
    result_queue.put(build_query_complete_message(queried_dicts))
    
    events[QUERY_DONE].set()
    logger.info(f"Query complete: {len(queried_images)} samples")


def _queried_to_dict(img: QueriedImage) -> Dict[str, Any]:
    """
    Convert QueriedImage to dictionary for serialization.
    
    Args:
        img: QueriedImage object
        
    Returns:
        Dictionary representation
    """
    return img.to_dict()


def _handle_annotate(
    al_loop: Optional[ActiveLearningLoop],
    payload: Dict[str, Any],
    result_queue,
    events: Dict
) -> None:
    """
    Handle ANNOTATE message.
    
    Processes user annotations and updates pools.
    
    Args:
        al_loop: ActiveLearningLoop instance
        payload: Message payload with annotations
        result_queue: Queue for sending results
        events: Event dictionary
    """
    if al_loop is None:
        raise RuntimeError("AL loop not initialized")
    
    annotations = payload.get("annotations", [])
    
    logger.info(f"Processing {len(annotations)} annotations")
    events[ANNOTATION_STARTED].set()
    
    # Process annotations
    result = al_loop.receive_annotations(annotations)
    
    # Send results
    result_queue.put(build_annotate_complete_message(
        num_annotated=result["moved_count"]
    ))
    
    # Also send detailed results
    result_queue.put(build_message("annotation_details", result))
    
    events[ANNOTATION_DONE].set()
    logger.info("Annotations processed")


def _handle_shutdown(result_queue, events: Dict) -> None:
    """
    Handle SHUTDOWN message.
    
    Performs graceful cleanup and termination.
    
    Args:
        result_queue: Queue for sending results
        events: Event dictionary
    """
    logger.info("Shutdown requested")
    
    # Perform any cleanup here
    # (e.g., save final state, close files, etc.)
    
    result_queue.put(build_message("shutdown_complete", {}))
    events[SHUTDOWN_COMPLETE].set()
    
    logger.info("Shutdown complete")
