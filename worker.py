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

# NOTE: All project-local imports (models, trainer, data_manager, etc.) are
# deferred to inside function bodies.  This is required because Python's
# multiprocessing "spawn" method re-imports the worker module in the child
# process.  On FUSE-mounted filesystems (university clusters) the child
# process may not yet have valid FUSE credentials when top-level imports
# execute, causing PermissionError (Errno 1).  Deferring imports to
# function bodies ensures they only run after the child is fully alive.

logger = logging.getLogger(__name__)


def _dict_to_config(config_dict: Dict[str, Any]):
    """
    Rebuild Config object from dictionary.
    
    Used by worker to reconstruct config without file I/O.
    """
    from config import (
        Config, ExperimentConfig, DataConfig, ModelConfig,
        TrainingConfig, ALConfig, CheckpointConfig, LoggingConfig
    )
    
    return Config(
        experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        data=DataConfig(**config_dict.get("data", {})),
        model=ModelConfig(**config_dict.get("model", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        active_learning=ALConfig(**config_dict.get("active_learning", {})),
        checkpoint=CheckpointConfig(**config_dict.get("checkpoint", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
    )


def worker_main(task_queue, result_queue, events: Dict, config_dict: Dict[str, Any]) -> None:
    """
    Worker process main function.
    
    Builds all AL components at startup (eager initialization),
    then enters command loop to handle messages from controller.
    
    Args:
        task_queue: Multiprocessing queue for receiving commands
        result_queue: Multiprocessing queue for sending results
        events: Dictionary of multiprocessing.Event objects
        config_dict: Configuration dictionary (serialized Config object)
    """
    # Deferred imports — must happen inside the child process, not at module level
    from protocol import (
        RUN_CYCLE, QUERY, ANNOTATE, SHUTDOWN,
        WORKER_INITIALIZED, STOP_REQUESTED, WORKER_ERROR, SHUTDOWN_COMPLETE,
        build_message, build_error_message,
    )

    logger.info("Worker process starting...")
    
    try:
        # 1. Rebuild config from dict (no file I/O needed)
        config = _dict_to_config(config_dict)
        logger.info(f"Config loaded: {config.experiment.name}")
        
        # 2. Build all AL components (one-time setup)
        al_loop = _build_al_loop({}, config)
        logger.info("AL loop built successfully")
        
        # 3. Signal ready to main process
        events[WORKER_INITIALIZED].set()
        logger.info("Worker initialized and ready")
        
    except Exception as e:
        error_msg = f"Worker initialization failed: {str(e)}"
        tb = traceback.format_exc()
        logger.error(f"{error_msg}\n{tb}")
        events[WORKER_ERROR].set()
        result_queue.put(build_error_message(
            error_type="worker_init_error",
            error_msg=error_msg,
            traceback=tb
        ))
        events[WORKER_INITIALIZED].set()  # Unblock main process
        return  # Exit worker
    
    # 4. Enter command loop
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
                # INIT_MODEL removed - AL components built at startup

                if msg_type == RUN_CYCLE:
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


def _build_al_loop(payload: Dict[str, Any], config) -> 'ActiveLearningLoop':
    """
    Build ActiveLearningLoop with all dependencies.
    
    Args:
        payload: Message payload (may contain additional config)
        config: Config object
        
    Returns:
        Initialized ActiveLearningLoop
    """
    import torch
    from torch.utils.data import DataLoader
    from models import get_model
    from data_manager import ALDataManager
    from trainer import Trainer
    from active_loop import ActiveLearningLoop
    from dataloader import get_datasets
    from strategies import get_strategy
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
    al_loop,
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
    from protocol import (
        TRAINING_STARTED, TRAINING_DONE, STOP_REQUESTED,
        build_message, build_progress_update_message,
        build_train_complete_message, build_cycle_complete_message,
    )
    if al_loop is None:
        raise RuntimeError("AL loop not initialized")
    
    cycle_num = payload.get("cycle_num", 1)
    
    logger.info(f"Starting cycle {cycle_num}")
    events[TRAINING_STARTED].set()
    
    # Prepare cycle
    prep_info = al_loop.prepare_cycle(cycle_num)
    prep_info["class_names"] = al_loop.class_names if hasattr(al_loop, 'class_names') else []
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

    # Finalize cycle (computes pool sizes)
    cycle_metrics = al_loop.finalize_cycle(test_metrics)

    # CRITICAL: Send CYCLE_COMPLETE first so controller has pool sizes
    # before processing TRAIN_COMPLETE (which triggers auto-dispatch query)
    result_queue.put(build_cycle_complete_message(
        cycle_num=cycle_num,
        metrics=cycle_metrics.model_dump()
    ))

    result_queue.put(build_train_complete_message(test_metrics))

    logger.info(f"Cycle {cycle_num} complete")


def _handle_query(
    al_loop,
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
    from protocol import QUERY_STARTED, QUERY_DONE, build_query_complete_message
    from state import QueriedImage
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


def _queried_to_dict(img) -> Dict[str, Any]:
    """
    Convert QueriedImage to dictionary for serialization.
    
    Args:
        img: QueriedImage object
        
    Returns:
        Dictionary representation
    """
    return img.to_dict()


def _handle_annotate(
    al_loop,
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
    from protocol import (
        ANNOTATION_STARTED, ANNOTATION_DONE,
        build_message, build_annotate_complete_message,
    )
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
    from protocol import SHUTDOWN_COMPLETE, build_message
    logger.info("Shutdown requested")
    
    # Perform any cleanup here
    # (e.g., save final state, close files, etc.)
    
    result_queue.put(build_message("shutdown_complete", {}))
    events[SHUTDOWN_COMPLETE].set()
    
    logger.info("Shutdown complete")
