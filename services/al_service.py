"""
ActiveLearningService - Background Training Service.

This module implements the Service layer that runs in a separate process
and handles all heavy computation (training, evaluation, querying).

This REPLACES:
- run_worker.py (manual worker start)
- worker_command_loop.py (command polling)

Key Changes from Worker Pattern:
1. No more JSON file polling - uses Pipe for instant communication
2. Auto-started by ServiceManager - no manual terminal command
3. Event-driven - receives commands, sends progress events

Usage (called by ServiceManager):
    from multiprocessing import Process, Pipe
    
    parent_conn, child_conn = Pipe()
    process = Process(
        target=run_active_learning_service,
        args=(child_conn, config_dict)
    )
    process.start()

Communication Protocol:
    CONTROLLER → SERVICE: Commands (CMD_START_CYCLE, CMD_PAUSE, etc.)
    SERVICE → CONTROLLER: Events (EPOCH_COMPLETE, QUERY_READY, etc.)
"""

import logging
import sys
import time
import traceback
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path for imports
# This will be adjusted based on actual project structure
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller.events import Event, EventType, create_event, error_event

logger = logging.getLogger(__name__)


def run_active_learning_service(pipe: Connection, config: Dict[str, Any]) -> None:
    """
    Entry point for the ActiveLearningService process.
    
    This function is called by ServiceManager.spawn_service() via
    multiprocessing.Process(target=run_active_learning_service, ...).
    
    Args:
        pipe: Bidirectional Pipe connection for communication
        config: Experiment configuration dictionary
    """
    # Configure logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - SERVICE - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logger.info("ActiveLearningService starting...")
    
    try:
        service = ActiveLearningService(pipe, config)
        service.run()
    except Exception as e:
        logger.error(f"Service crashed: {e}", exc_info=True)
        # Try to send error event before dying
        try:
            pipe.send(error_event(
                message=str(e),
                traceback=traceback.format_exc(),
                recoverable=False,
                source="service"
            ).to_dict())
        except:
            pass
        sys.exit(1)


class ActiveLearningService:
    """
    Background service for Active Learning training.
    
    This class orchestrates the AL loop in a separate process,
    communicating with the Controller via Pipe.
    
    Lifecycle:
        1. __init__: Store config, set up flags
        2. run(): Main loop - wait for commands, execute, send events
        3. Shutdown when CMD_SHUTDOWN received or pipe closes
    
    Attributes:
        pipe: Bidirectional Pipe connection
        config: Experiment configuration
        
        # Backend components (initialized lazily)
        trainer: Trainer instance
        data_manager: ALDataManager instance
        al_loop: ActiveLearningLoop instance
        
        # Control flags
        _should_stop: Whether to stop the service
        _is_paused: Whether training is paused
    """
    
    def __init__(self, pipe: Connection, config: Dict[str, Any]):
        """
        Initialize the service.
        
        Args:
            pipe: Bidirectional Pipe connection
            config: Experiment configuration dictionary
        """
        self.pipe = pipe
        self.config = config
        
        # Backend components - initialized when needed
        self.trainer = None
        self.data_manager = None
        self.al_loop = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = []
        
        # Control flags
        self._should_stop = False
        self._is_paused = False
        self._current_cycle = 0
        
        # Experiment directory
        self.exp_dir = Path(config.get("experiment_dir", "experiments/current"))
        
        logger.info(f"Service initialized with config: {list(config.keys())}")
    
    def run(self) -> None:
        """
        Main service loop.
        
        Waits for commands from Controller and executes them.
        Sends events back with progress and results.
        """
        logger.info("Service entering main loop")
        
        # Initialize backend components
        try:
            self._initialize_components()
            self._send_event(EventType.SERVICE_READY, {"message": "Service initialized"})
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}", exc_info=True)
            self._send_error(str(e), traceback.format_exc())
            return
        
        # Main command loop
        while not self._should_stop:
            try:
                # Wait for command (with timeout to check stop flag)
                if self.pipe.poll(timeout=0.5):
                    event_dict = self.pipe.recv()
                    event = Event.from_dict(event_dict)
                    self._handle_command(event)
                    
            except EOFError:
                # Pipe closed - parent process died
                logger.warning("Pipe closed - parent process terminated")
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self._send_error(str(e), traceback.format_exc())
                time.sleep(1)  # Prevent tight error loop
        
        logger.info("Service shutting down")
        self._cleanup()
    
    def _initialize_components(self) -> None:
        """
        Initialize backend components (Trainer, DataManager, ALLoop).
        
        This imports and initializes the existing backend code.
        """
        logger.info("Initializing backend components...")
        
        # Import backend modules
        # These imports are done here to avoid importing torch in main process
        try:
            import torch
            from backend.trainer import Trainer
            from backend.data_manager import ALDataManager
            from backend.active_loop import ActiveLearningLoop
            from backend.dataloader import get_dataloaders
            from backend.models import get_model
            from backend.strategies import get_strategy
            from backend.config import Config
        except ImportError as e:
            logger.warning(f"Backend import failed: {e}")
            logger.info("Running in mock mode for testing")
            self._mock_mode = True
            return
        
        self._mock_mode = False
        
        # Create Config object from config dict
        cfg = Config()
        cfg.model.name = self.config.get("model_name", "resnet18")
        cfg.model.pretrained = self.config.get("pretrained", True)
        cfg.model.num_classes = self.config.get("num_classes", 4)
        
        cfg.training.epochs = self.config.get("epochs_per_cycle", 10)
        cfg.training.batch_size = self.config.get("batch_size", 32)
        cfg.training.learning_rate = self.config.get("learning_rate", 0.001)
        cfg.training.optimizer = self.config.get("optimizer", "adam")
        cfg.training.weight_decay = self.config.get("weight_decay", 0.0001)
        cfg.training.early_stopping_patience = self.config.get("early_stopping_patience", 5)
        cfg.training.seed = self.config.get("seed", 42)
        
        cfg.active_learning.num_cycles = self.config.get("num_cycles", 5)
        cfg.active_learning.sampling_strategy = self.config.get("sampling_strategy", "uncertainty")
        cfg.active_learning.uncertainty_method = self.config.get("uncertainty_method", "entropy")
        cfg.active_learning.initial_pool_size = self.config.get("initial_pool_size", 40)
        cfg.active_learning.batch_size_al = self.config.get("batch_size_al", 20)
        cfg.active_learning.reset_mode = self.config.get("reset_mode", "reset_to_pretrained")
        
        cfg.data.data_dir = self.config.get("data_dir", "")
        cfg.data.val_split = self.config.get("val_split", 0.15)
        cfg.data.test_split = self.config.get("test_split", 0.15)
        cfg.data.augmentation = self.config.get("augmentation", True)
        
        # Initialize data loaders
        train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=cfg.training.batch_size,
            val_split=cfg.data.val_split,
            test_split=cfg.data.test_split,
            augmentation=cfg.data.augmentation,
            num_workers=cfg.data.num_workers,
            seed=cfg.training.seed
        )
        
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = dataset_info["class_names"]
        
        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_model(cfg.model, device=device)
        
        # Initialize trainer
        self.trainer = Trainer(model, cfg, self.exp_dir, device=device)
        
        # Initialize data manager
        self.data_manager = ALDataManager(
            train_loader.dataset,
            initial_pool_size=cfg.active_learning.initial_pool_size,
            seed=cfg.training.seed
        )
        
        # Initialize strategy
        strategy = get_strategy(cfg.active_learning.sampling_strategy)
        
        # Initialize AL loop
        self.al_loop = ActiveLearningLoop(
            trainer=self.trainer,
            data_manager=self.data_manager,
            strategy=strategy,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=self.exp_dir,
            config=cfg,
            class_names=self.class_names
        )
        
        self._config_obj = cfg
        
        logger.info("Backend components initialized successfully")
    
    def _handle_command(self, event: Event) -> None:
        """
        Handle incoming command from Controller.
        
        Args:
            event: Command event to handle
        """
        logger.info(f"Received command: {event.type.name}")
        
        handlers = {
            EventType.CMD_START_CYCLE: self._handle_start_cycle,
            EventType.CMD_PAUSE: self._handle_pause,
            EventType.CMD_RESUME: self._handle_resume,
            EventType.CMD_STOP: self._handle_stop,
            EventType.CMD_ANNOTATIONS: self._handle_annotations,
            EventType.CMD_SHUTDOWN: self._handle_shutdown,
        }
        
        handler = handlers.get(event.type)
        if handler:
            try:
                handler(event.payload)
            except Exception as e:
                logger.error(f"Error handling {event.type.name}: {e}", exc_info=True)
                self._send_error(f"Command failed: {e}", traceback.format_exc())
        else:
            logger.warning(f"Unknown command type: {event.type}")
    
    def _handle_start_cycle(self, payload: Dict[str, Any]) -> None:
        """
        Handle CMD_START_CYCLE - execute one AL cycle.
        
        Args:
            payload: May contain cycle number override
        """
        logger.info("Starting AL cycle execution")
        
        self._current_cycle += 1
        cycle = payload.get("cycle", self._current_cycle)
        
        try:
            self._execute_training_cycle(cycle)
        except Exception as e:
            logger.error(f"Cycle execution failed: {e}", exc_info=True)
            self._send_error(f"Cycle {cycle} failed: {e}", traceback.format_exc())
    
    def _handle_pause(self, payload: Dict[str, Any]) -> None:
        """Handle CMD_PAUSE - pause training."""
        logger.info("Pausing training")
        self._is_paused = True
    
    def _handle_resume(self, payload: Dict[str, Any]) -> None:
        """Handle CMD_RESUME - resume training."""
        logger.info("Resuming training")
        self._is_paused = False
    
    def _handle_stop(self, payload: Dict[str, Any]) -> None:
        """Handle CMD_STOP - stop experiment."""
        logger.info("Stopping experiment")
        self._should_stop = True
    
    def _handle_annotations(self, payload: Dict[str, Any]) -> None:
        """
        Handle CMD_ANNOTATIONS - process user annotations.
        
        Args:
            payload: {"annotations": [{"image_id": int, "label": int}, ...]}
        """
        logger.info("Processing annotations")
        
        annotations = payload.get("annotations", [])
        
        if getattr(self, '_mock_mode', False):
            # Mock mode - just acknowledge
            logger.info(f"Mock: received {len(annotations)} annotations")
            self._send_event(EventType.CYCLE_COMPLETE, {
                "cycle": self._current_cycle,
                "annotations_processed": len(annotations)
            })
            return
        
        try:
            # Process annotations through AL loop
            result = self.al_loop.receive_annotations(annotations)
            
            logger.info(f"Annotations processed: {result['moved_count']} samples added")
            
            # Update pool info
            pool_info = self.data_manager.get_pool_info()
            
            self._send_event(EventType.CYCLE_COMPLETE, {
                "cycle": self._current_cycle,
                "labeled_count": pool_info["labeled"],
                "unlabeled_count": pool_info["unlabeled"],
                "annotations_processed": result["moved_count"]
            })
            
        except Exception as e:
            logger.error(f"Annotation processing failed: {e}", exc_info=True)
            self._send_error(f"Annotation failed: {e}", traceback.format_exc())
    
    def _handle_shutdown(self, payload: Dict[str, Any]) -> None:
        """Handle CMD_SHUTDOWN - graceful shutdown."""
        logger.info("Shutdown requested")
        self._should_stop = True
    
    def _execute_training_cycle(self, cycle: int) -> None:
        """
        Execute one complete AL cycle.
        
        Phases:
            1. Prepare cycle (reset model if needed)
            2. Train for configured epochs
            3. Evaluate on test set
            4. Query samples for annotation
        
        Args:
            cycle: Cycle number (1-indexed)
        """
        logger.info(f"Executing cycle {cycle}")
        
        # Check if in mock mode
        if getattr(self, '_mock_mode', False):
            self._execute_mock_cycle(cycle)
            return
        
        # Phase 1: Prepare
        cycle_info = self.al_loop.prepare_cycle(cycle)
        logger.info(f"Cycle prepared: labeled={cycle_info['labeled_count']}, unlabeled={cycle_info['unlabeled_count']}")
        
        # Phase 2: Train
        epochs = self._config_obj.training.epochs
        
        for epoch in range(1, epochs + 1):
            # Check pause/stop
            if self._should_stop:
                logger.info("Stop requested during training")
                return
            
            while self._is_paused:
                time.sleep(0.5)
                if self._should_stop:
                    return
            
            # Train single epoch
            train_metrics = self.trainer.train_single_epoch(
                self.al_loop.train_loader,
                epoch
            )
            
            # Validate
            val_metrics = self.trainer.validate(self.val_loader)
            
            # Send epoch complete event
            self._send_event(EventType.EPOCH_COMPLETE, {
                "epoch": epoch,
                "train_loss": train_metrics["train_loss"],
                "train_accuracy": train_metrics["train_accuracy"],
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["val_accuracy"],
                "learning_rate": self.trainer.get_current_lr()
            })
            
            logger.info(
                f"Epoch {epoch}/{epochs}: "
                f"train_loss={train_metrics['train_loss']:.4f}, "
                f"val_acc={val_metrics['val_accuracy']:.4f}"
            )
        
        # Send training complete
        self._send_event(EventType.TRAINING_COMPLETE, {"cycle": cycle})
        
        # Phase 3: Evaluate
        test_metrics = self.trainer.evaluate(
            self.test_loader,
            class_names=self.class_names
        )
        
        self._send_event(EventType.EVALUATION_COMPLETE, {
            "cycle": cycle,
            "test_accuracy": test_metrics["test_accuracy"],
            "test_f1": test_metrics["test_f1"]
        })
        
        # Phase 4: Query
        queried_images = self.al_loop.query_samples()
        
        # Convert to serializable format
        queried_list = []
        for img in queried_images:
            if hasattr(img, 'to_dict'):
                queried_list.append(img.to_dict())
            elif isinstance(img, dict):
                queried_list.append(img)
            else:
                queried_list.append({
                    "image_id": getattr(img, 'image_id', 0),
                    "image_path": getattr(img, 'image_path', ''),
                    "uncertainty_score": getattr(img, 'uncertainty_score', 0.0)
                })
        
        self._send_event(EventType.QUERY_READY, {
            "cycle": cycle,
            "queried_images": queried_list,
            "count": len(queried_list)
        })
        
        logger.info(f"Cycle {cycle} training complete, {len(queried_list)} samples queried")
    
    def _execute_mock_cycle(self, cycle: int) -> None:
        """
        Execute mock cycle for testing without actual training.
        
        Args:
            cycle: Cycle number
        """
        logger.info(f"Executing MOCK cycle {cycle}")
        
        epochs = self.config.get("epochs_per_cycle", 10)
        
        for epoch in range(1, epochs + 1):
            if self._should_stop:
                return
            
            while self._is_paused:
                time.sleep(0.5)
                if self._should_stop:
                    return
            
            # Simulate training time
            time.sleep(0.5)
            
            # Send mock metrics
            self._send_event(EventType.EPOCH_COMPLETE, {
                "epoch": epoch,
                "train_loss": 1.0 - (epoch * 0.05),
                "train_accuracy": 0.5 + (epoch * 0.03),
                "val_loss": 1.1 - (epoch * 0.04),
                "val_accuracy": 0.45 + (epoch * 0.035),
                "learning_rate": 0.001
            })
            
            logger.info(f"Mock epoch {epoch}/{epochs} complete")
        
        # Send training complete
        self._send_event(EventType.TRAINING_COMPLETE, {"cycle": cycle})
        
        # Send mock evaluation
        self._send_event(EventType.EVALUATION_COMPLETE, {
            "cycle": cycle,
            "test_accuracy": 0.75 + (cycle * 0.02),
            "test_f1": 0.72 + (cycle * 0.02)
        })
        
        # Send mock queried images
        mock_queried = [
            {
                "image_id": i,
                "image_path": f"/data/image_{i}.jpg",
                "display_path": f"/data/image_{i}.jpg",
                "ground_truth": i % 4,
                "ground_truth_name": ["bus", "car", "motorcycle", "truck"][i % 4],
                "predicted_class": ["bus", "car", "motorcycle", "truck"][(i + 1) % 4],
                "predicted_confidence": 0.6 + (i * 0.02),
                "uncertainty_score": 0.8 - (i * 0.05),
                "selection_reason": "high_uncertainty",
                "model_probabilities": {"bus": 0.25, "car": 0.25, "motorcycle": 0.25, "truck": 0.25}
            }
            for i in range(self.config.get("batch_size_al", 20))
        ]
        
        self._send_event(EventType.QUERY_READY, {
            "cycle": cycle,
            "queried_images": mock_queried,
            "count": len(mock_queried)
        })
        
        logger.info(f"Mock cycle {cycle} complete")
    
    def _send_event(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        """
        Send event to Controller via Pipe.
        
        Args:
            event_type: Type of event
            payload: Event data
        """
        event = create_event(event_type, payload, source="service")
        try:
            self.pipe.send(event.to_dict())
            logger.debug(f"Sent event: {event_type.name}")
        except BrokenPipeError:
            logger.error("Pipe broken - cannot send event")
            self._should_stop = True
    
    def _send_error(self, message: str, tb: Optional[str] = None) -> None:
        """
        Send error event to Controller.
        
        Args:
            message: Error message
            tb: Traceback string
        """
        event = error_event(message, tb, recoverable=True, source="service")
        try:
            self.pipe.send(event.to_dict())
        except BrokenPipeError:
            logger.error("Pipe broken - cannot send error")
            self._should_stop = True
    
    def _cleanup(self) -> None:
        """Clean up resources before shutdown."""
        logger.info("Cleaning up resources")
        
        # Save any pending state
        if self.data_manager and hasattr(self.data_manager, 'save_state'):
            try:
                self.data_manager.save_state(self.exp_dir / "al_pool_state.json")
            except Exception as e:
                logger.warning(f"Failed to save pool state: {e}")
        
        # Close pipe
        try:
            self.pipe.close()
        except:
            pass
        
        logger.info("Cleanup complete")
