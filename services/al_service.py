"""
ActiveLearningService - Service layer for Active Learning Dashboard.

This service runs in a separate process and handles all ML training operations.
It communicates with the Controller via Pipe-based IPC, receiving commands
and sending back progress events.

Key responsibilities:
- Execute training cycles in isolation from UI
- Emit progress events for real-time updates
- Handle pause/resume/stop commands gracefully
- Integrate with existing backend components (Trainer, DataManager, ActiveLoop)
"""

import logging
import traceback
from multiprocessing.connection import Connection
from typing import Optional, Dict, Any

from controller.events import Event, EventType, create_service_event

logger = logging.getLogger(__name__)


def run_active_learning_service(pipe: Connection, config: dict) -> None:
    """
    Entry point for the ActiveLearningService process.
    
    This function is called by ServiceManager.spawn_service() to start
    the service in a separate daemon process.
    
    Args:
        pipe: Bidirectional pipe for communication with Controller
        config: Experiment configuration dictionary
    """
    try:
        service = ActiveLearningService(pipe, config)
        service.run()
    except Exception as e:
        logger.error(f"Service startup failed: {e}\n{traceback.format_exc()}")
        try:
            # Try to send error event before dying
            error_event = create_service_event(
                EventType.SERVICE_ERROR,
                {
                    "error_type": "startup_error",
                    "message": f"Service startup failed: {str(e)}",
                    "traceback": traceback.format_exc()
                }
            )
            pipe.send(error_event)
        except:
            pass  # Pipe might be broken, nothing we can do


class ActiveLearningService:
    """
    Active Learning Service that runs in a separate process.
    
    Handles all ML training operations and communicates with the Controller
    via events. Integrates with existing backend components while providing
    step-by-step control for interactive dashboard use.
    """
    
    def __init__(self, pipe: Connection, config: dict):
        """
        Initialize the ActiveLearningService.
        
        Args:
            pipe: Bidirectional pipe for communication with Controller
            config: Experiment configuration dictionary
        """
        self._pipe = pipe
        self._config = config
        
        # Control flags for training flow
        self._should_stop = False
        self._is_paused = False
        
        # Backend components (initialized lazily)
        self._trainer = None
        self._data_manager = None
        self._al_loop = None
        
        # Current state tracking
        self._current_cycle = 0
        self._current_epoch = 0
        
        logger.info("ActiveLearningService initialized")
    
    def run(self) -> None:
        """
        Main service loop with comprehensive error handling.
        
        Initializes backend components, sends SERVICE_READY event,
        then enters command processing loop until shutdown.
        All exceptions are caught and reported via SERVICE_ERROR events.
        """
        try:
            # Initialize backend components
            self._initialize_components()
            
            # Signal that service is ready
            self._send_event(EventType.SERVICE_READY, {
                "service_id": "active_learning_service",
                "status": "ready"
            })
            
            logger.info("Service ready, entering command loop")
            
            # Main command processing loop
            while not self._should_stop:
                try:
                    self._process_commands()
                except EOFError:
                    # Parent process died, shutdown gracefully
                    logger.info("Parent process died (EOFError), shutting down")
                    break
                except BrokenPipeError:
                    # Pipe was closed, shutdown gracefully
                    logger.info("Pipe broken, shutting down")
                    break
                except Exception as e:
                    # Log error but continue processing
                    logger.error(f"Error in command processing loop: {e}")
                    self._send_event(EventType.SERVICE_ERROR, {
                        "error_type": "command_loop_error",
                        "message": str(e),
                        "traceback": traceback.format_exc()
                    })
                    
        except Exception as e:
            # Critical initialization or setup error
            logger.error(f"Critical service error: {e}\n{traceback.format_exc()}")
            self._send_event(EventType.SERVICE_ERROR, {
                "error_type": "critical_error",
                "message": f"Critical service failure: {str(e)}",
                "traceback": traceback.format_exc()
            })
        finally:
            # Always attempt graceful shutdown
            self._shutdown()
    
    def _process_commands(self) -> None:
        """
        Process incoming commands from Controller with error handling.
        
        Polls the pipe for incoming events and routes them to
        appropriate command handlers. Handles pipe errors gracefully.
        """
        try:
            # Poll for incoming commands with timeout
            if self._pipe.poll(timeout=0.5):
                event = self._pipe.recv()
                self._handle_command(event)
        except EOFError:
            # Parent process died, propagate to main loop
            logger.info("Pipe closed by parent (EOFError)")
            raise
        except BrokenPipeError:
            # Pipe was broken, propagate to main loop
            logger.info("Pipe broken (BrokenPipeError)")
            raise
        except Exception as e:
            logger.error(f"Error processing commands: {e}")
            # Don't propagate non-pipe errors, just log them
    
    def _handle_command(self, event: Event) -> None:
        """
        Route command events to appropriate handlers.
        
        Args:
            event: Command event from Controller
        """
        logger.debug(f"Received command: {event.type}")
        
        try:
            if event.type == EventType.CMD_START_CYCLE:
                self._execute_training_cycle()
            elif event.type == EventType.CMD_PAUSE:
                self._is_paused = True
                logger.info("Training paused")
            elif event.type == EventType.CMD_RESUME:
                self._is_paused = False
                logger.info("Training resumed")
            elif event.type == EventType.CMD_STOP:
                self._should_stop = True
                logger.info("Stop command received")
            elif event.type == EventType.CMD_ANNOTATIONS:
                self._process_annotations(event.payload)
            elif event.type == EventType.CMD_SHUTDOWN:
                self._should_stop = True
                logger.info("Shutdown command received")
            else:
                logger.warning(f"Unknown command type: {event.type}")
                
        except Exception as e:
            logger.error(f"Error handling command {event.type}: {e}")
            self._send_event(EventType.SERVICE_ERROR, {
                "error_type": "command_error",
                "message": f"Error handling {event.type}: {str(e)}",
                "traceback": traceback.format_exc()
            })
    
    def _initialize_components(self) -> None:
        """
        Initialize backend components (Trainer, DataManager, ActiveLoop).
        
        This integrates with the existing backend components while
        adapting them for the new event-driven architecture.
        """
        try:
            # Import backend components
            from backend.trainer import Trainer
            from backend.data_manager import ALDataManager
            from backend.active_loop import ActiveLearningLoop
            from backend.models import get_model
            from backend.dataloader import get_dataloaders
            
            logger.info("Initializing backend components...")
            
            # Get data loaders (val and test are fixed, train will be managed by ALDataManager)
            train_dataset, val_loader, test_loader, class_names = get_dataloaders(self._config)
            
            # Initialize data manager with training dataset
            self._data_manager = ALDataManager(
                dataset=train_dataset,
                initial_pool_size=self._config.get('initial_pool_size', 50),
                seed=self._config.get('seed', 42)
            )
            
            # Initialize model and trainer
            model = get_model(self._config)
            self._trainer = Trainer(
                model=model,
                config=self._config,
                exp_dir=self._config.get('exp_dir', './experiments'),
                device=self._config.get('device', 'cuda')
            )
            
            # Initialize active learning loop
            # Import strategy function
            from backend.strategies import get_strategy
            strategy_fn = get_strategy(self._config.get('strategy', 'uncertainty'))
            
            self._al_loop = ActiveLearningLoop(
                trainer=self._trainer,
                data_manager=self._data_manager,
                strategy=strategy_fn,
                val_loader=val_loader,
                test_loader=test_loader,
                exp_dir=self._config.get('exp_dir', './experiments'),
                config=self._config,
                class_names=class_names
            )
            
            logger.info("Backend components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _send_event(self, event_type: EventType, payload: Dict[str, Any]) -> None:
        """
        Send event to Controller via pipe with error handling.
        
        Args:
            event_type: Type of event to send
            payload: Event payload data
        """
        try:
            event = create_service_event(event_type, payload)
            self._pipe.send(event)
        except BrokenPipeError:
            logger.warning("Pipe broken, cannot send event - parent may have died")
            self._should_stop = True
        except EOFError:
            logger.warning("Pipe closed, cannot send event - parent may have died")
            self._should_stop = True
        except Exception as e:
            logger.error(f"Failed to send event {event_type}: {e}")
            # Don't stop service for send failures, just log them
    
    def _execute_training_cycle(self) -> None:
        """
        Execute one complete Active Learning training cycle.
        
        This method:
        1. Prepares the cycle (resets model, gets data loaders)
        2. Trains for configured number of epochs with pause/stop checks
        3. Emits EPOCH_COMPLETE events after each epoch
        4. Queries samples for annotation after training
        5. Emits QUERY_READY event with queried images
        """
        try:
            # Increment cycle counter
            self._current_cycle += 1
            
            logger.info(f"Starting training cycle {self._current_cycle}")
            
            # Prepare cycle (reset model, get loaders)
            cycle_info = self._al_loop.prepare_cycle(self._current_cycle)
            
            # Get training configuration
            epochs_per_cycle = self._config.get('epochs_per_cycle', 10)
            
            # Training loop with epoch-by-epoch control
            for epoch in range(1, epochs_per_cycle + 1):
                # Check for stop command
                if self._should_stop:
                    logger.info(f"Training stopped at epoch {epoch}")
                    break
                
                # Handle pause state
                while self._is_paused and not self._should_stop:
                    # Process commands while paused
                    self._process_commands()
                
                if self._should_stop:
                    break
                
                # Train single epoch
                self._current_epoch = epoch
                metrics = self._al_loop.train_single_epoch(epoch)
                
                # Send epoch completion event
                self._send_event(EventType.EPOCH_COMPLETE, {
                    "cycle": self._current_cycle,
                    "epoch": epoch,
                    "train_loss": metrics.train_loss,
                    "val_loss": metrics.val_loss,
                    "train_accuracy": metrics.train_accuracy,
                    "val_accuracy": metrics.val_accuracy,
                    "learning_rate": metrics.learning_rate
                })
                
                logger.info(
                    f"Cycle {self._current_cycle}, Epoch {epoch}/{epochs_per_cycle} - "
                    f"Train Loss: {metrics.train_loss:.4f}, "
                    f"Val Acc: {metrics.val_accuracy:.4f}" if metrics.val_accuracy else "No validation"
                )
                
                # Check for early stopping
                if self._al_loop.should_stop_early():
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # If training completed without stop, proceed to query phase
            if not self._should_stop:
                # Run evaluation on test set
                test_metrics = self._al_loop.run_evaluation()
                
                # Query samples for annotation
                queried_images = self._al_loop.query_samples()
                
                # Convert queried images to serializable format
                queried_data = []
                for img in queried_images:
                    queried_data.append({
                        "image_id": img.image_id,
                        "image_path": img.image_path,
                        "display_path": img.display_path,
                        "ground_truth": img.ground_truth,
                        "ground_truth_name": img.ground_truth_name,
                        "model_probabilities": img.model_probabilities,
                        "predicted_class": img.predicted_class,
                        "predicted_confidence": img.predicted_confidence,
                        "uncertainty_score": img.uncertainty_score,
                        "selection_reason": img.selection_reason
                    })
                
                # Send query ready event
                self._send_event(EventType.QUERY_READY, {
                    "cycle": self._current_cycle,
                    "images": queried_data,
                    "test_metrics": test_metrics
                })
                
                logger.info(f"Cycle {self._current_cycle} training complete, {len(queried_images)} samples queried")
            
        except Exception as e:
            logger.error(f"Error in training cycle: {e}\n{traceback.format_exc()}")
            self._send_event(EventType.SERVICE_ERROR, {
                "error_type": "training_error",
                "message": f"Training cycle failed: {str(e)}",
                "traceback": traceback.format_exc()
            })
    
    def _process_annotations(self, payload: Dict[str, Any]) -> None:
        """
        Process user annotations and finalize the current cycle.
        
        Args:
            payload: Dictionary containing 'annotations' key with list of annotations
        """
        try:
            annotations = payload.get('annotations', [])
            
            if not annotations:
                logger.warning("No annotations provided")
                return
            
            logger.info(f"Processing {len(annotations)} annotations")
            
            # Update data manager with annotations
            annotation_result = self._al_loop.receive_annotations(annotations)
            
            # Finalize the cycle
            test_metrics = self._al_loop.run_evaluation()  # Get final test metrics
            cycle_metrics = self._al_loop.finalize_cycle(test_metrics)
            
            # Send cycle completion event
            self._send_event(EventType.CYCLE_COMPLETE, {
                "cycle": self._current_cycle,
                "results": {
                    "cycle": cycle_metrics.cycle,
                    "labeled_pool_size": cycle_metrics.labeled_pool_size,
                    "unlabeled_pool_size": cycle_metrics.unlabeled_pool_size,
                    "epochs_trained": cycle_metrics.epochs_trained,
                    "best_val_accuracy": cycle_metrics.best_val_accuracy,
                    "best_epoch": cycle_metrics.best_epoch,
                    "test_accuracy": cycle_metrics.test_accuracy,
                    "test_f1": cycle_metrics.test_f1,
                    "test_precision": cycle_metrics.test_precision,
                    "test_recall": cycle_metrics.test_recall
                },
                "annotation_result": annotation_result
            })
            
            logger.info(f"Cycle {self._current_cycle} completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing annotations: {e}\n{traceback.format_exc()}")
            self._send_event(EventType.SERVICE_ERROR, {
                "error_type": "annotation_error",
                "message": f"Annotation processing failed: {str(e)}",
                "traceback": traceback.format_exc()
            })
    
    def _shutdown(self) -> None:
        """
        Graceful shutdown cleanup with comprehensive error handling.
        
        Saves any pending state, cleans up resources, and closes connections.
        This method should never raise exceptions.
        """
        logger.info("Service shutting down...")
        
        try:
            # Save any pending state if needed
            if self._data_manager:
                try:
                    self._data_manager.save_state()
                    logger.info("Data manager state saved")
                except Exception as e:
                    logger.error(f"Failed to save data manager state: {e}")
            
            # Save trainer state if needed
            if self._trainer:
                try:
                    self._trainer.save_training_log()
                    logger.info("Training log saved")
                except Exception as e:
                    logger.error(f"Failed to save training log: {e}")
            
        except Exception as e:
            logger.error(f"Error during state saving: {e}")
        
        try:
            # Close pipe connection
            if self._pipe and not self._pipe.closed:
                self._pipe.close()
                logger.info("Pipe closed")
        except Exception as e:
            logger.error(f"Error closing pipe: {e}")
        
        logger.info("Service shutdown complete")