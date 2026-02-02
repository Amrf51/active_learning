"""
EventDispatcher: Central event routing and coordination.

This module provides the EventDispatcher class that serves as the central
coordinator for the MVC architecture. It routes events between layers,
manages business logic, and coordinates between ModelHandler and ServiceManager.
"""

import logging
import traceback
from typing import Dict, Callable, Any, Optional

from controller.events import Event, EventType, create_event
from controller.model_handler import ModelHandler
from controller.service_manager import ServiceManager

logger = logging.getLogger(__name__)


class EventDispatcher:
    """
    Central event dispatcher and coordinator for the MVC architecture.
    
    The EventDispatcher serves as the Controller layer's main component,
    routing events between View and Service layers while coordinating
    business logic through the ModelHandler.
    
    Key responsibilities:
    - Route events to appropriate handlers
    - Coordinate between ModelHandler and ServiceManager
    - Handle errors and provide user feedback
    - Manage experiment lifecycle
    - Provide data access interface for View layer
    """
    
    def __init__(self, model_handler: ModelHandler, service_manager: ServiceManager):
        """
        Initialize EventDispatcher with dependencies.
        
        Args:
            model_handler: Interface to Model layer
            service_manager: Interface to Service layer
        """
        self._model_handler = model_handler
        self._service_manager = service_manager
        
        # Register event handlers
        self._handlers = self._register_handlers()
        
        logger.info("EventDispatcher initialized")
    
    def _register_handlers(self) -> Dict[EventType, Callable[[Event], None]]:
        """
        Register event handlers for all event types.
        
        Returns:
            Dictionary mapping EventType to handler method
        """
        return {
            # View → Controller Events (User Actions)
            EventType.INITIALIZE_EXPERIMENT: self._handle_initialize,
            EventType.START_CYCLE: self._handle_start_cycle,
            EventType.PAUSE_TRAINING: self._handle_pause,
            EventType.RESUME_TRAINING: self._handle_resume,
            EventType.STOP_EXPERIMENT: self._handle_stop,
            EventType.SUBMIT_ANNOTATIONS: self._handle_annotations,
            
            # Service → Controller Events (Status Updates)
            EventType.SERVICE_READY: self._handle_service_ready,
            EventType.EPOCH_COMPLETE: self._handle_epoch_complete,
            EventType.CYCLE_COMPLETE: self._handle_cycle_complete,
            EventType.QUERY_READY: self._handle_query_ready,
            EventType.SERVICE_ERROR: self._handle_service_error
        }
    
    def dispatch(self, event: Event) -> bool:
        """
        Dispatch event to appropriate handler with error handling.
        
        Args:
            event: Event to dispatch
            
        Returns:
            True if event was handled successfully, False otherwise
        """
        logger.debug(f"Dispatching event: {event.type}")
        
        try:
            handler = self._handlers.get(event.type)
            if handler:
                handler(event)
                logger.debug(f"Event {event.type} handled successfully")
                return True
            else:
                logger.warning(f"No handler registered for event type: {event.type}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}\n{traceback.format_exc()}")
            
            # Set error state in model
            self._model_handler.set_error(f"Error handling {event.type.value}: {str(e)}")
            
            return False
    
    # View Event Handlers (User Actions)
    
    def _handle_initialize(self, event: Event) -> None:
        """
        Handle experiment initialization request.
        
        Validates configuration, initializes experiment in model,
        and spawns the service process.
        
        Args:
            event: INITIALIZE_EXPERIMENT event with config payload
        """
        config = event.payload.get('config', {})
        experiment_name = event.payload.get('experiment_name')

        # Add experiment_name to config if provided
        if experiment_name:
            config['name'] = experiment_name  # Added

        logger.info(f"Initializing new experiment: {experiment_name or 'unnamed'}")
        
        try:
            # Validate configuration
            validation = self._model_handler.validate_config(config)
            if not validation.is_valid:
                self._model_handler.set_error(f"Invalid configuration: {validation.error_message}")
                return
            
            # Initialize experiment in model
            experiment_id = self._model_handler.initialize_experiment(config)
            
            # Set initializing phase
            self._model_handler.set_phase("INITIALIZING")
            
            # Spawn service process
            success = self._service_manager.spawn_service(config, self._handle_service_event)
            
            if not success:
                self._model_handler.set_error("Failed to start service process")
                return
            
            logger.info(f"Experiment {experiment_id} initialization started")
            
        except Exception as e:
            logger.error(f"Failed to initialize experiment: {e}")
            self._model_handler.set_error(f"Initialization failed: {str(e)}")
    
    def _handle_start_cycle(self, event: Event) -> None:
        """
        Handle start cycle request.
        
        Sets training phase and sends start command to service.
        
        Args:
            event: START_CYCLE event
        """
        logger.info("Starting training cycle")
        
        try:
            # Set training phase
            self._model_handler.set_phase("TRAINING")
            
            # Send command to service
            cmd_event = create_event(
                EventType.CMD_START_CYCLE,
                {"cycle_number": self._model_handler._world_state.current_cycle + 1},
                source="controller"
            )
            
            success = self._service_manager.send_command(cmd_event)
            if not success:
                self._model_handler.set_error("Failed to send start command to service")
                
        except Exception as e:
            logger.error(f"Failed to start cycle: {e}")
            self._model_handler.set_error(f"Failed to start cycle: {str(e)}")
    
    def _handle_pause(self, event: Event) -> None:
        """
        Handle pause training request.
        
        Args:
            event: PAUSE_TRAINING event
        """
        logger.info("Pausing training")
        
        try:
            # Send pause command to service
            cmd_event = create_event(EventType.CMD_PAUSE, source="controller")
            success = self._service_manager.send_command(cmd_event)
            
            if success:
                # Update phase to indicate paused state
                # Note: We could add a PAUSED phase or keep TRAINING with a flag
                logger.info("Training pause command sent")
            else:
                self._model_handler.set_error("Failed to send pause command to service")
                
        except Exception as e:
            logger.error(f"Failed to pause training: {e}")
            self._model_handler.set_error(f"Failed to pause training: {str(e)}")
    
    def _handle_resume(self, event: Event) -> None:
        """
        Handle resume training request.
        
        Args:
            event: RESUME_TRAINING event
        """
        logger.info("Resuming training")
        
        try:
            # Send resume command to service
            cmd_event = create_event(EventType.CMD_RESUME, source="controller")
            success = self._service_manager.send_command(cmd_event)
            
            if success:
                # Ensure training phase is set
                self._model_handler.set_phase("TRAINING")
                logger.info("Training resume command sent")
            else:
                self._model_handler.set_error("Failed to send resume command to service")
                
        except Exception as e:
            logger.error(f"Failed to resume training: {e}")
            self._model_handler.set_error(f"Failed to resume training: {str(e)}")
    
    def _handle_stop(self, event: Event) -> None:
        """
        Handle stop experiment request.
        
        Args:
            event: STOP_EXPERIMENT event
        """
        logger.info("Stopping experiment")
        
        try:
            # Send stop command to service
            cmd_event = create_event(EventType.CMD_STOP, source="controller")
            success = self._service_manager.send_command(cmd_event)
            
            if success:
                # Set idle phase
                self._model_handler.set_phase("IDLE")
                logger.info("Experiment stop command sent")
            else:
                self._model_handler.set_error("Failed to send stop command to service")
                
        except Exception as e:
            logger.error(f"Failed to stop experiment: {e}")
            self._model_handler.set_error(f"Failed to stop experiment: {str(e)}")
    
    def _handle_annotations(self, event: Event) -> None:
        """
        Handle annotation submission.
        
        Processes user annotations and sends them to service.
        
        Args:
            event: SUBMIT_ANNOTATIONS event with annotations payload
        """
        annotations = event.payload.get('annotations', [])
        
        logger.info(f"Processing {len(annotations)} annotations")
        
        try:
            # Validate annotations
            if not annotations:
                self._model_handler.set_error("No annotations provided")
                return
            
            # Send annotations to service
            cmd_event = create_event(
                EventType.CMD_ANNOTATIONS,
                {"annotations": annotations},
                source="controller"
            )
            
            success = self._service_manager.send_command(cmd_event)
            
            if success:
                # Set processing phase (service will update when complete)
                self._model_handler.set_phase("TRAINING")  # Processing annotations
                logger.info(f"Sent {len(annotations)} annotations to service")
            else:
                self._model_handler.set_error("Failed to send annotations to service")
                
        except Exception as e:
            logger.error(f"Failed to process annotations: {e}")
            self._model_handler.set_error(f"Failed to process annotations: {str(e)}")
    
    # Service Event Handlers (Status Updates)
    
    def _handle_service_event(self, event: Event) -> None:
        """
        Handle events from service (called by ServiceManager).
        
        This is the callback function passed to ServiceManager that
        routes service events back through the dispatcher.
        
        Args:
            event: Event from service
        """
        logger.debug(f"Received service event: {event.type}")
        self.dispatch(event)
    
    def _handle_service_ready(self, event: Event) -> None:
        """
        Handle service ready notification.
        
        Args:
            event: SERVICE_READY event
        """
        logger.info("Service is ready")
        
        # Set idle phase - ready for commands
        self._model_handler.set_phase("IDLE")
    
    def _handle_epoch_complete(self, event: Event) -> None:
        """
        Handle epoch completion notification.
        
        Updates WorldState with new metrics and persists to database.
        
        Args:
            event: EPOCH_COMPLETE event with metrics payload
        """
        metrics = event.payload
        epoch = metrics.get('epoch', 0)
        cycle = metrics.get('cycle', 0)
        
        logger.debug(f"Epoch {epoch} of cycle {cycle} completed")
        
        try:
            # Update current metrics in WorldState
            self._model_handler.update_current_metrics(metrics)
            
            # Persist metrics to database
            self._model_handler.persist_epoch_metrics(metrics)
            
            # Set pending updates flag for UI refresh
            self._model_handler.set_pending_updates(True)
            
        except Exception as e:
            logger.error(f"Failed to handle epoch completion: {e}")
            self._model_handler.set_error(f"Failed to process epoch results: {str(e)}")
    
    def _handle_cycle_complete(self, event: Event) -> None:
        """
        Handle cycle completion notification.
        
        Finalizes the cycle and sets idle phase.
        
        Args:
            event: CYCLE_COMPLETE event with results payload
        """
        results = event.payload.get('results', {})
        cycle = results.get('cycle', 0)
        
        logger.info(f"Cycle {cycle} completed")
        
        try:
            # Finalize cycle in model
            self._model_handler.finalize_cycle(results)
            
            # Set idle phase - ready for next cycle
            self._model_handler.set_phase("IDLE")
            
        except Exception as e:
            logger.error(f"Failed to handle cycle completion: {e}")
            self._model_handler.set_error(f"Failed to finalize cycle: {str(e)}")
    
    def _handle_query_ready(self, event: Event) -> None:
        """
        Handle query ready notification.
        
        Sets queried images and awaiting annotation phase.
        
        Args:
            event: QUERY_READY event with queried images payload
        """
        images = event.payload.get('images', [])
        cycle = event.payload.get('cycle', 0)
        
        logger.info(f"Query ready for cycle {cycle} with {len(images)} images")
        
        try:
            # Set queried images in model
            self._model_handler.set_queried_images(images)
            
            # Phase is automatically set to AWAITING_ANNOTATION by set_queried_images
            
        except Exception as e:
            logger.error(f"Failed to handle query ready: {e}")
            self._model_handler.set_error(f"Failed to process query results: {str(e)}")
    
    def _handle_service_error(self, event: Event) -> None:
        """
        Handle service error notification.
        
        Sets error state with service error information.
        
        Args:
            event: SERVICE_ERROR event with error payload
        """
        error_type = event.payload.get('error_type', 'unknown')
        message = event.payload.get('message', 'Service error occurred')
        
        logger.error(f"Service error ({error_type}): {message}")
        
        # Set error state
        self._model_handler.set_error(f"Service error: {message}")
    
    # Data Access Methods (delegate to ModelHandler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        return self._model_handler.get_status()
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return self._model_handler.get_training_progress()
    
    def get_queried_images(self) -> list:
        """Get current queried images."""
        return self._model_handler.get_queried_images()
    
    def has_pending_updates(self) -> bool:
        """Check if there are pending UI updates."""
        return self._model_handler.has_pending_updates()
    
    def clear_pending_updates(self) -> None:
        """Clear pending updates flag."""
        self._model_handler.clear_pending_updates()
    
    def get_results_history(self, page: int = 1, limit: int = 20):
        """Get paginated experiment history."""
        return self._model_handler.get_results_history(page, limit)
    
    def get_epoch_history(self, experiment_id: str, cycle: Optional[int] = None, 
                         page: int = 1, limit: int = 50):
        """Get paginated epoch metrics."""
        return self._model_handler.get_epoch_history(experiment_id, cycle, page, limit)
    
    def get_pool_page(self, pool_type: str, page: int = 1, limit: int = 50):
        """Get paginated pool items."""
        return self._model_handler.get_pool_page(pool_type, page, limit)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary."""
        return self._model_handler.get_experiment_summary()
    
    def validate_config(self, config: Dict[str, Any]):
        """Validate experiment configuration."""
        return self._model_handler.validate_config(config)
    
    # Lifecycle Management
    
    def shutdown(self) -> None:
        """
        Shutdown the dispatcher and cleanup resources.
        
        Terminates service processes and cleans up all resources.
        Should be called when the application is shutting down.
        """
        logger.info("EventDispatcher shutting down...")
        
        try:
            # Shutdown service manager
            self._service_manager.shutdown()
            
            # Reset model state
            self._model_handler.reset_experiment()
            
        except Exception as e:
            logger.error(f"Error during dispatcher shutdown: {e}")
        
        logger.info("EventDispatcher shutdown complete")
    
    def reset_experiment(self) -> None:
        """
        Reset current experiment state.
        
        Used for error recovery or starting fresh.
        """
        logger.info("Resetting experiment state")
        
        try:
            # Shutdown service if running
            self._service_manager.shutdown()
            
            # Reset model state
            self._model_handler.reset_experiment()
            
        except Exception as e:
            logger.error(f"Error during experiment reset: {e}")
            self._model_handler.set_error(f"Reset failed: {str(e)}")
    
    def retry_last_action(self) -> None:
        """
        Retry the last failed action.
        
        This is a simplified retry mechanism. In a more sophisticated
        implementation, you might store the last action and replay it.
        """
        logger.info("Retrying last action")
        
        # For now, just clear error state and set to idle
        # In practice, you might want to store and replay the last command
        self._model_handler.set_phase("IDLE")
    
    # Service Management Utilities
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get information about the service process.
        
        Returns:
            Dictionary with service status information
        """
        return self._service_manager.get_process_info()
    
    def is_service_alive(self) -> bool:
        """Check if service process is running."""
        return self._service_manager.is_alive()