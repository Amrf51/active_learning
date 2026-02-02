"""
ModelHandler: Interface between Controller and Model layers.

This module provides the ModelHandler class that manages access to both
WorldState (fast in-memory access) and DatabaseManager (persistent storage).
It formats data for view consumption and handles state mutations.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from model.world_state import WorldState
from model.database import DatabaseManager
from model.schemas import ExperimentConfig, ValidationResult, EpochMetrics, ExperimentPhase


logger = logging.getLogger(__name__)


class ModelHandler:
    """
    Interface between Controller and Model layers.
    
    Provides a unified interface for accessing both fast in-memory state
    (WorldState) and persistent storage (SQLite). Handles data formatting
    for view consumption and manages state mutations.
    
    Key responsibilities:
    - Fast state access from WorldState (~1ms)
    - Paginated historical data from SQLite (~50ms)
    - Data formatting for view layer
    - State mutation coordination
    - Configuration validation
    """
    
    def __init__(self, world_state: WorldState, db_manager: DatabaseManager):
        """
        Initialize ModelHandler with state and database references.
        
        Args:
            world_state: In-memory state manager
            db_manager: SQLite database manager
        """
        self._world_state = world_state
        self._db_manager = db_manager
        
        logger.info("ModelHandler initialized")
    
    # State Accessors (from WorldState - fast)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current experiment status from WorldState.
        
        Provides immediate access to current experiment state including
        phase, progress, and error information.
        
        Returns:
            Dictionary with current status information
        """
        status = self._world_state.get_status()
        
        # Add computed fields for view layer
        status.update({
            'is_active': status['experiment_id'] is not None,
            'is_training': status['phase'] == 'TRAINING',
            'is_waiting_annotation': status['phase'] == 'AWAITING_ANNOTATION',
            'is_error': status['phase'] == 'ERROR',
            'has_error': status['error_message'] is not None,
            'progress_text': self._format_progress_text(status)
        })
        
        return status
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Get current training progress from WorldState.
        
        Provides real-time training metrics and progress information
        for live dashboard updates.
        
        Returns:
            Dictionary with training progress information
        """
        progress = self._world_state.get_training_progress()
        
        # Add formatted data for view layer
        progress.update({
            'progress_text': f"Epoch {progress['current_epoch']}/{progress['epochs_per_cycle']}",
            'has_metrics': bool(progress['current_metrics']),
            'has_history': bool(progress['epoch_history']),
            'latest_metrics': progress['current_metrics'],
            'epoch_count': len(progress['epoch_history'])
        })
        
        # Format epoch history for charts
        if progress['epoch_history']:
            progress['loss_history'] = self._format_loss_history(progress['epoch_history'])
            progress['accuracy_history'] = self._format_accuracy_history(progress['epoch_history'])
        
        return progress
    
    def get_queried_images(self) -> List[Dict[str, Any]]:
        """
        Get current queried images for annotation from WorldState.
        
        Returns:
            List of queried image data formatted for view layer
        """
        images = self._world_state.get_queried_images()
        
        # Format images for view layer
        formatted_images = []
        for img in images:
            formatted_img = img.copy()
            
            # Add display-friendly fields
            formatted_img.update({
                'display_name': self._get_image_display_name(img),
                'confidence_text': f"{img.get('predicted_confidence', 0):.2%}",
                'uncertainty_text': f"{img.get('uncertainty_score', 0):.3f}",
                'has_ground_truth': 'ground_truth' in img and img['ground_truth'] is not None
            })
            
            formatted_images.append(formatted_img)
        
        return formatted_images
    
    def has_pending_updates(self) -> bool:
        """
        Check if there are pending UI updates.
        
        This is used by the view layer to determine if it should refresh
        the display. It's a fast in-memory flag check.
        
        Returns:
            True if UI should refresh, False otherwise
        """
        return self._world_state.has_pending_updates()
    
    def clear_pending_updates(self) -> None:
        """Clear the pending updates flag after UI refresh."""
        self._world_state.clear_pending_updates()
    
    # State Mutators
    
    def set_phase(self, phase: str, error_message: Optional[str] = None) -> None:
        """
        Update the experiment phase.
        
        Args:
            phase: New phase (IDLE, INITIALIZING, TRAINING, AWAITING_ANNOTATION, ERROR)
            error_message: Optional error message if phase is ERROR
        """
        # Convert string to ExperimentPhase enum
        phase_enum = ExperimentPhase(phase) if isinstance(phase, str) else phase
        
        self._world_state.set_phase(phase_enum, error_message)
        
        # Persist phase change to database if experiment exists
        if self._world_state.experiment_id:
            self._db_manager.update_experiment_phase(
                self._world_state.experiment_id,
                phase,
                error_message=error_message
            )
        
        logger.debug(f"Phase updated to: {phase}")
    
    def set_error(self, message: str) -> None:
        """
        Set error state with message.
        
        Args:
            message: Error message to display
        """
        self.set_phase("ERROR", message)
        logger.error(f"Error state set: {message}")
    
    def set_pending_updates(self, value: bool = True) -> None:
        """
        Set the pending updates flag.
        
        Args:
            value: Whether there are pending updates
        """
        self._world_state.set_pending_updates(value)
    
    def update_current_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update current training metrics.
        
        Args:
            metrics: Dictionary with training metrics
        """
        epoch = metrics.get('epoch', self._world_state.current_epoch)
        self._world_state.update_training_progress(epoch, metrics)
        
        logger.debug(f"Metrics updated for epoch {epoch}")
    
    def set_queried_images(self, images: List[Dict[str, Any]]) -> None:
        """
        Set the queried images for annotation.
        
        Args:
            images: List of queried image data
        """
        self._world_state.set_queried_images(images)
        
        # Persist queried images to database
        if self._world_state.experiment_id:
            self._db_manager.insert_queried_images(
                self._world_state.experiment_id,
                self._world_state.current_cycle,
                images
            )
        
        logger.info(f"Set {len(images)} queried images for annotation")
    
    def finalize_cycle(self, results: Dict[str, Any]) -> None:
        """
        Finalize the current cycle with results.
        
        Args:
            results: Cycle completion results
        """
        self._world_state.complete_cycle(results)
        
        # Persist cycle summary to database
        if self._world_state.experiment_id:
            self._db_manager.insert_cycle_summary(
                self._world_state.experiment_id,
                self._world_state.current_cycle,
                results
            )
        
        logger.info(f"Cycle {self._world_state.current_cycle} finalized")
    
    # Database Accessors (from SQLite - paginated)
    
    def get_results_history(self, page: int = 1, limit: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated experiment history from database.
        
        Args:
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (experiments_list, total_count)
        """
        experiments, total_count = self._db_manager.get_experiments_paginated(page, limit)
        
        # Format experiments for view layer
        formatted_experiments = []
        for exp in experiments:
            formatted_exp = exp.copy()
            formatted_exp.update({
                'duration_text': self._format_duration(exp.get('created_at'), exp.get('updated_at')),
                'status_text': self._format_status_text(exp.get('phase', 'UNKNOWN')),
                'progress_text': f"{exp.get('current_cycle', 0)}/{exp.get('total_cycles', 0)} cycles"
            })
            formatted_experiments.append(formatted_exp)
        
        return formatted_experiments, total_count
    
    def get_epoch_history(self, experiment_id: str, cycle: Optional[int] = None, 
                         page: int = 1, limit: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated epoch metrics from database.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Specific cycle (optional)
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (metrics_list, total_count)
        """
        metrics, total_count = self._db_manager.get_epoch_metrics_paginated(
            experiment_id, cycle, page, limit
        )
        
        # Format metrics for view layer
        formatted_metrics = []
        for metric in metrics:
            formatted_metric = metric.copy()
            formatted_metric.update({
                'train_acc_text': f"{metric.get('train_acc', 0):.2%}",
                'val_acc_text': f"{metric.get('val_acc', 0):.2%}",
                'train_loss_text': f"{metric.get('train_loss', 0):.4f}",
                'val_loss_text': f"{metric.get('val_loss', 0):.4f}",
                'lr_text': f"{metric.get('learning_rate', 0):.6f}"
            })
            formatted_metrics.append(formatted_metric)
        
        return formatted_metrics, total_count
    
    def get_pool_page(self, pool_type: str, page: int = 1, limit: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated pool items from database.
        
        Args:
            pool_type: Type of pool ('labeled', 'unlabeled', 'test')
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (items_list, total_count)
        """
        if not self._world_state.experiment_id:
            return [], 0
        
        items, total_count = self._db_manager.get_pool_items_paginated(
            self._world_state.experiment_id, pool_type, page, limit
        )
        
        # Format items for view layer
        formatted_items = []
        for item in items:
            formatted_item = item.copy()
            formatted_item.update({
                'confidence_text': f"{item.get('confidence', 0):.2%}",
                'uncertainty_text': f"{item.get('uncertainty', 0):.3f}",
                'display_name': self._get_image_display_name(item)
            })
            formatted_items.append(formatted_item)
        
        return formatted_items, total_count
    
    # Persistence Methods
    
    def persist_epoch_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Persist epoch metrics to database.
        
        Args:
            metrics: Dictionary with epoch metrics
            
        Returns:
            True if persistence successful
        """
        if not self._world_state.experiment_id:
            logger.warning("Cannot persist metrics: no active experiment")
            return False
        
        success = self._db_manager.insert_epoch_metrics(
            self._world_state.experiment_id,
            self._world_state.current_cycle,
            metrics.get('epoch', self._world_state.current_epoch),
            metrics
        )
        
        if success:
            logger.debug(f"Persisted metrics for epoch {metrics.get('epoch')}")
        else:
            logger.error(f"Failed to persist metrics for epoch {metrics.get('epoch')}")
        
        return success
    
    def initialize_experiment(self, config: Dict[str, Any]) -> str:
        """
        Initialize a new experiment in both WorldState and database.
        
        Args:
            config: Experiment configuration dictionary
            
        Returns:
            experiment_id: Generated unique experiment ID
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration first
        validation = self.validate_config(config)
        if not validation.is_valid:
            raise ValueError(f"Invalid configuration: {validation.error_message}")
        
        # Generate experiment name if not provided
        experiment_name = config.get('name', f"Experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Initialize in WorldState
        experiment_id = self._world_state.initialize_experiment(experiment_name, config)
        
        # Persist to database
        success = self._db_manager.insert_experiment(experiment_id, experiment_name, config)
        
        if not success:
            # Rollback WorldState if database insert failed
            self._world_state.reset()
            raise RuntimeError("Failed to persist experiment to database")
        
        logger.info(f"Initialized experiment: {experiment_id}")
        return experiment_id
    
    # Validation
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate experiment configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult indicating if configuration is valid
        """
        try:
            # Create ExperimentConfig object for validation
            exp_config = ExperimentConfig.from_dict(config)
            return exp_config.validate()
            
        except Exception as e:
            return ValidationResult(False, f"Configuration validation failed: {str(e)}")
    
    # Helper Methods
    
    def _format_progress_text(self, status: Dict[str, Any]) -> str:
        """Format progress text for display."""
        if status['phase'] == 'IDLE':
            return "Ready"
        elif status['phase'] == 'INITIALIZING':
            return "Initializing..."
        elif status['phase'] == 'TRAINING':
            return f"Training - Cycle {status['current_cycle']}/{status['total_cycles']}"
        elif status['phase'] == 'AWAITING_ANNOTATION':
            return "Waiting for annotations"
        elif status['phase'] == 'ERROR':
            return "Error occurred"
        else:
            return status['phase']
    
    def _format_loss_history(self, epoch_history: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Format loss history for charting."""
        return [
            {
                'epoch': epoch.get('epoch', 0),
                'train_loss': epoch.get('train_loss', 0),
                'val_loss': epoch.get('val_loss', 0)
            }
            for epoch in epoch_history
        ]
    
    def _format_accuracy_history(self, epoch_history: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Format accuracy history for charting."""
        return [
            {
                'epoch': epoch.get('epoch', 0),
                'train_acc': epoch.get('train_acc', 0),
                'val_acc': epoch.get('val_acc', 0)
            }
            for epoch in epoch_history
        ]
    
    def _get_image_display_name(self, image_data: Dict[str, Any]) -> str:
        """Get display-friendly image name."""
        image_path = image_data.get('image_path', '')
        if image_path:
            return image_path.split('/')[-1]  # Get filename only
        return f"Image {image_data.get('image_id', 'Unknown')}"
    
    def _format_duration(self, start_time: Optional[str], end_time: Optional[str]) -> str:
        """Format duration between timestamps."""
        if not start_time:
            return "Unknown"
        
        try:
            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            end = datetime.fromisoformat(end_time.replace('Z', '+00:00')) if end_time else datetime.now()
            
            duration = end - start
            
            if duration.days > 0:
                return f"{duration.days}d {duration.seconds // 3600}h"
            elif duration.seconds > 3600:
                return f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m"
            elif duration.seconds > 60:
                return f"{duration.seconds // 60}m"
            else:
                return f"{duration.seconds}s"
                
        except Exception:
            return "Unknown"
    
    def _format_status_text(self, phase: str) -> str:
        """Format status text for display."""
        status_map = {
            'IDLE': 'Idle',
            'INITIALIZING': 'Initializing',
            'TRAINING': 'Training',
            'AWAITING_ANNOTATION': 'Awaiting Annotation',
            'ERROR': 'Error'
        }
        return status_map.get(phase, phase.title())
    
    # Additional utility methods
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive experiment summary.
        
        Returns:
            Dictionary with experiment overview
        """
        status = self.get_status()
        
        if not status['is_active']:
            return {'is_active': False}
        
        # Get pool counts if available
        pool_counts = {}
        if self._world_state.experiment_id:
            pool_counts = {
                'labeled': self._db_manager.get_pool_count(self._world_state.experiment_id, 'labeled'),
                'unlabeled': self._db_manager.get_pool_count(self._world_state.experiment_id, 'unlabeled'),
                'test': self._db_manager.get_pool_count(self._world_state.experiment_id, 'test')
            }
        
        return {
            'is_active': True,
            'experiment_id': status['experiment_id'],
            'experiment_name': status['experiment_name'],
            'phase': status['phase'],
            'current_cycle': status['current_cycle'],
            'total_cycles': status['total_cycles'],
            'progress_percentage': status['progress_percentage'],
            'pool_counts': pool_counts,
            'config': self._world_state.config
        }
    
    def reset_experiment(self) -> None:
        """Reset the current experiment state."""
        self._world_state.reset()
        logger.info("Experiment state reset")