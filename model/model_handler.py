"""ModelHandler - Orchestrates backend operations and updates WorldState.

The ModelHandler is the core of the model layer. It:
1. Processes events from the controller
2. Orchestrates backend services (ActiveLearningLoop, Trainer, etc.)
3. Updates WorldState after each operation
4. Handles errors and state transitions
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import torch

from .world_state import WorldState, Phase, EpochMetrics, QueriedImage, ProbeImage
from .experiment_manager import ExperimentManager
from controller.events import Event, EventType

logger = logging.getLogger(__name__)


class ModelHandler:
    """Orchestrates backend operations and manages WorldState."""
    
    def __init__(self, experiments_dir: Path):
        """Initialize ModelHandler.
        
        Args:
            experiments_dir: Root directory for experiment storage
        """
        self.experiments_dir = Path(experiments_dir)
        self.world_state = WorldState()
        self.exp_manager = ExperimentManager(experiments_dir)
        
        # Backend components (initialized when experiment is created/loaded)
        self._active_loop: Optional[Any] = None  # ActiveLearningLoop instance
        self._config: Optional[Any] = None  # Experiment config
        
        # Control flags
        self._pause_requested = False
        self._stop_requested = False
        
        logger.info(f"ModelHandler initialized with experiments_dir: {experiments_dir}")
    
    def handle_event(self, event: Event) -> None:
        """Process event and update state.
        
        Args:
            event: Event to process
        """
        try:
            if event.type == EventType.CREATE_EXPERIMENT:
                self._create_experiment(event.payload)
            elif event.type == EventType.LOAD_EXPERIMENT:
                self._load_experiment(event.payload)
            elif event.type == EventType.START_CYCLE:
                self._start_cycle(event.payload)
            elif event.type == EventType.SUBMIT_ANNOTATIONS:
                self._submit_annotations(event.payload)
            elif event.type == EventType.CONTINUE:
                self._continue()
            elif event.type == EventType.PAUSE:
                self._pause()
            elif event.type == EventType.STOP:
                self._stop()
            else:
                logger.warning(f"Unknown event type: {event.type}")
        
        except Exception as e:
            logger.error(f"Error handling event {event.type}: {e}", exc_info=True)
            self.world_state.phase = Phase.ERROR
            self.world_state.error_message = str(e)
    
    def _create_experiment(self, payload: Dict[str, Any]) -> None:
        """Handle CREATE_EXPERIMENT event.
        
        Args:
            payload: Dictionary containing:
                - experiment_name: Name of the experiment
                - config: ExperimentConfig dictionary
                - dataset_info: DatasetInfo dictionary (optional)
        """
        logger.info("Creating new experiment")
        
        experiment_name = payload.get('experiment_name', 'Unnamed Experiment')
        config = payload.get('config', {})
        
        # Create experiment via ExperimentManager
        exp_id = self.exp_manager.create_experiment(payload)
        
        # Store config for later use
        self._config = config
        
        # Update WorldState
        self.world_state.experiment_id = exp_id
        self.world_state.experiment_name = experiment_name
        self.world_state.total_cycles = config.get('num_cycles', 0)
        self.world_state.epochs_per_cycle = config.get('epochs_per_cycle', 0)
        self.world_state.current_cycle = 0
        self.world_state.current_epoch = 0
        self.world_state.phase = Phase.IDLE
        self.world_state.error_message = None
        
        # Initialize pool sizes from config
        self.world_state.labeled_count = config.get('initial_pool_size', 0)
        # Unlabeled count will be set when ActiveLearningLoop is initialized
        
        logger.info(f"Experiment created: {exp_id} (name: {experiment_name})")
        
        # CRITICAL FIX: Initialize backend components
        try:
            self._initialize_backend_components(config, exp_id)
            logger.info("Backend components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize backend components: {e}", exc_info=True)
            self.world_state.phase = Phase.ERROR
            self.world_state.error_message = f"Backend initialization failed: {str(e)}"
            raise
    
    def _load_experiment(self, payload: Dict[str, Any]) -> None:
        """Handle LOAD_EXPERIMENT event.
        
        Args:
            payload: Dictionary containing 'experiment_id'
        """
        exp_id = payload.get('experiment_id')
        if not exp_id:
            raise ValueError("experiment_id required in payload")
        
        logger.info(f"Loading experiment: {exp_id}")
        
        # Load experiment via ExperimentManager
        exp_data = self.exp_manager.load_experiment(exp_id)
        config = exp_data['config']
        
        # Restore WorldState
        self.world_state.experiment_id = exp_id
        self.world_state.experiment_name = exp_data['name']
        self.world_state.total_cycles = config.get('num_cycles', 0)
        self.world_state.epochs_per_cycle = config.get('epochs_per_cycle', 0)
        self.world_state.phase = Phase.IDLE
        self.world_state.error_message = None
        
        # Load cycle results to determine current cycle
        cycle_results = self.exp_manager.get_cycle_results(exp_id)
        self.world_state.current_cycle = len(cycle_results)
        
        logger.info(f"Experiment loaded: {exp_id}, current cycle: {self.world_state.current_cycle}")
    
    def _start_cycle(self, payload: Dict[str, Any]) -> None:
        """Handle START_CYCLE event.
        
        Sets phase to TRAINING, prepares ActiveLearningLoop for the cycle.
        
        Args:
            payload: Dictionary containing cycle configuration
        """
        if self._active_loop is None:
            raise RuntimeError("ActiveLearningLoop not initialized. Create or load experiment first.")
        
        logger.info(f"Starting cycle {self.world_state.current_cycle + 1}")
        
        # Set phase to TRAINING
        self.world_state.phase = Phase.TRAINING
        self.world_state.current_epoch = 0
        self.world_state.epoch_metrics = []
        
        # Prepare cycle in ActiveLearningLoop
        cycle_num = self.world_state.current_cycle + 1
        cycle_info = self._active_loop.prepare_cycle(cycle_num)
        
        # Update pool sizes
        self.world_state.labeled_count = cycle_info['labeled_count']
        self.world_state.unlabeled_count = cycle_info['unlabeled_count']
        
        logger.info(f"Cycle {cycle_num} prepared: labeled={cycle_info['labeled_count']}, "
                   f"unlabeled={cycle_info['unlabeled_count']}")
    
    def train_epoch(self) -> EpochMetrics:
        """Train one epoch and update WorldState.
        
        This method is called by the controller's training thread.
        
        Returns:
            EpochMetrics from the training epoch
        """
        if self._active_loop is None:
            raise RuntimeError("ActiveLearningLoop not initialized")
        
        if self.world_state.phase != Phase.TRAINING:
            raise RuntimeError(f"Cannot train in phase {self.world_state.phase}")
        
        # Train one epoch
        epoch_num = self.world_state.current_epoch + 1
        metrics = self._active_loop.train_single_epoch(epoch_num)
        
        # Update WorldState
        self.world_state.epoch_metrics.append(metrics)
        self.world_state.current_epoch = epoch_num
        
        logger.info(f"Epoch {epoch_num} complete: train_loss={metrics.train_loss:.4f}, "
                   f"train_acc={metrics.train_accuracy:.4f}")
        
        # Check if training is complete
        if epoch_num >= self.world_state.epochs_per_cycle:
            self._finish_training()
        
        return metrics
    
    def _finish_training(self) -> None:
        """Finish training phase and transition to querying."""
        logger.info("Training complete, transitioning to QUERYING")
        
        # Transition to QUERYING phase
        self.world_state.phase = Phase.QUERYING
        
        # Run evaluation
        test_metrics = self._active_loop.run_evaluation()
        
        # Query samples using AL strategy
        queried_images = self._active_loop.query_samples()
        
        # Convert backend QueriedImage to world state format
        self.world_state.queried_images = queried_images
        
        # Get probe images if available
        if self._active_loop.probe_images:
            self.world_state.probe_images = self._active_loop.probe_images
        
        # Transition to AWAITING_ANNOTATION
        self.world_state.phase = Phase.AWAITING_ANNOTATION
        
        logger.info(f"Queried {len(queried_images)} samples for annotation")
    
    def _submit_annotations(self, payload: Dict[str, Any]) -> None:
        """Handle SUBMIT_ANNOTATIONS event.
        
        Passes annotations to ActiveLearningLoop, saves cycle results,
        increments cycle counter, and transitions to COMPLETED if final cycle.
        
        Args:
            payload: Dictionary containing 'annotations' list
        """
        if self._active_loop is None:
            raise RuntimeError("ActiveLearningLoop not initialized")
        
        if self.world_state.phase != Phase.AWAITING_ANNOTATION:
            raise RuntimeError(f"Cannot submit annotations in phase {self.world_state.phase}")
        
        annotations = payload.get('annotations', [])
        if not annotations:
            raise ValueError("No annotations provided")
        
        logger.info(f"Submitting {len(annotations)} annotations")
        
        # Pass annotations to ActiveLearningLoop
        annotation_result = self._active_loop.receive_annotations(annotations)
        
        # Get test metrics from the last evaluation
        test_metrics = self._active_loop.run_evaluation()
        
        # Finalize cycle and get metrics
        cycle_metrics = self._active_loop.finalize_cycle(test_metrics)
        
        # Save cycle results via ExperimentManager
        cycle_data = {
            'cycle': cycle_metrics.cycle,
            'labeled_count': cycle_metrics.labeled_pool_size,
            'unlabeled_count': cycle_metrics.unlabeled_pool_size,
            'epochs_trained': cycle_metrics.epochs_trained,
            'best_val_accuracy': cycle_metrics.best_val_accuracy,
            'test_accuracy': cycle_metrics.test_accuracy,
            'test_f1': cycle_metrics.test_f1,
            'test_precision': cycle_metrics.test_precision,
            'test_recall': cycle_metrics.test_recall,
            'per_class_metrics': cycle_metrics.per_class_metrics
        }
        
        self.exp_manager.save_cycle_result(self.world_state.experiment_id, cycle_data)
        
        # Update pool sizes
        pool_info = self._active_loop.get_current_pool_info()
        self.world_state.labeled_count = pool_info['labeled']
        self.world_state.unlabeled_count = pool_info['unlabeled']
        
        # Increment current_cycle
        self.world_state.current_cycle += 1
        
        # Clear queried images
        self.world_state.queried_images = []
        
        # Check if all cycles are completed
        if self.world_state.current_cycle >= self.world_state.total_cycles:
            self.world_state.phase = Phase.COMPLETED
            logger.info("All cycles completed")
        else:
            self.world_state.phase = Phase.IDLE
            logger.info(f"Cycle {cycle_metrics.cycle} complete, ready for next cycle")
        
        logger.info(f"Annotations processed: {annotation_result['moved_count']} samples added to labeled pool")
    
    def _continue(self) -> None:
        """Handle CONTINUE event.
        
        This is used after annotations are submitted to continue to the next cycle.
        It's essentially a no-op that confirms the user wants to proceed.
        """
        logger.info("Continue requested")
        
        # If we're in AWAITING_ANNOTATION phase, this means annotations were already
        # submitted and we just need to acknowledge and stay in IDLE
        if self.world_state.phase == Phase.AWAITING_ANNOTATION:
            logger.warning("Continue requested but annotations not yet submitted")
        else:
            logger.info("Continue acknowledged, ready for next cycle")
    
    def _pause(self) -> None:
        """Handle PAUSE event.
        
        Sets flag to pause training loop. The controller's training thread
        should check this flag and suspend execution.
        """
        logger.info("Pause requested")
        
        self._pause_requested = True
        
        # Update WorldState phase if currently training
        if self.world_state.phase == Phase.TRAINING:
            self.world_state.phase = Phase.IDLE
            logger.info("Training paused")
    
    def _stop(self) -> None:
        """Handle STOP event.
        
        Terminates training, saves current state.
        """
        logger.info("Stop requested")
        
        self._stop_requested = True
        self._pause_requested = False
        
        # Save current state if experiment is active
        if self.world_state.experiment_id and self._active_loop:
            try:
                # Save current pool state
                exp_dir = self.experiments_dir / self.world_state.experiment_id
                self._active_loop.data_manager.save_state(exp_dir / "al_pool_state.json")
                logger.info("Pool state saved")
            except Exception as e:
                logger.error(f"Failed to save state on stop: {e}")
        
        # Update WorldState phase
        if self.world_state.phase == Phase.TRAINING:
            self.world_state.phase = Phase.IDLE
            logger.info("Training stopped")
    
    def is_paused(self) -> bool:
        """Check if pause is requested."""
        return self._pause_requested
    
    def is_stopped(self) -> bool:
        """Check if stop is requested."""
        return self._stop_requested
    
    def clear_control_flags(self) -> None:
        """Clear pause and stop flags."""
        self._pause_requested = False
        self._stop_requested = False
    
    def _initialize_backend_components(self, config: Dict[str, Any], exp_id: str) -> None:
        """Initialize backend components (Trainer, ALDataManager, ActiveLearningLoop).
        
        This is the critical missing link between MVC and the existing backend.
        
        Args:
            config: Experiment configuration dictionary
            exp_id: Experiment ID
        """
        from backend.trainer import Trainer
        from backend.data_manager import ALDataManager
        from backend.active_loop import ActiveLearningLoop
        from backend.strategies import get_strategy
        from backend.dataloader import get_datasets
        from backend.models import get_model
        
        logger.info("Initializing backend components...")
        
        # 1. Validate data directory
        data_dir = Path(config['data_dir'])
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
        
        # 2. Create datasets with splits
        logger.info(f"Loading dataset from: {data_dir}")
        datasets = get_datasets(
            data_dir=str(data_dir),
            val_split=config.get('val_split', 0.15),
            test_split=config.get('test_split', 0.15),
            augmentation=config.get('augmentation', True),
            seed=config.get('seed', 42)
        )
        
        train_dataset = datasets['train_dataset']
        val_dataset = datasets['val_dataset']
        test_dataset = datasets['test_dataset']
        class_names = datasets['class_names']
        
        logger.info(f"Dataset loaded: {datasets['splits_info']['train_samples']} train, "
                   f"{datasets['splits_info']['val_samples']} val, "
                   f"{datasets['splits_info']['test_samples']} test")
        
        # 3. Create DataLoaders for val and test (train loader created by ALDataManager)
        from torch.utils.data import DataLoader
        
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)
        pin_memory = torch.cuda.is_available()
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # 4. Create ALDataManager
        logger.info("Creating ALDataManager...")
        data_manager = ALDataManager(
            dataset=train_dataset,
            initial_pool_size=config.get('initial_pool_size', 50),
            seed=config.get('seed', 42),
            exp_dir=self.experiments_dir / exp_id
        )
        
        # 5. Create model
        logger.info(f"Creating model: {config.get('model_name', 'resnet18')}")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create a simple config object for get_model()
        class ModelConfig:
            def __init__(self, name, num_classes, pretrained):
                self.name = name
                self.num_classes = num_classes
                self.pretrained = pretrained
        
        model_config = ModelConfig(
            name=config.get('model_name', 'resnet18'),
            num_classes=config.get('num_classes', len(class_names)),
            pretrained=config.get('pretrained', True)
        )
        model = get_model(model_config, device=device)
        
        # 6. Create a minimal config object for Trainer
        # The Trainer expects a config object with nested attributes
        class SimpleConfig:
            def __init__(self, config_dict):
                self.training = type('obj', (object,), {
                    'optimizer': config_dict.get('optimizer', 'adam'),
                    'learning_rate': config_dict.get('learning_rate', 0.001),
                    'weight_decay': config_dict.get('weight_decay', 1e-4),
                    'early_stopping_patience': config_dict.get('early_stopping_patience', 5)
                })()
                self.model = type('obj', (object,), {
                    'name': config_dict.get('model_name', 'resnet18'),
                    'num_classes': config_dict.get('num_classes', len(class_names)),
                    'pretrained': config_dict.get('pretrained', True)
                })()
                self.active_learning = type('obj', (object,), {
                    'num_cycles': config_dict.get('num_cycles', 5),
                    'batch_size_al': config_dict.get('batch_size_al', 10),
                    'sampling_strategy': config_dict.get('sampling_strategy', 'uncertainty'),
                    'uncertainty_method': config_dict.get('uncertainty_method', 'entropy'),
                    'reset_mode': config_dict.get('reset_mode', 'pretrained')
                })()
                self.data = type('obj', (object,), {
                    'num_workers': config_dict.get('num_workers', 4)
                })()
                self.checkpoint = type('obj', (object,), {
                    'save_best_per_cycle': True
                })()
        
        config_obj = SimpleConfig(config)
        
        # 7. Create Trainer
        logger.info("Creating Trainer...")
        trainer = Trainer(
            model=model,
            config=config_obj,
            exp_dir=self.experiments_dir / exp_id,
            device=device
        )
        
        # 8. Get AL strategy
        logger.info(f"Getting strategy: {config.get('sampling_strategy', 'uncertainty')}")
        strategy = get_strategy(
            strategy_name=config.get('sampling_strategy', 'uncertainty'),
            uncertainty_method=config.get('uncertainty_method', 'entropy')
        )
        
        # 9. Create ActiveLearningLoop
        logger.info("Creating ActiveLearningLoop...")
        self._active_loop = ActiveLearningLoop(
            trainer=trainer,
            data_manager=data_manager,
            strategy=strategy,
            val_loader=val_loader,
            test_loader=test_loader,
            exp_dir=self.experiments_dir / exp_id,
            config=config_obj,
            class_names=class_names
        )
        
        # 10. Update WorldState with actual pool sizes
        pool_info = data_manager.get_pool_info()
        self.world_state.labeled_count = pool_info['labeled']
        self.world_state.unlabeled_count = pool_info['unlabeled']
        
        logger.info(f"Backend initialization complete - Labeled: {pool_info['labeled']}, "
                   f"Unlabeled: {pool_info['unlabeled']}")
