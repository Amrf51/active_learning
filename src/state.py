"""
State management for Active Learning Dashboard.

This module provides:
- Pydantic models for all experiment state
- StateManager for atomic file operations with locking
- Schemas for frontend-backend communication

The state file serves as the single source of truth between
the Streamlit frontend and the training worker subprocess.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ExperimentPhase(str, Enum):
    """Possible phases of an experiment lifecycle."""
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PREPARING = "PREPARING"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    EVALUATING = "EVALUATING"
    QUERYING = "QUERYING"
    AWAITING_ANNOTATION = "AWAITING_ANNOTATION"
    ANNOTATIONS_SUBMITTED = "ANNOTATIONS_SUBMITTED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"
    ABORT = "ABORT"


class Command(str, Enum):
    """Commands for dashboard-worker communication."""
    START_CYCLE = "START_CYCLE"
    PAUSE = "PAUSE"
    STOP = "STOP"
    CONTINUE = "CONTINUE"


class EpochMetrics(BaseModel):
    """Metrics for a single training epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    learning_rate: Optional[float] = None


class CycleMetrics(BaseModel):
    """Aggregated metrics for a complete AL cycle."""
    cycle: int
    labeled_pool_size: int
    unlabeled_pool_size: int
    epochs_trained: int
    best_val_accuracy: float
    best_epoch: int
    test_accuracy: float
    test_f1: float
    test_precision: float
    test_recall: float
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    epoch_history: List[EpochMetrics] = Field(default_factory=list)
    annotation_accuracy: Optional[float] = None


class QueriedImage(BaseModel):
    """Information about an image selected for annotation."""
    image_id: int
    image_path: str
    display_path: str
    ground_truth: int
    ground_truth_name: str
    model_probabilities: Dict[str, float]
    predicted_class: str
    predicted_confidence: float
    uncertainty_score: float
    selection_reason: str


class ProbeImage(BaseModel):
    """A probe image tracked across all cycles."""
    image_id: int
    image_path: str
    display_path: str
    true_class: str
    true_class_idx: int
    probe_type: str
    predictions_by_cycle: Dict[int, Dict[str, float]] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Experiment configuration snapshot (immutable after start)."""
    model_name: str
    pretrained: bool
    num_classes: int
    class_names: List[str]
    epochs_per_cycle: int
    batch_size: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    early_stopping_patience: int
    num_cycles: int
    sampling_strategy: str
    uncertainty_method: str
    initial_pool_size: int
    batch_size_al: int
    reset_mode: str
    seed: int
    data_dir: str
    val_split: float
    test_split: float
    augmentation: bool


class DatasetInfo(BaseModel):
    """Information about the dataset."""
    total_images: int
    num_classes: int
    class_names: List[str]
    class_counts: Dict[str, int]
    train_samples: int
    val_samples: int
    test_samples: int


class ExperimentState(BaseModel):
    """
    Complete experiment state - the single source of truth.
    
    This model is serialized to JSON and shared between
    the Streamlit frontend and the training worker.
    """
    experiment_id: str
    experiment_name: str
    created_at: datetime
    
    phase: ExperimentPhase = ExperimentPhase.IDLE
    command: Optional[Command] = None
    worker_pid: Optional[int] = None
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None
    
    config: Optional[ExperimentConfig] = None
    
    current_cycle: int = 0
    total_cycles: int = 0
    current_epoch: int = 0
    epochs_in_cycle: int = 0
    
    labeled_count: int = 0
    unlabeled_count: int = 0
    total_train_samples: int = 0
    
    cycle_results: List[CycleMetrics] = Field(default_factory=list)
    current_cycle_epochs: List[EpochMetrics] = Field(default_factory=list)
    
    queried_images: List[QueriedImage] = Field(default_factory=list)
    
    probe_images: List[ProbeImage] = Field(default_factory=list)
    
    dataset_info: Optional[DatasetInfo] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserAnnotation(BaseModel):
    """A single user annotation for one image."""
    image_id: int
    user_label: int
    user_label_name: str
    timestamp: datetime
    was_correct: Optional[bool] = None


class AnnotationSubmission(BaseModel):
    """Batch of annotations submitted by user."""
    experiment_id: str
    cycle: int
    annotations: List[UserAnnotation]
    submitted_at: datetime


class ActiveExperimentPointer(BaseModel):
    """Points to the currently active experiment."""
    experiment_id: Optional[str] = None
    experiment_dir: Optional[str] = None
    started_at: Optional[datetime] = None


class StateManager:
    """
    Manages experiment state with atomic file operations.
    
    Uses file locking to prevent race conditions between
    the Streamlit frontend (reader) and worker (writer).
    
    Usage:
        manager = StateManager(Path("experiments/exp_001"))
        state = manager.read_state()
        manager.update_state(phase=ExperimentPhase.TRAINING)
    """
    
    STATE_FILENAME = "experiment_state.json"
    ANNOTATIONS_FILENAME = "user_annotations.json"
    LOCK_FILENAME = ".state.lock"
    LOCK_TIMEOUT = 10
    
    def __init__(self, experiment_dir: Path):
        """
        Initialize StateManager for an experiment directory.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        self.experiment_dir = Path(experiment_dir)
        self.state_file = self.experiment_dir / self.STATE_FILENAME
        self.annotations_file = self.experiment_dir / self.ANNOTATIONS_FILENAME
        self.lock_file = self.experiment_dir / self.LOCK_FILENAME
        self._lock = FileLock(str(self.lock_file), timeout=self.LOCK_TIMEOUT)
    
    def initialize_state(
        self,
        experiment_id: str,
        experiment_name: str
    ) -> ExperimentState:
        """
        Create and save a new experiment state.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable name
            
        Returns:
            Newly created ExperimentState
        """
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        state = ExperimentState(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            created_at=datetime.now(),
            phase=ExperimentPhase.IDLE
        )
        
        self.write_state(state)
        logger.info(f"Initialized experiment state: {experiment_id}")
        
        return state
    
    def read_state(self) -> ExperimentState:
        """
        Read current experiment state from file.
        
        Returns:
            Current ExperimentState
            
        Raises:
            FileNotFoundError: If state file doesn't exist
            Timeout: If lock cannot be acquired
        """
        try:
            with self._lock:
                if not self.state_file.exists():
                    raise FileNotFoundError(
                        f"State file not found: {self.state_file}"
                    )
                content = self.state_file.read_text(encoding="utf-8")
                return ExperimentState.model_validate_json(content)
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.state_file}")
            raise
    
    def write_state(self, state: ExperimentState) -> None:
        """
        Write experiment state to file atomically.
        
        Args:
            state: ExperimentState to save
        """
        try:
            with self._lock:
                self.experiment_dir.mkdir(parents=True, exist_ok=True)
                
                temp_file = self.state_file.with_suffix(".tmp")
                temp_file.write_text(
                    state.model_dump_json(indent=2),
                    encoding="utf-8"
                )
                temp_file.replace(self.state_file)
                
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.state_file}")
            raise
    
    def update_state(self, **updates) -> ExperimentState:
        """
        Read-modify-write state with specific field updates.
        
        Args:
            **updates: Fields to update
            
        Returns:
            Updated ExperimentState
        """
        try:
            with self._lock:
                content = self.state_file.read_text(encoding="utf-8")
                state = ExperimentState.model_validate_json(content)
                
                for key, value in updates.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
                    else:
                        logger.warning(f"Unknown state field: {key}")
                
                temp_file = self.state_file.with_suffix(".tmp")
                temp_file.write_text(
                    state.model_dump_json(indent=2),
                    encoding="utf-8"
                )
                temp_file.replace(self.state_file)
                
                return state
                
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.state_file}")
            raise
    
    def update_heartbeat(self) -> None:
        """Update the heartbeat timestamp."""
        self.update_state(last_heartbeat=datetime.now())
    
    def add_epoch_metrics(self, metrics: EpochMetrics) -> None:
        """
        Append epoch metrics to current cycle history.
        
        Args:
            metrics: EpochMetrics to add
        """
        try:
            with self._lock:
                content = self.state_file.read_text(encoding="utf-8")
                state = ExperimentState.model_validate_json(content)
                
                state.current_cycle_epochs.append(metrics)
                state.current_epoch = metrics.epoch
                state.last_heartbeat = datetime.now()
                
                temp_file = self.state_file.with_suffix(".tmp")
                temp_file.write_text(
                    state.model_dump_json(indent=2),
                    encoding="utf-8"
                )
                temp_file.replace(self.state_file)
                
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.state_file}")
            raise
    
    def finalize_cycle(self, cycle_metrics: CycleMetrics) -> None:
        """
        Record completed cycle metrics and reset epoch tracking.
        
        Args:
            cycle_metrics: Completed cycle metrics
        """
        try:
            with self._lock:
                content = self.state_file.read_text(encoding="utf-8")
                state = ExperimentState.model_validate_json(content)
                
                cycle_metrics.epoch_history = state.current_cycle_epochs.copy()
                state.cycle_results.append(cycle_metrics)
                state.current_cycle_epochs = []
                state.last_heartbeat = datetime.now()
                
                temp_file = self.state_file.with_suffix(".tmp")
                temp_file.write_text(
                    state.model_dump_json(indent=2),
                    encoding="utf-8"
                )
                temp_file.replace(self.state_file)
                
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.state_file}")
            raise
    
    def set_queried_images(self, images: List[QueriedImage]) -> None:
        """
        Set the list of images awaiting annotation.
        
        Args:
            images: List of QueriedImage objects
        """
        self.update_state(
            queried_images=images,
            phase=ExperimentPhase.AWAITING_ANNOTATION
        )
    
    def read_annotations(self) -> Optional[AnnotationSubmission]:
        """
        Read user annotations if available.
        
        Returns:
            AnnotationSubmission or None if not present
        """
        try:
            with self._lock:
                if not self.annotations_file.exists():
                    return None
                content = self.annotations_file.read_text(encoding="utf-8")
                return AnnotationSubmission.model_validate_json(content)
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.annotations_file}")
            raise
    
    def write_annotations(self, submission: AnnotationSubmission) -> None:
        """
        Write user annotations to file.
        
        Args:
            submission: AnnotationSubmission to save
        """
        try:
            with self._lock:
                temp_file = self.annotations_file.with_suffix(".tmp")
                temp_file.write_text(
                    submission.model_dump_json(indent=2),
                    encoding="utf-8"
                )
                temp_file.replace(self.annotations_file)
                
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.annotations_file}")
            raise
    
    def clear_annotations(self) -> None:
        """Delete annotations file after processing."""
        try:
            with self._lock:
                if self.annotations_file.exists():
                    self.annotations_file.unlink()
                    logger.info("Annotations file cleared")
        except Timeout:
            logger.error(f"Timeout acquiring lock for {self.annotations_file}")
            raise
    
    def state_exists(self) -> bool:
        """Check if state file exists."""
        return self.state_file.exists()
    
    def annotations_pending(self) -> bool:
        """Check if annotations file exists (user submitted)."""
        return self.annotations_file.exists()
    
    def get_phase(self) -> ExperimentPhase:
        """Quick read of just the current phase."""
        state = self.read_state()
        return state.phase
    
    def is_worker_alive(self, timeout_seconds: int = 30) -> bool:
        """
        Check if worker is alive based on heartbeat.
        
        Args:
            timeout_seconds: Seconds before considering worker dead
            
        Returns:
            True if heartbeat is recent, False otherwise
        """
        state = self.read_state()
        
        if state.last_heartbeat is None:
            return False
        
        age = datetime.now() - state.last_heartbeat
        return age.total_seconds() < timeout_seconds
    
    def set_command(self, command: Command) -> None:
        """
        Set a command for the worker to execute.
        
        Args:
            command: Command to set
        """
        self.update_state(command=command)
        logger.info(f"Command set: {command}")
    
    def clear_command(self) -> None:
        """Clear the current command."""
        self.update_state(command=None)
        logger.info("Command cleared")
    
    def update_probe_images(self, probe_images: List[ProbeImage]) -> None:
        """
        Update probe images in state.
        
        Args:
            probe_images: List of ProbeImage objects
        """
        self.update_state(probe_images=probe_images)
        logger.info(f"Updated {len(probe_images)} probe images")


class ExperimentManager:
    """
    Manages multiple experiments and tracks the active one.
    
    Usage:
        manager = ExperimentManager(Path("experiments"))
        exp_dir = manager.create_experiment("uncertainty_test")
        manager.set_active(exp_dir)
    """
    
    ACTIVE_POINTER_FILE = "active_experiment.json"
    
    def __init__(self, base_dir: Path):
        """
        Initialize ExperimentManager.
        
        Args:
            base_dir: Base directory for all experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.pointer_file = self.base_dir / self.ACTIVE_POINTER_FILE
    
    def create_experiment(
        self,
        name: str,
        experiment_id: Optional[str] = None
    ) -> Path:
        """
        Create a new experiment directory.
        
        Args:
            name: Experiment name
            experiment_id: Optional custom ID (auto-generated if None)
            
        Returns:
            Path to new experiment directory
        """
        if experiment_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{name}_{timestamp}"
        
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "queries").mkdir(exist_ok=True)
        (exp_dir / "probes").mkdir(exist_ok=True)
        
        logger.info(f"Created experiment directory: {exp_dir}")
        
        return exp_dir
    
    def set_active(self, experiment_dir: Path) -> None:
        """
        Set an experiment as the active one.
        
        Args:
            experiment_dir: Path to experiment directory
        """
        pointer = ActiveExperimentPointer(
            experiment_id=experiment_dir.name,
            experiment_dir=str(experiment_dir),
            started_at=datetime.now()
        )
        
        self.pointer_file.write_text(
            pointer.model_dump_json(indent=2),
            encoding="utf-8"
        )
        
        logger.info(f"Set active experiment: {experiment_dir.name}")
    
    def get_active(self) -> Optional[ActiveExperimentPointer]:
        """
        Get the currently active experiment.
        
        Returns:
            ActiveExperimentPointer or None if no active experiment
        """
        if not self.pointer_file.exists():
            return None
        
        content = self.pointer_file.read_text(encoding="utf-8")
        return ActiveExperimentPointer.model_validate_json(content)
    
    def clear_active(self) -> None:
        """Clear the active experiment pointer."""
        if self.pointer_file.exists():
            self.pointer_file.unlink()
            logger.info("Cleared active experiment")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments with basic info.
        
        Returns:
            List of experiment info dicts
        """
        experiments = []
        
        for exp_dir in sorted(self.base_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if exp_dir.name.startswith("."):
                continue
            
            state_file = exp_dir / StateManager.STATE_FILENAME
            
            info = {
                "experiment_id": exp_dir.name,
                "path": str(exp_dir),
                "has_state": state_file.exists(),
                "created": None,
                "phase": None
            }
            
            if state_file.exists():
                try:
                    manager = StateManager(exp_dir)
                    state = manager.read_state()
                    info["created"] = state.created_at.isoformat()
                    info["phase"] = state.phase.value
                    info["experiment_name"] = state.experiment_name
                except Exception as e:
                    logger.warning(f"Could not read state for {exp_dir}: {e}")
            
            experiments.append(info)
        
        return experiments
    
    def is_experiment_running(self) -> bool:
        """
        Check if any experiment is currently running.
        
        Returns:
            True if an experiment is in an active phase
        """
        active = self.get_active()
        if active is None or active.experiment_dir is None:
            return False
        
        try:
            manager = StateManager(Path(active.experiment_dir))
            state = manager.read_state()
            
            active_phases = {
                ExperimentPhase.INITIALIZING,
                ExperimentPhase.PREPARING,
                ExperimentPhase.TRAINING,
                ExperimentPhase.VALIDATING,
                ExperimentPhase.EVALUATING,
                ExperimentPhase.QUERYING,
                ExperimentPhase.AWAITING_ANNOTATION,
                ExperimentPhase.ANNOTATIONS_SUBMITTED
            }
            
            return state.phase in active_phases
            
        except FileNotFoundError:
            return False


def create_experiment_config_from_config(config) -> ExperimentConfig:
    """
    Create ExperimentConfig from a Config object.
    
    Args:
        config: Config object from config.py
        
    Returns:
        ExperimentConfig for state storage
    """
    return ExperimentConfig(
        model_name=config.model.name,
        pretrained=config.model.pretrained,
        num_classes=config.model.num_classes,
        class_names=[],
        epochs_per_cycle=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        optimizer=config.training.optimizer,
        weight_decay=config.training.weight_decay,
        early_stopping_patience=config.training.early_stopping_patience,
        num_cycles=config.active_learning.num_cycles,
        sampling_strategy=config.active_learning.sampling_strategy,
        uncertainty_method=config.active_learning.uncertainty_method,
        initial_pool_size=config.active_learning.initial_pool_size,
        batch_size_al=config.active_learning.batch_size_al,
        reset_mode=config.active_learning.reset_mode,
        seed=config.training.seed,
        data_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        augmentation=config.data.augmentation
    )