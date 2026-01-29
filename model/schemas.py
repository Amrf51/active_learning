"""
Data schemas for the Active Learning Dashboard.

This module defines dataclasses for structured data exchange between
components. All schemas include serialization methods for JSON conversion
and validation for data integrity.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json


@dataclass
class EpochMetrics:
    """
    Training metrics for a single epoch.
    
    Contains all relevant metrics collected during one training epoch,
    including losses, accuracies, and learning rate information.
    """
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float  # learning rate
    epoch_duration: Optional[float] = None  # seconds
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the metrics
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpochMetrics':
        """
        Create EpochMetrics from dictionary.
        
        Args:
            data: Dictionary with epoch metrics data
            
        Returns:
            EpochMetrics instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate numeric fields
        numeric_fields = ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr']
        for field in numeric_fields:
            if not isinstance(data[field], (int, float)):
                raise ValueError(f"Field {field} must be numeric, got {type(data[field])}")
        
        # Validate epoch is positive integer
        if not isinstance(data['epoch'], int) or data['epoch'] < 0:
            raise ValueError(f"Epoch must be non-negative integer, got {data['epoch']}")
        
        return cls(**data)
    
    def is_valid(self) -> bool:
        """
        Validate the metrics data.
        
        Returns:
            True if all metrics are valid
        """
        try:
            # Check for NaN or infinite values
            numeric_values = [self.train_loss, self.val_loss, self.train_acc, self.val_acc, self.lr]
            for value in numeric_values:
                if not isinstance(value, (int, float)) or value != value:  # NaN check
                    return False
            
            # Check reasonable ranges
            if self.train_acc < 0 or self.train_acc > 1:
                return False
            if self.val_acc < 0 or self.val_acc > 1:
                return False
            if self.lr < 0:
                return False
            
            return True
        except Exception:
            return False


@dataclass
class CycleSummary:
    """
    Summary data for a completed active learning cycle.
    
    Contains high-level metrics and statistics for an entire AL cycle,
    including dataset statistics and final performance metrics.
    """
    cycle: int
    labeled_count: int
    unlabeled_count: int
    best_val_acc: float
    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    cycle_duration: Optional[float] = None  # seconds
    query_strategy: Optional[str] = None
    samples_queried: Optional[int] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the cycle summary
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CycleSummary':
        """
        Create CycleSummary from dictionary.
        
        Args:
            data: Dictionary with cycle summary data
            
        Returns:
            CycleSummary instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ['cycle', 'labeled_count', 'unlabeled_count', 'best_val_acc']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate cycle is non-negative integer
        if not isinstance(data['cycle'], int) or data['cycle'] < 0:
            raise ValueError(f"Cycle must be non-negative integer, got {data['cycle']}")
        
        # Validate counts are non-negative integers
        for field in ['labeled_count', 'unlabeled_count']:
            if not isinstance(data[field], int) or data[field] < 0:
                raise ValueError(f"{field} must be non-negative integer, got {data[field]}")
        
        # Validate accuracy is in valid range
        if not isinstance(data['best_val_acc'], (int, float)) or not (0 <= data['best_val_acc'] <= 1):
            raise ValueError(f"best_val_acc must be between 0 and 1, got {data['best_val_acc']}")
        
        return cls(**data)
    
    def get_total_samples(self) -> int:
        """Get total number of samples in the dataset."""
        return self.labeled_count + self.unlabeled_count
    
    def get_labeling_ratio(self) -> float:
        """Get ratio of labeled to total samples."""
        total = self.get_total_samples()
        return self.labeled_count / total if total > 0 else 0.0


@dataclass
class ExperimentConfig:
    """
    Configuration for an active learning experiment.
    
    Contains all parameters needed to configure and run an active learning
    experiment, including dataset, model, and strategy settings.
    """
    dataset: str
    model_type: str
    strategy: str
    initial_samples: int = 100
    samples_per_cycle: int = 50
    total_cycles: int = 10
    epochs_per_cycle: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """
        Create ExperimentConfig from dictionary.
        
        Args:
            data: Dictionary with configuration data
            
        Returns:
            ExperimentConfig instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        required_fields = ['dataset', 'model_type', 'strategy']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(data[field], str) or not data[field].strip():
                raise ValueError(f"Field {field} must be non-empty string")
        
        # Validate numeric parameters
        numeric_validations = {
            'initial_samples': (1, 10000),
            'samples_per_cycle': (1, 1000),
            'total_cycles': (1, 100),
            'epochs_per_cycle': (1, 1000),
            'batch_size': (1, 1024),
            'learning_rate': (1e-6, 1.0)
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            if field in data:
                value = data[field]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"{field} must be numeric, got {type(value)}")
                if not (min_val <= value <= max_val):
                    raise ValueError(f"{field} must be between {min_val} and {max_val}, got {value}")
        
        # Ensure hyperparameters is a dict
        if 'hyperparameters' in data and not isinstance(data['hyperparameters'], dict):
            raise ValueError("hyperparameters must be a dictionary")
        
        return cls(**data)
    
    def validate(self) -> 'ValidationResult':
        """
        Validate the configuration.
        
        Returns:
            ValidationResult indicating if configuration is valid
        """
        try:
            # Check required string fields
            if not self.dataset or not self.model_type or not self.strategy:
                return ValidationResult(False, "Dataset, model_type, and strategy are required")
            
            # Validate supported values (these would come from your system's capabilities)
            supported_datasets = ['cifar10', 'cifar100', 'imagenet', 'custom']
            supported_models = ['resnet18', 'resnet50', 'vgg16', 'custom']
            supported_strategies = ['uncertainty', 'diversity', 'random', 'entropy']
            
            if self.dataset not in supported_datasets:
                return ValidationResult(False, f"Unsupported dataset: {self.dataset}")
            
            if self.model_type not in supported_models:
                return ValidationResult(False, f"Unsupported model: {self.model_type}")
            
            if self.strategy not in supported_strategies:
                return ValidationResult(False, f"Unsupported strategy: {self.strategy}")
            
            # Validate numeric ranges
            if self.initial_samples < 10:
                return ValidationResult(False, "initial_samples must be at least 10")
            
            if self.samples_per_cycle < 1:
                return ValidationResult(False, "samples_per_cycle must be at least 1")
            
            if self.total_cycles < 1:
                return ValidationResult(False, "total_cycles must be at least 1")
            
            if self.epochs_per_cycle < 1:
                return ValidationResult(False, "epochs_per_cycle must be at least 1")
            
            if self.batch_size < 1:
                return ValidationResult(False, "batch_size must be at least 1")
            
            if self.learning_rate <= 0:
                return ValidationResult(False, "learning_rate must be positive")
            
            return ValidationResult(True, "Configuration is valid")
            
        except Exception as e:
            return ValidationResult(False, f"Validation error: {str(e)}")
    
    def get_estimated_duration(self) -> float:
        """
        Estimate total experiment duration in hours.
        
        Returns:
            Estimated duration in hours (rough estimate)
        """
        # Rough estimates based on typical training times
        epochs_per_hour = 10  # Depends on dataset size and model complexity
        total_epochs = self.total_cycles * self.epochs_per_cycle
        return total_epochs / epochs_per_hour


@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    Contains validation status and error message if validation failed.
    """
    is_valid: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation of the validation result
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """
        Create ValidationResult from dictionary.
        
        Args:
            data: Dictionary with validation result data
            
        Returns:
            ValidationResult instance
        """
        return cls(
            is_valid=data.get('is_valid', False),
            error_message=data.get('error_message')
        )
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid


# Utility functions for schema operations

def serialize_to_json(obj: Union[EpochMetrics, CycleSummary, ExperimentConfig, ValidationResult]) -> str:
    """
    Serialize schema object to JSON string.
    
    Args:
        obj: Schema object to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(obj.to_dict(), indent=2)


def deserialize_from_json(json_str: str, schema_class) -> Union[EpochMetrics, CycleSummary, ExperimentConfig, ValidationResult]:
    """
    Deserialize JSON string to schema object.
    
    Args:
        json_str: JSON string to deserialize
        schema_class: Target schema class
        
    Returns:
        Schema object instance
        
    Raises:
        ValueError: If JSON is invalid or doesn't match schema
    """
    try:
        data = json.loads(json_str)
        return schema_class.from_dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Failed to deserialize to {schema_class.__name__}: {e}")


def validate_schema_dict(data: Dict[str, Any], schema_class) -> ValidationResult:
    """
    Validate dictionary data against schema without creating object.
    
    Args:
        data: Dictionary to validate
        schema_class: Schema class to validate against
        
    Returns:
        ValidationResult indicating if data is valid
    """
    try:
        schema_class.from_dict(data)
        return ValidationResult(True, "Data is valid")
    except Exception as e:
        return ValidationResult(False, str(e))


# Type aliases for convenience
MetricsDict = Dict[str, Union[int, float, str]]
ConfigDict = Dict[str, Any]
SummaryDict = Dict[str, Union[int, float, str]]