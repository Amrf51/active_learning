"""ExperimentManager - Persistence layer for experiments using SQLite and file storage."""

import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid


class ExperimentManager:
    """Handles persistence of experiments, cycle results, and artifacts."""
    
    def __init__(self, experiments_dir: Path):
        """Initialize ExperimentManager with experiments directory.
        
        Args:
            experiments_dir: Root directory for experiment storage
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.experiments_dir / "experiments.db"
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config TEXT NOT NULL,
                status TEXT DEFAULT 'created',
                dataset_path TEXT,
                model_name TEXT,
                strategy TEXT
            )
        """)
        
        # Cycle results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycle_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT REFERENCES experiments(id),
                cycle INTEGER NOT NULL,
                labeled_count INTEGER,
                unlabeled_count INTEGER,
                epochs_trained INTEGER,
                best_val_accuracy REAL,
                test_accuracy REAL,
                test_f1 REAL,
                test_precision REAL,
                test_recall REAL,
                per_class_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(experiment_id, cycle)
            )
        """)
        
        # Epoch metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT REFERENCES experiments(id),
                cycle INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_loss REAL,
                train_accuracy REAL,
                val_loss REAL,
                val_accuracy REAL,
                UNIQUE(experiment_id, cycle, epoch)
            )
        """)
        
        conn.commit()
        conn.close()

    def _sanitize_folder_name(self, name: str) -> str:
        """Sanitize experiment name for filesystem use.
        
        Removes or replaces special characters that are invalid in folder names.
        
        Args:
            name: Raw experiment name
        
        Returns:
            Sanitized folder name safe for filesystem use
        """
        # Replace spaces with underscores
        sanitized = name.replace(' ', '_')
        
        # Remove or replace invalid characters for Windows/Unix filesystems
        # Invalid: < > : " / \ | ? *
        sanitized = re.sub(r'[<>:"/\\|?*]', '', sanitized)
        
        # Remove leading/trailing dots and spaces (Windows doesn't like these)
        sanitized = sanitized.strip('. ')
        
        # Ensure name is not empty
        if not sanitized:
            sanitized = 'experiment'
        
        # Limit length to avoid filesystem issues (255 chars is common limit)
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        
        return sanitized

    def _get_unique_folder_name(self, base_name: str) -> str:
        """Get a unique folder name by appending suffix if needed.
        
        Args:
            base_name: Base folder name (already sanitized)
        
        Returns:
            Unique folder name that doesn't exist yet
        """
        folder_path = self.experiments_dir / base_name
        
        # If folder doesn't exist, use base name
        if not folder_path.exists():
            return base_name
        
        # Append numeric suffix until we find a unique name
        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if not (self.experiments_dir / candidate).exists():
                return candidate
            counter += 1

    def create_experiment(self, config: Dict[str, Any]) -> str:
        """Create a new experiment with unique ID and folder structure.
        
        The experiment folder is named using the user-provided experiment_name
        (sanitized for filesystem use) instead of the UUID. The UUID is still
        used as the database primary key.
        
        Args:
            config: Experiment configuration dictionary containing:
                - experiment_name: Experiment name (top-level)
                - name: Experiment name (fallback, inside config)
                - dataset_path: Path to dataset
                - model_name: Model architecture name
                - strategy: Active learning strategy
                - Other training parameters
        
        Returns:
            Unique experiment ID (UUID)
        """
        # Generate unique experiment ID (used as database key)
        exp_id = str(uuid.uuid4())
        
        # Get experiment name - check top-level first, then inside nested config
        experiment_name = config.get('experiment_name')
        if not experiment_name:
            nested_config = config.get('config', {})
            experiment_name = nested_config.get('name', config.get('name', 'Unnamed Experiment'))
        
        # Sanitize experiment name for filesystem use
        sanitized_name = self._sanitize_folder_name(experiment_name)
        
        # Get unique folder name (handle collisions)
        folder_name = self._get_unique_folder_name(sanitized_name)
        
        # Create experiment folder structure using sanitized name
        exp_dir = self.experiments_dir / folder_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "queries").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)
        
        # Get other config values - check nested config first
        nested_config = config.get('config', {})
        dataset_path = nested_config.get('dataset_path') or config.get('dataset_path')
        model_name = nested_config.get('model_name') or config.get('model_name')
        strategy = nested_config.get('sampling_strategy') or nested_config.get('strategy') or config.get('strategy')
        
        # Store the folder name in config for later retrieval
        config['folder_name'] = folder_name
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiments (id, name, config, dataset_path, model_name, strategy)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            experiment_name,
            json.dumps(config),
            dataset_path,
            model_name,
            strategy
        ))
        
        conn.commit()
        conn.close()
        
        return exp_id
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments.
        
        Returns:
            List of experiment dictionaries with id, name, created_at, status, etc.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, created_at, status, dataset_path, model_name, strategy
            FROM experiments
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def _get_folder_name(self, exp_id: str) -> str:
        """Get the folder name for an experiment.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            Folder name for the experiment
        
        Raises:
            ValueError: If experiment not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT config FROM experiments WHERE id = ?", (exp_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            raise ValueError(f"Experiment {exp_id} not found")
        
        config = json.loads(row[0])
        
        # Check if folder_name is stored in config (new experiments)
        if 'folder_name' in config:
            return config['folder_name']
        
        # Fallback for old experiments that used UUID as folder name
        return exp_id
    
    def load_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Load experiment by ID.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            Experiment dictionary with all fields including parsed config
        
        Raises:
            ValueError: If experiment not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, created_at, config, status, dataset_path, model_name, strategy
            FROM experiments
            WHERE id = ?
        """, (exp_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            raise ValueError(f"Experiment {exp_id} not found")
        
        result = dict(row)
        result['config'] = json.loads(result['config'])
        return result
    
    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Get experiment by ID (alias for load_experiment).
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            Experiment dictionary with all fields including parsed config
        
        Raises:
            ValueError: If experiment not found
        """
        return self.load_experiment(exp_id)
    
    def delete_experiment(self, exp_id: str) -> bool:
        """Delete experiment and its folder.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            True if deleted successfully, False if not found
        """
        # Check if experiment exists and get folder name
        try:
            folder_name = self._get_folder_name(exp_id)
        except ValueError:
            return False
        
        # Delete from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM epoch_metrics WHERE experiment_id = ?", (exp_id,))
        cursor.execute("DELETE FROM cycle_results WHERE experiment_id = ?", (exp_id,))
        cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
        
        conn.commit()
        conn.close()
        
        # Delete experiment folder using folder_name
        exp_dir = self.experiments_dir / folder_name
        if exp_dir.exists():
            import shutil
            shutil.rmtree(exp_dir)
        
        return True

    def save_cycle_result(self, exp_id: str, cycle_data: Dict[str, Any]) -> None:
        """Save cycle results to database.
        
        Args:
            exp_id: Experiment ID
            cycle_data: Dictionary containing:
                - cycle: Cycle number
                - labeled_count: Number of labeled samples
                - unlabeled_count: Number of unlabeled samples
                - epochs_trained: Number of epochs trained
                - best_val_accuracy: Best validation accuracy
                - test_accuracy: Test accuracy
                - test_f1: Test F1 score (optional)
                - test_precision: Test precision (optional)
                - test_recall: Test recall (optional)
                - per_class_metrics: Per-class metrics dict (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert per_class_metrics to JSON if present
        per_class_json = None
        if 'per_class_metrics' in cycle_data and cycle_data['per_class_metrics'] is not None:
            per_class_json = json.dumps(cycle_data['per_class_metrics'])
        
        cursor.execute("""
            INSERT OR REPLACE INTO cycle_results 
            (experiment_id, cycle, labeled_count, unlabeled_count, epochs_trained,
             best_val_accuracy, test_accuracy, test_f1, test_precision, test_recall,
             per_class_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            cycle_data['cycle'],
            cycle_data.get('labeled_count'),
            cycle_data.get('unlabeled_count'),
            cycle_data.get('epochs_trained'),
            cycle_data.get('best_val_accuracy'),
            cycle_data.get('test_accuracy'),
            cycle_data.get('test_f1'),
            cycle_data.get('test_precision'),
            cycle_data.get('test_recall'),
            per_class_json
        ))
        
        conn.commit()
        conn.close()
    
    def get_cycle_results(self, exp_id: str, limit: Optional[int] = None, 
                         offset: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get cycle results for an experiment with pagination support.
        
        Args:
            exp_id: Experiment ID
            limit: Maximum number of results to return (optional)
            offset: Number of results to skip (optional)
        
        Returns:
            List of cycle result dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = """
            SELECT cycle, labeled_count, unlabeled_count, epochs_trained,
                   best_val_accuracy, test_accuracy, test_f1, test_precision,
                   test_recall, per_class_metrics, created_at
            FROM cycle_results
            WHERE experiment_id = ?
            ORDER BY cycle ASC
        """
        
        params = [exp_id]
        
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            # Parse per_class_metrics JSON if present
            if result['per_class_metrics']:
                result['per_class_metrics'] = json.loads(result['per_class_metrics'])
            results.append(result)
        
        return results
    
    def get_cycle_summaries(self, exp_id: str) -> List[Any]:
        """Get cycle summaries for an experiment.
        
        Returns CycleSummary objects for use in the Results page.
        
        Args:
            exp_id: Experiment ID
        
        Returns:
            List of CycleSummary objects
        """
        from model.schemas import CycleSummary
        from datetime import datetime
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT cycle, labeled_count, unlabeled_count, test_accuracy, test_f1, 
                   test_precision, test_recall, best_val_accuracy, epochs_trained,
                   per_class_metrics, created_at
            FROM cycle_results
            WHERE experiment_id = ?
            ORDER BY cycle ASC
        """, (exp_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        summaries = []
        for row in rows:
            # Parse timestamp
            timestamp = datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now()
            
            # Create CycleSummary with additional fields
            summary = CycleSummary(
                cycle=row['cycle'],
                labeled_count=row['labeled_count'],
                unlabeled_count=row['unlabeled_count'],
                test_accuracy=row['test_accuracy'],
                test_f1=row['test_f1'],
                timestamp=timestamp
            )
            
            # Add additional fields that aren't in CycleSummary dataclass
            summary.test_precision = row['test_precision']
            summary.test_recall = row['test_recall']
            summary.best_val_accuracy = row['best_val_accuracy']
            summary.epochs_trained = row['epochs_trained']
            summary.best_epoch = row['epochs_trained']  # Approximation
            summary.annotation_accuracy = None  # Not stored yet
            
            # Parse per_class_metrics if present
            if row['per_class_metrics']:
                summary.per_class_metrics = json.loads(row['per_class_metrics'])
            else:
                summary.per_class_metrics = None
            
            summaries.append(summary)
        
        return summaries

    def save_checkpoint(self, exp_id: str, cycle: int, state_dict: Dict[str, Any]) -> Path:
        """Save model checkpoint to file.
        
        Args:
            exp_id: Experiment ID
            cycle: Cycle number
            state_dict: PyTorch model state dictionary
        
        Returns:
            Path to saved checkpoint file
        """
        import torch
        
        folder_name = self._get_folder_name(exp_id)
        checkpoint_dir = self.experiments_dir / folder_name / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"cycle_{cycle}.pth"
        torch.save(state_dict, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, exp_id: str, cycle: int) -> Dict[str, Any]:
        """Load model checkpoint from file.
        
        Args:
            exp_id: Experiment ID
            cycle: Cycle number
        
        Returns:
            PyTorch model state dictionary
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        import torch
        
        folder_name = self._get_folder_name(exp_id)
        checkpoint_path = self.experiments_dir / folder_name / "checkpoints" / f"cycle_{cycle}.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path)
    
    def save_confusion_matrix(self, exp_id: str, cycle: int, cm: Any) -> Path:
        """Save confusion matrix to file.
        
        Args:
            exp_id: Experiment ID
            cycle: Cycle number
            cm: Confusion matrix as numpy array
        
        Returns:
            Path to saved confusion matrix file
        """
        import numpy as np
        
        folder_name = self._get_folder_name(exp_id)
        results_dir = self.experiments_dir / folder_name / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        cm_path = results_dir / f"confusion_matrix_{cycle}.npy"
        np.save(cm_path, cm)
        
        return cm_path
    
    def load_confusion_matrix(self, exp_id: str, cycle: int) -> Any:
        """Load confusion matrix from file.
        
        Args:
            exp_id: Experiment ID
            cycle: Cycle number
        
        Returns:
            Confusion matrix as numpy array
        
        Raises:
            FileNotFoundError: If confusion matrix file doesn't exist
        """
        import numpy as np
        
        folder_name = self._get_folder_name(exp_id)
        cm_path = self.experiments_dir / folder_name / "results" / f"confusion_matrix_{cycle}.npy"
        
        if not cm_path.exists():
            raise FileNotFoundError(f"Confusion matrix not found: {cm_path}")
        
        return np.load(cm_path)
