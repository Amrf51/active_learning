"""
DatabaseManager - SQLite Persistence Layer.

This module provides persistent storage for experiment data using SQLite.
It replaces the JSON file approach with proper database schema.

Key Design Decisions:
1. SQLite for reliability and query capabilities
2. Paginated access for large datasets
3. JSON blobs for complex nested data (config, per_class_metrics)
4. Indexes for common query patterns

Tables:
- experiments: Experiment metadata and config
- cycles: Cycle summaries
- epochs: Per-epoch metrics
- pool_items: Labeled/unlabeled pool tracking (optional, for large datasets)

Usage:
    db = DatabaseManager("experiments.db")
    
    # Create experiment
    db.insert_experiment(experiment_id, name, config_dict)
    
    # Update progress
    db.update_experiment_phase(experiment_id, "TRAINING")
    db.insert_epoch_metrics(experiment_id, cycle, epoch_metrics)
    
    # Query history
    metrics = db.get_epoch_metrics_paginated(experiment_id, cycle, page=1, limit=50)
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import logging

from .schemas import (
    ExperimentPhase,
    ExperimentConfig,
    EpochMetrics,
    CycleSummary
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for experiment persistence.
    
    Thread Safety:
        SQLite connections are not thread-safe. This class creates a new
        connection for each operation using context managers. For the
        listener thread, consider using a separate DatabaseManager instance.
    
    Attributes:
        db_path: Path to SQLite database file
    """
    
    def __init__(self, db_path: str | Path):
        """
        Initialize DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file.
                     Use ":memory:" for in-memory database (testing).
        """
        self.db_path = str(db_path)
        self._is_memory = (self.db_path == ":memory:")
        
        # For in-memory databases, keep a persistent connection
        # because each new connection creates a fresh database
        if self._is_memory:
            self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._persistent_conn.row_factory = sqlite3.Row
        else:
            self._persistent_conn = None
            
        self._init_schema()
        logger.info(f"DatabaseManager initialized: {self.db_path}")
    
    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection with row factory set to dict.
        """
        if self._is_memory:
            # Use persistent connection for in-memory database
            yield self._persistent_conn
            self._persistent_conn.commit()
        else:
            # Create new connection for file-based database
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    
    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # ═══════════════════════════════════════════════════════════
            # Experiments Table
            # ═══════════════════════════════════════════════════════════
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    experiment_name TEXT NOT NULL,
                    config_json TEXT,
                    phase TEXT DEFAULT 'IDLE',
                    current_cycle INTEGER DEFAULT 0,
                    labeled_count INTEGER DEFAULT 0,
                    unlabeled_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # ═══════════════════════════════════════════════════════════
            # Cycles Table
            # ═══════════════════════════════════════════════════════════
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    labeled_count INTEGER,
                    unlabeled_count INTEGER,
                    epochs_trained INTEGER,
                    best_val_accuracy REAL,
                    best_epoch INTEGER,
                    test_accuracy REAL,
                    test_f1 REAL,
                    test_precision REAL,
                    test_recall REAL,
                    per_class_metrics_json TEXT,
                    annotation_accuracy REAL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    UNIQUE(experiment_id, cycle)
                )
            """)
            
            # ═══════════════════════════════════════════════════════════
            # Epochs Table
            # ═══════════════════════════════════════════════════════════
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS epochs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    train_accuracy REAL,
                    val_loss REAL,
                    val_accuracy REAL,
                    learning_rate REAL,
                    timestamp TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    UNIQUE(experiment_id, cycle, epoch)
                )
            """)
            
            # ═══════════════════════════════════════════════════════════
            # Indexes for Performance
            # ═══════════════════════════════════════════════════════════
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cycles_experiment 
                ON cycles(experiment_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_epochs_experiment_cycle 
                ON epochs(experiment_id, cycle)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiments_active 
                ON experiments(is_active)
            """)
            
            logger.debug("Database schema initialized")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Experiment CRUD
    # ═══════════════════════════════════════════════════════════════════════
    
    def insert_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        config: Dict[str, Any],
        set_active: bool = True
    ) -> bool:  
        """
        Insert a new experiment.
        
        Args:
            experiment_id: Unique identifier
            experiment_name: Human-readable name
            config: Configuration dictionary
            set_active: Whether to set this as the active experiment
        """
        now = datetime.now().isoformat()
        config_json = json.dumps(config)
        
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing active if setting new active
            if set_active:
                cursor.execute("UPDATE experiments SET is_active = 0")
            
            cursor.execute("""
                INSERT INTO experiments 
                (experiment_id, experiment_name, config_json, phase, is_active, created_at, updated_at)
                VALUES (?, ?, ?, 'IDLE', ?, ?, ?)
            """, (experiment_id, experiment_name, config_json, int(set_active), now, now))
            
        logger.info(f"Inserted experiment: {experiment_id}")
        return True  # Added return statement
    
    def update_experiment_phase(
        self,
        experiment_id: str,
        phase: str,
        error_message: Optional[str] = None
    ) -> None:
        """
        Update experiment phase.
        
        Args:
            experiment_id: Experiment to update
            phase: New phase value
            error_message: Error details if phase is ERROR
        """
        now = datetime.now().isoformat()
        
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET phase = ?, error_message = ?, updated_at = ?
                WHERE experiment_id = ?
            """, (phase, error_message, now, experiment_id))
            
        logger.debug(f"Updated experiment {experiment_id} phase to {phase}")
    
    def update_experiment_progress(
        self,
        experiment_id: str,
        current_cycle: int,
        labeled_count: int,
        unlabeled_count: int
    ) -> None:
        """
        Update experiment progress counters.
        
        Args:
            experiment_id: Experiment to update
            current_cycle: Current cycle number
            labeled_count: Labeled pool size
            unlabeled_count: Unlabeled pool size
        """
        now = datetime.now().isoformat()
        
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE experiments 
                SET current_cycle = ?, labeled_count = ?, unlabeled_count = ?, updated_at = ?
                WHERE experiment_id = ?
            """, (current_cycle, labeled_count, unlabeled_count, now, experiment_id))
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment to retrieve
            
        Returns:
            Experiment data dict or None if not found
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments WHERE experiment_id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            result = dict(row)
            if result.get("config_json"):
                result["config"] = json.loads(result["config_json"])
            if result.get("created_at"):
                result["created_at"] = datetime.fromisoformat(result["created_at"])
            
            return result
    
    def get_active_experiment(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active experiment.
        
        Returns:
            Active experiment data dict or None
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM experiments WHERE is_active = 1
            """)
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            result = dict(row)
            if result.get("config_json"):
                result["config"] = json.loads(result["config_json"])
            if result.get("created_at"):
                result["created_at"] = datetime.fromisoformat(result["created_at"])
            
            return result
    
    def set_active_experiment(self, experiment_id: str) -> None:
        """
        Set an experiment as active.
        
        Args:
            experiment_id: Experiment to activate
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE experiments SET is_active = 0")
            cursor.execute("""
                UPDATE experiments SET is_active = 1 WHERE experiment_id = ?
            """, (experiment_id,))
            
        logger.info(f"Set active experiment: {experiment_id}")
    
    def list_experiments(
        self,
        page: int = 1,
        limit: int = 20
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        List experiments with pagination.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            
        Returns:
            Tuple of (experiments list, total count)
        """
        offset = (page - 1) * limit
        
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM experiments")
            total = cursor.fetchone()[0]
            
            # Get page
            cursor.execute("""
                SELECT experiment_id, experiment_name, phase, current_cycle, 
                       is_active, created_at, updated_at
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            experiments = []
            for row in cursor.fetchall():
                exp = dict(row)
                if exp.get("created_at"):
                    exp["created_at"] = datetime.fromisoformat(exp["created_at"])
                experiments.append(exp)
            
            return experiments, total
    
    # ═══════════════════════════════════════════════════════════════════════
    # Epoch Metrics CRUD
    # ═══════════════════════════════════════════════════════════════════════
    
    def insert_epoch_metrics(
        self,
        experiment_id: str,
        cycle: int,
        metrics: EpochMetrics
    ) -> None:
        """
        Insert epoch metrics.
        
        Args:
            experiment_id: Experiment ID
            cycle: Cycle number
            metrics: Epoch metrics to insert
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO epochs
                (experiment_id, cycle, epoch, train_loss, train_accuracy, 
                 val_loss, val_accuracy, learning_rate, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, cycle, metrics.epoch,
                metrics.train_loss, metrics.train_accuracy,
                metrics.val_loss, metrics.val_accuracy,
                metrics.learning_rate,
                metrics.timestamp.isoformat() if metrics.timestamp else None
            ))
    
    def get_epoch_metrics(
        self,
        experiment_id: str,
        cycle: int
    ) -> List[EpochMetrics]:
        """
        Get all epoch metrics for a cycle.
        
        Args:
            experiment_id: Experiment ID
            cycle: Cycle number
            
        Returns:
            List of EpochMetrics
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM epochs
                WHERE experiment_id = ? AND cycle = ?
                ORDER BY epoch
            """, (experiment_id, cycle))
            
            return [
                EpochMetrics(
                    epoch=row["epoch"],
                    train_loss=row["train_loss"],
                    train_accuracy=row["train_accuracy"],
                    val_loss=row["val_loss"],
                    val_accuracy=row["val_accuracy"],
                    learning_rate=row["learning_rate"],
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None
                )
                for row in cursor.fetchall()
            ]
    
    def get_epoch_metrics_paginated(
        self,
        experiment_id: str,
        cycle: int,
        page: int = 1,
        limit: int = 50
    ) -> Tuple[List[EpochMetrics], int]:
        """
        Get epoch metrics with pagination.
        
        Args:
            experiment_id: Experiment ID
            cycle: Cycle number
            page: Page number (1-indexed)
            limit: Items per page
            
        Returns:
            Tuple of (metrics list, total count)
        """
        offset = (page - 1) * limit
        
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("""
                SELECT COUNT(*) FROM epochs
                WHERE experiment_id = ? AND cycle = ?
            """, (experiment_id, cycle))
            total = cursor.fetchone()[0]
            
            # Get page
            cursor.execute("""
                SELECT * FROM epochs
                WHERE experiment_id = ? AND cycle = ?
                ORDER BY epoch
                LIMIT ? OFFSET ?
            """, (experiment_id, cycle, limit, offset))
            
            metrics = [
                EpochMetrics(
                    epoch=row["epoch"],
                    train_loss=row["train_loss"],
                    train_accuracy=row["train_accuracy"],
                    val_loss=row["val_loss"],
                    val_accuracy=row["val_accuracy"],
                    learning_rate=row["learning_rate"],
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None
                )
                for row in cursor.fetchall()
            ]
            
            return metrics, total
    
    # ═══════════════════════════════════════════════════════════════════════
    # Cycle Summary CRUD
    # ═══════════════════════════════════════════════════════════════════════
    
    def insert_cycle_summary(
        self,
        experiment_id: str,
        summary: CycleSummary
    ) -> None:
        """
        Insert cycle summary.
        
        Args:
            experiment_id: Experiment ID
            summary: Cycle summary to insert
        """
        per_class_json = json.dumps(summary.per_class_metrics) if summary.per_class_metrics else None
        
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cycles
                (experiment_id, cycle, labeled_count, unlabeled_count, epochs_trained,
                 best_val_accuracy, best_epoch, test_accuracy, test_f1, 
                 test_precision, test_recall, per_class_metrics_json,
                 annotation_accuracy, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, summary.cycle,
                summary.labeled_count, summary.unlabeled_count,
                summary.epochs_trained, summary.best_val_accuracy,
                summary.best_epoch, summary.test_accuracy, summary.test_f1,
                summary.test_precision, summary.test_recall,
                per_class_json, summary.annotation_accuracy,
                summary.started_at.isoformat() if summary.started_at else None,
                summary.completed_at.isoformat() if summary.completed_at else None
            ))
            
        logger.info(f"Inserted cycle summary: {experiment_id} cycle {summary.cycle}")
    
    def get_cycle_summaries(
        self,
        experiment_id: str
    ) -> List[CycleSummary]:
        """
        Get all cycle summaries for an experiment.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            List of CycleSummary
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM cycles
                WHERE experiment_id = ?
                ORDER BY cycle
            """, (experiment_id,))
            
            summaries = []
            for row in cursor.fetchall():
                per_class = None
                if row["per_class_metrics_json"]:
                    per_class = json.loads(row["per_class_metrics_json"])
                
                summaries.append(CycleSummary(
                    cycle=row["cycle"],
                    labeled_count=row["labeled_count"],
                    unlabeled_count=row["unlabeled_count"],
                    epochs_trained=row["epochs_trained"],
                    best_val_accuracy=row["best_val_accuracy"],
                    best_epoch=row["best_epoch"],
                    test_accuracy=row["test_accuracy"],
                    test_f1=row["test_f1"],
                    test_precision=row["test_precision"] or 0.0,
                    test_recall=row["test_recall"] or 0.0,
                    per_class_metrics=per_class,
                    annotation_accuracy=row["annotation_accuracy"],
                    started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                    completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
                ))
            
            return summaries
    
    def get_cycle_summary(
        self,
        experiment_id: str,
        cycle: int
    ) -> Optional[CycleSummary]:
        """
        Get specific cycle summary.
        
        Args:
            experiment_id: Experiment ID
            cycle: Cycle number
            
        Returns:
            CycleSummary or None
        """
        summaries = self.get_cycle_summaries(experiment_id)
        for s in summaries:
            if s.cycle == cycle:
                return s
        return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ═══════════════════════════════════════════════════════════════════════
    
    def delete_experiment(self, experiment_id: str) -> None:
        """
        Delete experiment and all related data.
        
        Args:
            experiment_id: Experiment to delete
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM epochs WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM cycles WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
            
        logger.info(f"Deleted experiment: {experiment_id}")
    
    def clear_all(self) -> None:
        """
        Clear all data from database.
        
        Use with caution - for testing only.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM epochs")
            cursor.execute("DELETE FROM cycles")
            cursor.execute("DELETE FROM experiments")
            
        logger.warning("Cleared all database data")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get database statistics.
        
        Returns:
            Dict with counts of experiments, cycles, epochs
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM experiments")
            experiments = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cycles")
            cycles = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM epochs")
            epochs = cursor.fetchone()[0]
            
            return {
                "experiments": experiments,
                "cycles": cycles,
                "epochs": epochs
            }
