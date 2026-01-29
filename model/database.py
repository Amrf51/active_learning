"""
DatabaseManager: SQLite persistence layer for experiment data.

This module provides the DatabaseManager class that handles all SQLite operations
for persistent storage of experiment data, metrics, and historical information.
It's designed to work alongside WorldState for the hybrid storage approach.
"""

import sqlite3
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite database manager for experiment persistence.
    
    This class handles all database operations for the Active Learning Dashboard,
    providing persistent storage for experiment data, metrics, and historical
    information. It's designed to complement WorldState for fast access.
    
    Database Schema:
    - experiments: Main experiment records
    - cycle_summaries: Summary data for each AL cycle
    - epoch_metrics: Detailed metrics for each training epoch
    - pool_items: Dataset pool items (labeled/unlabeled)
    - queried_images: Images queried for annotation in each cycle
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._init_schema()
        
        logger.info(f"DatabaseManager initialized with database: {self.db_path}")
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        
        Ensures proper connection handling and automatic cleanup.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def _init_schema(self) -> None:
        """
        Initialize database schema with all required tables and indexes.
        
        Creates tables for experiments, cycle summaries, epoch metrics,
        pool items, and queried images with appropriate indexes for performance.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Experiments table - main experiment records
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT 'IDLE',
                    current_cycle INTEGER DEFAULT 0,
                    total_cycles INTEGER DEFAULT 0,
                    current_epoch INTEGER DEFAULT 0,
                    epochs_per_cycle INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # Cycle summaries table - summary for each AL cycle
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cycle_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    labeled_count INTEGER DEFAULT 0,
                    unlabeled_count INTEGER DEFAULT 0,
                    best_val_acc REAL,
                    test_acc REAL,
                    test_f1 REAL,
                    cycle_duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                    UNIQUE (experiment_id, cycle)
                )
            """)
            
            # Epoch metrics table - detailed metrics for each epoch
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS epoch_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL,
                    val_loss REAL,
                    train_acc REAL,
                    val_acc REAL,
                    learning_rate REAL,
                    epoch_duration REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                    UNIQUE (experiment_id, cycle, epoch)
                )
            """)
            
            # Pool items table - dataset items with their pool status
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pool_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    pool_type TEXT NOT NULL, -- 'labeled', 'unlabeled', 'test'
                    class_idx INTEGER,
                    true_label TEXT,
                    predicted_label TEXT,
                    confidence REAL,
                    uncertainty REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Queried images table - images queried for annotation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queried_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    predictions_json TEXT, -- JSON array of class predictions
                    annotations_json TEXT, -- JSON of user annotations
                    uncertainty_score REAL,
                    query_strategy TEXT,
                    annotated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (id)
                )
            """)
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_experiments_active ON experiments (is_active)",
                "CREATE INDEX IF NOT EXISTS idx_cycle_summaries_exp ON cycle_summaries (experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_epoch_metrics_exp_cycle ON epoch_metrics (experiment_id, cycle)",
                "CREATE INDEX IF NOT EXISTS idx_pool_items_exp_type ON pool_items (experiment_id, pool_type)",
                "CREATE INDEX IF NOT EXISTS idx_queried_images_exp_cycle ON queried_images (experiment_id, cycle)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database schema initialized successfully")
    
    # Experiment CRUD operations
    
    def insert_experiment(self, experiment_id: str, name: str, config: Dict[str, Any]) -> bool:
        """
        Insert a new experiment record.
        
        Args:
            experiment_id: Unique experiment identifier
            name: Experiment name
            config: Experiment configuration dictionary
            
        Returns:
            True if insertion successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Deactivate any existing active experiments
                cursor.execute("UPDATE experiments SET is_active = 0")
                
                # Insert new experiment
                cursor.execute("""
                    INSERT INTO experiments (
                        id, name, config_json, phase, total_cycles, epochs_per_cycle, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (
                    experiment_id,
                    name,
                    json.dumps(config),
                    'INITIALIZING',
                    config.get('total_cycles', 10),
                    config.get('epochs_per_cycle', 5)
                ))
                
                conn.commit()
                logger.info(f"Inserted experiment: {experiment_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert experiment {experiment_id}: {e}")
            return False
    
    def update_experiment_phase(self, experiment_id: str, phase: str, 
                               current_cycle: Optional[int] = None,
                               current_epoch: Optional[int] = None,
                               error_message: Optional[str] = None) -> bool:
        """
        Update experiment phase and progress.
        
        Args:
            experiment_id: Experiment identifier
            phase: New phase
            current_cycle: Current cycle number (optional)
            current_epoch: Current epoch number (optional)
            error_message: Error message if phase is ERROR (optional)
            
        Returns:
            True if update successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                updates = ["phase = ?", "updated_at = CURRENT_TIMESTAMP"]
                params = [phase]
                
                if current_cycle is not None:
                    updates.append("current_cycle = ?")
                    params.append(current_cycle)
                
                if current_epoch is not None:
                    updates.append("current_epoch = ?")
                    params.append(current_epoch)
                
                if error_message is not None:
                    updates.append("error_message = ?")
                    params.append(error_message)
                
                params.append(experiment_id)
                
                cursor.execute(f"""
                    UPDATE experiments 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update experiment {experiment_id}: {e}")
            return False
    
    def get_active_experiment(self) -> Optional[Dict[str, Any]]:
        """
        Get the currently active experiment.
        
        Returns:
            Dictionary with experiment data or None if no active experiment
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM experiments 
                    WHERE is_active = 1 
                    ORDER BY updated_at DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    experiment = dict(row)
                    # Parse config JSON
                    experiment['config'] = json.loads(experiment['config_json'])
                    return experiment
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get active experiment: {e}")
            return None
    
    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment by ID.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Dictionary with experiment data or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
                
                row = cursor.fetchone()
                if row:
                    experiment = dict(row)
                    experiment['config'] = json.loads(experiment['config_json'])
                    return experiment
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get experiment {experiment_id}: {e}")
            return None
    
    # Epoch metrics operations
    
    def insert_epoch_metrics(self, experiment_id: str, cycle: int, epoch: int,
                           metrics: Dict[str, Any]) -> bool:
        """
        Insert epoch training metrics.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Cycle number
            epoch: Epoch number
            metrics: Dictionary with training metrics
            
        Returns:
            True if insertion successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO epoch_metrics (
                        experiment_id, cycle, epoch, train_loss, val_loss,
                        train_acc, val_acc, learning_rate, epoch_duration
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, cycle, epoch,
                    metrics.get('train_loss'),
                    metrics.get('val_loss'),
                    metrics.get('train_acc'),
                    metrics.get('val_acc'),
                    metrics.get('learning_rate'),
                    metrics.get('epoch_duration')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert epoch metrics: {e}")
            return False
    
    def get_epoch_metrics_paginated(self, experiment_id: str, cycle: Optional[int] = None,
                                  page: int = 1, limit: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated epoch metrics.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Specific cycle (optional, gets all cycles if None)
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (metrics_list, total_count)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with optional cycle filter
                where_clause = "WHERE experiment_id = ?"
                params = [experiment_id]
                
                if cycle is not None:
                    where_clause += " AND cycle = ?"
                    params.append(cycle)
                
                # Get total count
                cursor.execute(f"SELECT COUNT(*) FROM epoch_metrics {where_clause}", params)
                total_count = cursor.fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * limit
                cursor.execute(f"""
                    SELECT * FROM epoch_metrics {where_clause}
                    ORDER BY cycle, epoch
                    LIMIT ? OFFSET ?
                """, params + [limit, offset])
                
                metrics = [dict(row) for row in cursor.fetchall()]
                return metrics, total_count
                
        except Exception as e:
            logger.error(f"Failed to get epoch metrics: {e}")
            return [], 0
    
    # Cycle summary operations
    
    def insert_cycle_summary(self, experiment_id: str, cycle: int, 
                           summary: Dict[str, Any]) -> bool:
        """
        Insert cycle summary data.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Cycle number
            summary: Dictionary with cycle summary data
            
        Returns:
            True if insertion successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO cycle_summaries (
                        experiment_id, cycle, labeled_count, unlabeled_count,
                        best_val_acc, test_acc, test_f1, cycle_duration
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, cycle,
                    summary.get('labeled_count'),
                    summary.get('unlabeled_count'),
                    summary.get('best_val_acc'),
                    summary.get('test_acc'),
                    summary.get('test_f1'),
                    summary.get('cycle_duration')
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert cycle summary: {e}")
            return False
    
    def get_cycle_summaries(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Get all cycle summaries for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of cycle summary dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM cycle_summaries 
                    WHERE experiment_id = ? 
                    ORDER BY cycle
                """, (experiment_id,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get cycle summaries: {e}")
            return []
    
    # Pool items operations
    
    def insert_pool_items(self, experiment_id: str, items: List[Dict[str, Any]]) -> bool:
        """
        Insert multiple pool items.
        
        Args:
            experiment_id: Experiment identifier
            items: List of pool item dictionaries
            
        Returns:
            True if insertion successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clear existing items for this experiment
                cursor.execute("DELETE FROM pool_items WHERE experiment_id = ?", (experiment_id,))
                
                # Insert new items
                for item in items:
                    cursor.execute("""
                        INSERT INTO pool_items (
                            experiment_id, image_path, pool_type, class_idx,
                            true_label, predicted_label, confidence, uncertainty
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        experiment_id,
                        item.get('image_path'),
                        item.get('pool_type'),
                        item.get('class_idx'),
                        item.get('true_label'),
                        item.get('predicted_label'),
                        item.get('confidence'),
                        item.get('uncertainty')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert pool items: {e}")
            return False
    
    def get_pool_items_paginated(self, experiment_id: str, pool_type: Optional[str] = None,
                               page: int = 1, limit: int = 50) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated pool items.
        
        Args:
            experiment_id: Experiment identifier
            pool_type: Filter by pool type ('labeled', 'unlabeled', 'test')
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (items_list, total_count)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with optional pool_type filter
                where_clause = "WHERE experiment_id = ?"
                params = [experiment_id]
                
                if pool_type:
                    where_clause += " AND pool_type = ?"
                    params.append(pool_type)
                
                # Get total count
                cursor.execute(f"SELECT COUNT(*) FROM pool_items {where_clause}", params)
                total_count = cursor.fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * limit
                cursor.execute(f"""
                    SELECT * FROM pool_items {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, params + [limit, offset])
                
                items = [dict(row) for row in cursor.fetchall()]
                return items, total_count
                
        except Exception as e:
            logger.error(f"Failed to get pool items: {e}")
            return [], 0
    
    def get_pool_count(self, experiment_id: str, pool_type: Optional[str] = None) -> int:
        """
        Get count of pool items.
        
        Args:
            experiment_id: Experiment identifier
            pool_type: Filter by pool type (optional)
            
        Returns:
            Count of items
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if pool_type:
                    cursor.execute("""
                        SELECT COUNT(*) FROM pool_items 
                        WHERE experiment_id = ? AND pool_type = ?
                    """, (experiment_id, pool_type))
                else:
                    cursor.execute("""
                        SELECT COUNT(*) FROM pool_items 
                        WHERE experiment_id = ?
                    """, (experiment_id,))
                
                return cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Failed to get pool count: {e}")
            return 0
    
    # Queried images operations
    
    def insert_queried_images(self, experiment_id: str, cycle: int, 
                            images: List[Dict[str, Any]]) -> bool:
        """
        Insert queried images for a cycle.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Cycle number
            images: List of queried image dictionaries
            
        Returns:
            True if insertion successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                for image in images:
                    cursor.execute("""
                        INSERT INTO queried_images (
                            experiment_id, cycle, image_path, predictions_json,
                            uncertainty_score, query_strategy
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        experiment_id, cycle,
                        image.get('image_path'),
                        json.dumps(image.get('predictions', [])),
                        image.get('uncertainty_score'),
                        image.get('query_strategy')
                    ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to insert queried images: {e}")
            return False
    
    def update_queried_image_annotations(self, experiment_id: str, cycle: int,
                                       image_path: str, annotations: Dict[str, Any]) -> bool:
        """
        Update annotations for a queried image.
        
        Args:
            experiment_id: Experiment identifier
            cycle: Cycle number
            image_path: Path to the image
            annotations: Annotation data
            
        Returns:
            True if update successful
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE queried_images 
                    SET annotations_json = ?, annotated_at = CURRENT_TIMESTAMP
                    WHERE experiment_id = ? AND cycle = ? AND image_path = ?
                """, (
                    json.dumps(annotations),
                    experiment_id, cycle, image_path
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update queried image annotations: {e}")
            return False
    
    # Utility methods
    
    def get_experiments_paginated(self, page: int = 1, limit: int = 20) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get paginated list of all experiments.
        
        Args:
            page: Page number (1-based)
            limit: Items per page
            
        Returns:
            Tuple of (experiments_list, total_count)
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get total count
                cursor.execute("SELECT COUNT(*) FROM experiments")
                total_count = cursor.fetchone()[0]
                
                # Get paginated results
                offset = (page - 1) * limit
                cursor.execute("""
                    SELECT id, name, phase, current_cycle, total_cycles, 
                           created_at, updated_at, is_active
                    FROM experiments 
                    ORDER BY updated_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                experiments = [dict(row) for row in cursor.fetchall()]
                return experiments, total_count
                
        except Exception as e:
            logger.error(f"Failed to get experiments: {e}")
            return [], 0
    
    def cleanup_old_experiments(self, keep_days: int = 30) -> int:
        """
        Clean up old experiment data.
        
        Args:
            keep_days: Number of days to keep data
            
        Returns:
            Number of experiments cleaned up
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get experiments older than keep_days
                cursor.execute("""
                    SELECT id FROM experiments 
                    WHERE is_active = 0 
                    AND datetime(updated_at) < datetime('now', '-{} days')
                """.format(keep_days))
                
                old_experiments = [row[0] for row in cursor.fetchall()]
                
                if not old_experiments:
                    return 0
                
                # Delete related data
                placeholders = ','.join(['?'] * len(old_experiments))
                
                tables = ['queried_images', 'pool_items', 'epoch_metrics', 
                         'cycle_summaries', 'experiments']
                
                for table in tables:
                    cursor.execute(f"""
                        DELETE FROM {table} 
                        WHERE experiment_id IN ({placeholders})
                    """, old_experiments)
                
                conn.commit()
                logger.info(f"Cleaned up {len(old_experiments)} old experiments")
                return len(old_experiments)
                
        except Exception as e:
            logger.error(f"Failed to cleanup old experiments: {e}")
            return 0