"""
ActiveLearningLoop - Orchestrates the Active Learning training cycles.

Responsibilities:
- Manage AL cycle execution
- Train model on labeled data each cycle
- Query samples from unlabeled pool
- Update data manager with new labels
- Track metrics and results
- Save cycle data for frontend visualization
"""

import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Orchestrates the Active Learning training loop.
    
    Process for each cycle:
    1. Get labeled data from manager
    2. Train model on labeled data
    3. Evaluate on test set
    4. Query uncertain samples from unlabeled pool using strategy
    5. Update data manager (move queried samples to labeled)
    6. Log metrics and results
    
    Repeat for N cycles
    """
    
    def __init__(self,
                 trainer,
                 data_manager,
                 strategy,
                 test_loader,
                 exp_dir: Path,
                 config):
        """
        Initialize ActiveLearningLoop.
        
        Args:
            trainer: Trainer instance (has train() and evaluate() methods)
            data_manager: ALDataManager instance (manages labeled/unlabeled pools)
            strategy: Strategy function for querying samples
                     Signature: strategy(model, unlabeled_loader, n_samples, device) -> query_indices
            test_loader: Fixed test set DataLoader (never changes across cycles)
            exp_dir: Experiment directory for saving results
            config: Configuration object with active_learning settings
                    Should have: num_cycles, batch_size_al, etc.
        """
        self.trainer = trainer
        self.data_manager = data_manager
        self.strategy = strategy
        self.test_loader = test_loader
        self.exp_dir = Path(exp_dir)
        self.config = config
        
        # Track results across cycles
        self.cycle_results = []
        
        logger.info("ActiveLearningLoop initialized")
        logger.info(f"  Max cycles: {config.active_learning.num_cycles}")
        logger.info(f"  Batch size for querying: {config.active_learning.batch_size_al}")
        logger.info(f"  Strategy: {config.active_learning.sampling_strategy}")
    
    def run_cycle(self, cycle_num: int):
        """
        Execute one Active Learning cycle.
        
        Steps:
        1. Train on current labeled set
        2. Evaluate on test set
        3. Query new samples from unlabeled pool
        4. Update data manager
        
        Args:
            cycle_num: Current cycle number (1-indexed)
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"AL CYCLE {cycle_num}/{self.config.active_learning.num_cycles}")
        logger.info(f"{'='*70}")
        
        # Step 1: Get labeled data and train
        logger.info("Step 1: Training on labeled data...")
        
        train_loader = self.data_manager.get_labeled_loader(
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        pool_info = self.data_manager.get_pool_info()
        logger.info(f"  Pool sizes - Labeled: {pool_info['labeled']}, "
                   f"Unlabeled: {pool_info['unlabeled']}")
        logger.info(f"  Training batches: {len(train_loader)}")
        
        # Train (for AL, we typically don't use validation)
        self.trainer.train(train_loader, val_loader=None)
        
        # Step 2: Evaluate on test set
        logger.info("Step 2: Evaluating on test set...")
        
        metrics = self.trainer.evaluate(self.test_loader)
        
        # Store cycle results
        cycle_result = {
            "cycle": cycle_num,
            "labeled_pool_size": pool_info["labeled"],
            "unlabeled_pool_size": pool_info["unlabeled"],
            "test_accuracy": metrics.get("test_accuracy", 0),
            "test_f1": metrics.get("test_f1", 0),
            "test_precision": metrics.get("test_precision", 0),
            "test_recall": metrics.get("test_recall", 0),
        }
        
        logger.info(f"  Accuracy: {metrics.get('test_accuracy', 0):.4f}")
        logger.info(f"  F1 Score: {metrics.get('test_f1', 0):.4f}")
        
        # Step 3 & 4: Query and update pools (if not last cycle)
        if cycle_num < self.config.active_learning.num_cycles:
            logger.info("Step 3: Querying uncertain samples...")
            
            # Get unlabeled data
            unlabeled_loader = self.data_manager.get_unlabeled_loader(
                batch_size=128,
                shuffle=False,
                num_workers=4
            )
            
            logger.info(f"  Unlabeled batches: {len(unlabeled_loader)}")
            logger.info(f"  Requesting {self.config.active_learning.batch_size_al} samples")
            
            # Query using strategy
            try:
                query_indices = self.strategy(
                    self.trainer.model,
                    unlabeled_loader,
                    self.config.active_learning.batch_size_al,
                    self.trainer.device
                )
            except Exception as e:
                logger.error(f"Error in strategy: {e}")
                raise
            
            logger.info(f"  Queried indices count: {len(query_indices)}")
            
            # Step 4: Update data manager
            logger.info("Step 4: Updating labeled pool...")
            
            self.data_manager.update_labeled_pool(query_indices)
            
            # Save queried indices for reference
            queried_file = self.exp_dir / f"cycle_{cycle_num}_queried_indices.json"
            with open(queried_file, "w") as f:
                json.dump(query_indices.tolist() if hasattr(query_indices, 'tolist') else query_indices, f)
            logger.info(f"  Queried indices saved to {queried_file}")
        
        else:
            logger.info("Step 3: Final cycle - no querying")
        
        # Save cycle result
        self.cycle_results.append(cycle_result)
        self._save_cycle_results()
        
        logger.info(f"Cycle {cycle_num} complete!\n")
    
    def run_all_cycles(self) -> List[Dict]:
        """
        Execute all Active Learning cycles.
        
        Returns:
            List of cycle results (each result is a dict with metrics)
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING ACTIVE LEARNING TRAINING")
        logger.info("="*70)
        
        # Run each cycle
        for cycle in range(1, self.config.active_learning.num_cycles + 1):
            self.run_cycle(cycle)
        
        logger.info("\n" + "="*70)
        logger.info("ACTIVE LEARNING COMPLETE")
        logger.info("="*70)
        
        # Print summary
        self._log_summary()
        
        # Save final state
        self.data_manager.save_state()
        logger.info("Pool state saved")
        
        return self.cycle_results
    
    def _save_cycle_results(self):
        """Save all cycle results to JSON."""
        results_file = self.exp_dir / "al_cycle_results.json"
        with open(results_file, "w") as f:
            json.dump(self.cycle_results, f, indent=2)
        logger.info(f"Cycle results saved to {results_file}")
    
    def _log_summary(self):
        """Log summary of all completed cycles."""
        logger.info("AL CYCLE SUMMARY")
        logger.info("-" * 70)
        
        logger.info(f"{'Cycle':<8} {'Labeled':<10} {'Unlabeled':<12} {'Accuracy':<12} {'F1':<10}")
        logger.info("-" * 70)
        
        for result in self.cycle_results:
            logger.info(
                f"{result['cycle']:<8} "
                f"{result['labeled_pool_size']:<10} "
                f"{result['unlabeled_pool_size']:<12} "
                f"{result['test_accuracy']:<12.4f} "
                f"{result['test_f1']:<10.4f}"
            )
        
        logger.info("-" * 70)
        
        # Best accuracy
        best = max(self.cycle_results, key=lambda x: x["test_accuracy"])
        logger.info(f"\nBest accuracy: {best['test_accuracy']:.4f} at cycle {best['cycle']}")
        
        # Improvement
        first_acc = self.cycle_results[0]["test_accuracy"]
        last_acc = self.cycle_results[-1]["test_accuracy"]
        improvement = last_acc - first_acc
        logger.info(f"Improvement: {improvement:+.4f} ({improvement/first_acc*100:+.1f}%)")
    
    def get_results(self) -> List[Dict]:
        """
        Get cycle results.
        
        Returns:
            List of cycle results with metrics
        """
        return self.cycle_results
    
    def get_best_cycle(self) -> Dict:
        """
        Get best performing cycle.
        
        Returns:
            Cycle result with highest test accuracy
        """
        if not self.cycle_results:
            return None
        return max(self.cycle_results, key=lambda x: x["test_accuracy"])