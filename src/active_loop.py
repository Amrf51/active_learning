"""
ActiveLearningLoop - Orchestrates Active Learning training cycles.

This is the "boss" that manages the AL workflow:
1. Commands Trainer to train on labeled data
2. Commands Trainer to evaluate on test set
3. Applies sampling strategy to query new samples
4. Updates ALDataManager pools
5. Tracks metrics across cycles

The Trainer is "dumb" - it only trains what it's given.
All AL logic lives here in the orchestrator.
"""

import json
from pathlib import Path
from typing import Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class ActiveLearningLoop:
    """
    Orchestrates the Active Learning training loop.
    
    Cycle process:
    1. Reset model weights (from-scratch training)
    2. Get labeled DataLoader from manager
    3. Train model on labeled data (with validation for early stopping)
    4. Evaluate on test set
    5. Query uncertain samples from unlabeled pool
    6. Update manager (move queried samples to labeled)
    7. Save cycle checkpoint and metrics
    
    Repeat for N cycles.
    """
    
    def __init__(
        self,
        trainer,
        data_manager,
        strategy: Callable,
        val_loader,
        test_loader,
        exp_dir: Path,
        config
    ):
        """
        Initialize ActiveLearningLoop.
        
        Args:
            trainer: Trainer instance (has train(), evaluate(), reset_model_weights())
            data_manager: ALDataManager instance (manages labeled/unlabeled pools)
            strategy: Strategy function with signature:
                      (model, unlabeled_loader, n_samples, device) -> query_indices
            val_loader: Fixed validation DataLoader (for early stopping during training)
            test_loader: Fixed test DataLoader (for evaluation after each cycle)
            exp_dir: Experiment directory for saving results
            config: Config object with active_learning settings
        """
        self.trainer = trainer
        self.data_manager = data_manager
        self.strategy = strategy
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.exp_dir = Path(exp_dir)
        self.config = config
        
        self.cycle_results = []
        
        al_config = config.active_learning
        logger.info("ActiveLearningLoop initialized:")
        logger.info(f"  Cycles: {al_config.num_cycles}")
        logger.info(f"  Samples per query: {al_config.batch_size_al}")
        logger.info(f"  Strategy: {al_config.sampling_strategy}")
        logger.info(f"  Reset mode: {al_config.reset_mode}")
    
    def run_cycle(self, cycle_num: int) -> Dict:
        """
        Execute one Active Learning cycle.
        
        Args:
            cycle_num: Current cycle number (1-indexed)
            
        Returns:
            Dict with cycle metrics
        """
        al_config = self.config.active_learning
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CYCLE {cycle_num}/{al_config.num_cycles}")
        logger.info(f"{'='*60}")
        
        # Step 1: Reset model based on configured mode
        reset_mode = al_config.reset_mode
        self.trainer.reset_model_weights(mode=reset_mode)
        
        # Step 2: Get current pool info
        pool_info = self.data_manager.get_pool_info()
        logger.info(
            f"Pool status: Labeled={pool_info['labeled']}, "
            f"Unlabeled={pool_info['unlabeled']}"
        )
        
        # Step 3: Get labeled loader and train
        train_loader = self.data_manager.get_labeled_loader(
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers
        )
        
        logger.info(f"Training on {pool_info['labeled']} labeled samples...")
        train_summary = self.trainer.train(train_loader, val_loader=self.val_loader)
        
        # Step 4: Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self.trainer.evaluate(self.test_loader)
        
        # Step 5: Save cycle checkpoint
        if self.config.checkpoint.save_best_per_cycle:
            self.trainer.save_cycle_checkpoint(cycle_num)
        
        # Step 6: Query new samples (if not final cycle and unlabeled pool not empty)
        queried_indices = []
        if cycle_num < al_config.num_cycles and pool_info['unlabeled'] > 0:
            logger.info(f"Querying {al_config.batch_size_al} samples...")
            
            unlabeled_loader = self.data_manager.get_unlabeled_loader(
                batch_size=self.config.training.batch_size,
                shuffle=False,
                num_workers=self.config.data.num_workers
            )
            
            # Apply strategy
            query_indices = self.strategy(
                self.trainer.model,
                unlabeled_loader,
                al_config.batch_size_al,
                self.trainer.device
            )
            
            # Step 7: Update pools (simulated annotation)
            absolute_indices = self.data_manager.update_labeled_pool(query_indices)
            queried_indices = absolute_indices
            
            # Save queried indices for analysis
            query_file = self.exp_dir / f"cycle_{cycle_num}_queried_indices.json"
            with open(query_file, "w") as f:
                json.dump(queried_indices, f)
        
        elif pool_info['unlabeled'] == 0:
            logger.info("No unlabeled samples remaining")
        else:
            logger.info("Final cycle - no querying")
        
        # Build cycle result
        cycle_result = {
            "cycle": cycle_num,
            "labeled_pool_size": pool_info['labeled'],
            "unlabeled_pool_size": pool_info['unlabeled'],
            "queried_count": len(queried_indices),
            "best_val_accuracy": train_summary.get("best_val_accuracy", 0),
            "best_epoch": train_summary.get("best_epoch", 0),
            "epochs_trained": train_summary.get("epochs_trained", 0),
            "test_accuracy": test_metrics["test_accuracy"],
            "test_f1": test_metrics["test_f1"],
            "test_precision": test_metrics["test_precision"],
            "test_recall": test_metrics["test_recall"],
        }
        
        self.cycle_results.append(cycle_result)
        self._save_results()
        
        logger.info(
            f"Cycle {cycle_num} complete | "
            f"Val Acc: {cycle_result['best_val_accuracy']:.4f}, "
            f"Test Acc: {cycle_result['test_accuracy']:.4f}"
        )
        
        return cycle_result
    
    def run_all_cycles(self) -> List[Dict]:
        """
        Execute all Active Learning cycles.
        
        Returns:
            List of cycle results
        """
        num_cycles = self.config.active_learning.num_cycles
        
        logger.info("\n" + "="*60)
        logger.info("STARTING ACTIVE LEARNING")
        logger.info("="*60)
        
        for cycle in range(1, num_cycles + 1):
            self.run_cycle(cycle)
        
        logger.info("\n" + "="*60)
        logger.info("ACTIVE LEARNING COMPLETE")
        logger.info("="*60)
        
        self._log_summary()
        
        # Save final pool state
        self.data_manager.save_state(self.exp_dir / "al_pool_state.json")
        
        return self.cycle_results
    
    def _save_results(self):
        """Save cycle results to JSON."""
        results_file = self.exp_dir / "al_cycle_results.json"
        
        output = {
            "strategy": self.config.active_learning.sampling_strategy,
            "num_cycles": self.config.active_learning.num_cycles,
            "initial_pool_size": self.config.active_learning.initial_pool_size,
            "batch_size_al": self.config.active_learning.batch_size_al,
            "cycles": self.cycle_results,
        }
        
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
    
    def _log_summary(self):
        """Log summary of all cycles."""
        logger.info("\nAL CYCLE SUMMARY")
        logger.info("-" * 70)
        logger.info(f"{'Cycle':<6} {'Labeled':<8} {'Val Acc':<10} {'Test Acc':<10} {'Test F1':<10}")
        logger.info("-" * 70)
        
        for r in self.cycle_results:
            logger.info(
                f"{r['cycle']:<6} {r['labeled_pool_size']:<8} "
                f"{r['best_val_accuracy']:<10.4f} {r['test_accuracy']:<10.4f} "
                f"{r['test_f1']:<10.4f}"
            )
        
        logger.info("-" * 70)
        
        # Best cycle
        best = max(self.cycle_results, key=lambda x: x["test_accuracy"])
        logger.info(f"Best test accuracy: {best['test_accuracy']:.4f} at cycle {best['cycle']}")
        
        # Improvement from first to last
        first_acc = self.cycle_results[0]["test_accuracy"]
        last_acc = self.cycle_results[-1]["test_accuracy"]
        improvement = last_acc - first_acc
        logger.info(f"Improvement: {improvement:+.4f} ({100*improvement/first_acc:+.1f}%)")
    
    def get_results(self) -> List[Dict]:
        """Get all cycle results."""
        return self.cycle_results.copy()
    
    def get_best_cycle(self) -> Optional[Dict]:
        """Get cycle with best test accuracy."""
        if not self.cycle_results:
            return None
        return max(self.cycle_results, key=lambda x: x["test_accuracy"])