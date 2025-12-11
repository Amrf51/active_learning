"""
Worker Command Loop Implementation

This module implements the main command loop for the AL worker process.
It polls the experiment state for commands from the dashboard and executes
the appropriate operations.

The command loop handles:
- START_CYCLE: Execute a complete AL cycle
- PAUSE: Pause current operation gracefully
- STOP: Stop worker and exit
- CONTINUE: Continue after annotations are submitted
"""

import logging
import time
from datetime import datetime
from typing import Optional

from src.state import (
    StateManager,
    ExperimentPhase,
    Command,
    EpochMetrics,
    CycleMetrics
)

logger = logging.getLogger(__name__)


def run_command_loop(
    state_manager: StateManager,
    al_loop,
    trainer,
    data_manager,
    config
) -> None:
    """
    Main command loop for the worker process.
    
    Continuously polls the experiment state for commands from the dashboard
    and executes the appropriate operations.
    
    Args:
        state_manager: StateManager instance
        al_loop: ActiveLearningLoop instance
        trainer: Trainer instance
        data_manager: ALDataManager instance
        config: Config object
    """
    logger.info("Starting worker command loop")
    
    # Initialize probe images on first run if needed
    state = state_manager.read_state()
    if state.current_cycle == 0 and not state.probe_images:
        logger.info("Initializing probe images for first run")
        probe_images = al_loop._initialize_probe_images(n_probes=12)
        state_manager.update_probe_images(probe_images)
    
    # Main command loop
    while True:
        try:
            # Update heartbeat
            state_manager.update_heartbeat()
            
            # Read current state
            state = state_manager.read_state()
            
            # Check for commands from dashboard
            if state.command == Command.START_CYCLE:
                logger.info("Received START_CYCLE command")
                
                # Check if we can start a new cycle
                if state.current_cycle >= state.config.num_cycles:
                    logger.warning("All cycles completed - ignoring START_CYCLE")
                    state_manager.clear_command()
                    state_manager.update_state(phase=ExperimentPhase.COMPLETED)
                    continue
                
                # Execute the cycle
                execute_cycle(state_manager, al_loop, trainer, data_manager, config)
                
                # Clear command after execution
                state_manager.clear_command()
                
            elif state.command == Command.PAUSE:
                logger.info("Received PAUSE command")
                
                # Set paused state
                state_manager.update_state(
                    phase=ExperimentPhase.IDLE,
                    command=None
                )
                
                logger.info("Worker paused - waiting for commands")
                
            elif state.command == Command.STOP:
                logger.info("Received STOP command")
                
                # Save final state
                save_final_state(state_manager, data_manager)
                
                logger.info("Worker stopping gracefully")
                break
                
            elif state.command == Command.CONTINUE:
                logger.info("Received CONTINUE command")
                
                # Process annotations and continue
                handle_continue_command(state_manager, al_loop, data_manager)
                
                # Clear command
                state_manager.clear_command()
                
            elif state.phase == ExperimentPhase.AWAITING_ANNOTATION:
                # Check if annotations have been submitted
                if state_manager.annotations_pending():
                    logger.info("Annotations detected - processing automatically")
                    handle_continue_command(state_manager, al_loop, data_manager)
                else:
                    # Wait for annotations
                    time.sleep(2.0)
                    
            elif state.phase == ExperimentPhase.IDLE:
                # Wait for commands
                time.sleep(1.0)
                
            elif state.phase == ExperimentPhase.COMPLETED:
                logger.info("Experiment completed - worker can exit")
                break
                
            elif state.phase == ExperimentPhase.ERROR:
                logger.error("Experiment in error state - worker exiting")
                break
                
            else:
                # Unknown phase - wait and continue
                logger.warning(f"Unknown phase: {state.phase} - waiting")
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            logger.info("Worker interrupted by user")
            break
            
        except Exception as e:
            logger.error(f"Error in command loop: {e}", exc_info=True)
            
            # Set error state
            try:
                state_manager.update_state(
                    phase=ExperimentPhase.ERROR,
                    error_message=str(e)
                )
            except:
                pass
            
            # Wait before retrying
            time.sleep(5.0)
    
    logger.info("Command loop exited")


def execute_cycle(
    state_manager: StateManager,
    al_loop,
    trainer,
    data_manager,
    config
) -> None:
    """
    Execute one complete AL cycle.
    
    Phases:
    1. PREPARING: Call al_loop.prepare_cycle()
    2. TRAINING: Train for configured epochs with pause/stop checks
    3. EVALUATING: Run evaluation and save confusion matrix
    4. QUERYING: Query samples for annotation
    5. AWAITING_ANNOTATION: Wait for user annotations
    
    Args:
        state_manager: StateManager instance
        al_loop: ActiveLearningLoop instance
        trainer: Trainer instance
        data_manager: ALDataManager instance
        config: Config object
    """
    logger.info("Starting AL cycle execution")
    
    try:
        # Get current state
        state = state_manager.read_state()
        cycle_num = state.current_cycle + 1
        
        logger.info(f"Executing cycle {cycle_num}")
        
        # Phase 1: PREPARING
        logger.info("Phase 1: Preparing cycle")
        state_manager.update_state(phase=ExperimentPhase.PREPARING)
        
        cycle_info = al_loop.prepare_cycle(cycle_num)
        
        # Update state with cycle info
        state_manager.update_state(
            current_cycle=cycle_num,
            labeled_count=cycle_info["labeled_count"],
            unlabeled_count=cycle_info["unlabeled_count"],
            current_cycle_epochs=[]  # Reset epoch history for new cycle
        )
        
        logger.info(f"Cycle prepared - Labeled: {cycle_info['labeled_count']}, Unlabeled: {cycle_info['unlabeled_count']}")
        
        # Phase 2: TRAINING
        logger.info("Phase 2: Training")
        state_manager.update_state(phase=ExperimentPhase.TRAINING)
        
        epochs = config.training.epochs
        logger.info(f"Training for {epochs} epochs")
        
        for epoch in range(1, epochs + 1):
            # Check for PAUSE/STOP commands before each epoch
            command = check_pause_stop_commands(state_manager)
            if command == Command.PAUSE:
                logger.info(f"Pause requested during training at epoch {epoch}")
                state_manager.update_state(phase=ExperimentPhase.IDLE)
                return
            elif command == Command.STOP:
                logger.info(f"Stop requested during training at epoch {epoch}")
                save_final_state(state_manager, data_manager)
                return
            
            # Train one epoch
            logger.info(f"Training epoch {epoch}/{epochs}")
            epoch_metrics = al_loop.train_single_epoch(epoch)
            
            # Update state with epoch metrics
            state_manager.add_epoch_metrics(epoch_metrics)
            
            logger.info(
                f"Epoch {epoch} complete - "
                f"Train Loss: {epoch_metrics.train_loss:.4f}, "
                f"Train Acc: {epoch_metrics.train_accuracy:.4f}"
                + (f", Val Acc: {epoch_metrics.val_accuracy:.4f}" if epoch_metrics.val_accuracy else "")
            )
            
            # Check early stopping
            if al_loop.should_stop_early():
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Phase 3: EVALUATING
        logger.info("Phase 3: Evaluating")
        state_manager.update_state(phase=ExperimentPhase.EVALUATING)
        
        # Run evaluation with confusion matrix saving
        cm_path = state_manager.experiment_dir / f"cycle_{cycle_num}_confusion_matrix.npy"
        test_metrics = trainer.evaluate(
            al_loop.test_loader,
            class_names=al_loop.class_names,
            save_cm_path=cm_path
        )
        
        logger.info(
            f"Evaluation complete - "
            f"Test Acc: {test_metrics['test_accuracy']:.4f}, "
            f"Test F1: {test_metrics['test_f1']:.4f}"
        )
        
        # Update probe image predictions
        logger.info("Updating probe image predictions")
        al_loop._update_probe_predictions(cycle_num)
        state_manager.update_probe_images(al_loop.probe_images)
        
        # Finalize cycle metrics
        cycle_metrics = al_loop.finalize_cycle(test_metrics)
        state_manager.finalize_cycle(cycle_metrics)
        
        # Save state persistence after cycle completion
        save_cycle_state(state_manager, trainer, data_manager, cycle_num)
        
        # Phase 4: QUERYING
        logger.info("Phase 4: Querying samples")
        state_manager.update_state(phase=ExperimentPhase.QUERYING)
        
        # Check if we have more cycles to run
        if cycle_num >= config.active_learning.num_cycles:
            logger.info("All cycles completed")
            state_manager.update_state(phase=ExperimentPhase.COMPLETED)
            return
        
        # Check if we have unlabeled samples
        pool_info = data_manager.get_pool_info()
        if pool_info["unlabeled"] == 0:
            logger.info("No unlabeled samples remaining - experiment complete")
            state_manager.update_state(phase=ExperimentPhase.COMPLETED)
            return
        
        # Query samples for annotation
        queried_images = al_loop.query_samples()
        
        if not queried_images:
            logger.warning("No samples queried - experiment complete")
            state_manager.update_state(phase=ExperimentPhase.COMPLETED)
            return
        
        # Update state with queried images
        state_manager.set_queried_images(queried_images)
        
        logger.info(f"Queried {len(queried_images)} samples for annotation")
        
        # Phase 5: AWAITING_ANNOTATION
        logger.info("Phase 5: Awaiting annotations")
        state_manager.update_state(phase=ExperimentPhase.AWAITING_ANNOTATION)
        
        logger.info(f"Cycle {cycle_num} execution complete - waiting for annotations")
        
    except Exception as e:
        logger.error(f"Error during cycle execution: {e}", exc_info=True)
        state_manager.update_state(
            phase=ExperimentPhase.ERROR,
            error_message=f"Cycle execution failed: {e}"
        )
        raise


def handle_continue_command(
    state_manager: StateManager,
    al_loop,
    data_manager
) -> None:
    """
    Handle CONTINUE command after annotations are submitted.
    
    Args:
        state_manager: StateManager instance
        al_loop: ActiveLearningLoop instance
        data_manager: ALDataManager instance
    """
    logger.info("Processing annotations and continuing")
    
    try:
        # Read annotations
        annotations = state_manager.read_annotations()
        
        if annotations is None:
            logger.warning("No annotations found - cannot continue")
            return
        
        # Convert annotations to format expected by AL loop
        annotation_list = []
        for ann in annotations.annotations:
            annotation_list.append({
                "image_id": ann.image_id,
                "user_label": ann.user_label
            })
        
        # Process annotations
        result = al_loop.receive_annotations(annotation_list)
        
        logger.info(f"Annotations processed: {result['moved_count']} samples added to labeled pool")
        
        # Clear annotations file
        state_manager.clear_annotations()
        
        # Update state to idle (ready for next cycle)
        state_manager.update_state(phase=ExperimentPhase.IDLE)
        
    except Exception as e:
        logger.error(f"Error processing annotations: {e}", exc_info=True)
        state_manager.update_state(
            phase=ExperimentPhase.ERROR,
            error_message=f"Annotation processing failed: {e}"
        )


def save_cycle_state(
    state_manager: StateManager,
    trainer,
    data_manager,
    cycle_num: int
) -> None:
    """
    Save state persistence after each cycle completion.
    
    This includes:
    - Pool state (labeled/unlabeled indices) to JSON
    - Model checkpoint for best model
    - Cycle metrics already saved via state_manager.finalize_cycle()
    
    Args:
        state_manager: StateManager instance
        trainer: Trainer instance
        data_manager: ALDataManager instance
        cycle_num: Current cycle number
    """
    logger.info(f"Saving state after cycle {cycle_num}")
    
    try:
        # Save pool state to JSON
        pool_state_file = state_manager.experiment_dir / f"cycle_{cycle_num}_pool_state.json"
        data_manager.save_state(pool_state_file)
        logger.info(f"Pool state saved to {pool_state_file}")
        
        # Save model checkpoint for best model in this cycle
        trainer.save_cycle_checkpoint(cycle_num)
        logger.info(f"Model checkpoint saved for cycle {cycle_num}")
        
        # Save overall pool state (for resuming)
        overall_pool_state_file = state_manager.experiment_dir / "al_pool_state.json"
        data_manager.save_state(overall_pool_state_file)
        
        # Save training history
        trainer.save_training_log()
        
        logger.info(f"Cycle {cycle_num} state persistence complete")
        
    except Exception as e:
        logger.error(f"Error saving cycle state: {e}", exc_info=True)
        # Don't raise - this shouldn't stop the experiment


def save_final_state(
    state_manager: StateManager,
    data_manager
) -> None:
    """
    Save final experiment state before worker exits.
    
    Args:
        state_manager: StateManager instance
        data_manager: ALDataManager instance
    """
    logger.info("Saving final experiment state")
    
    try:
        # Save pool state
        pool_state_file = state_manager.experiment_dir / "al_pool_state.json"
        data_manager.save_state(pool_state_file)
        
        # Update final state
        state_manager.update_state(
            phase=ExperimentPhase.IDLE,
            worker_pid=None,
            last_heartbeat=datetime.now()
        )
        
        logger.info("Final state saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving final state: {e}", exc_info=True)


def check_pause_stop_commands(state_manager: StateManager) -> Optional[Command]:
    """
    Check for PAUSE or STOP commands during long operations.
    
    This function should be called periodically during training epochs
    to allow graceful interruption.
    
    Args:
        state_manager: StateManager instance
        
    Returns:
        Command if PAUSE or STOP detected, None otherwise
    """
    try:
        state = state_manager.read_state()
        
        if state.command in [Command.PAUSE, Command.STOP]:
            return state.command
        
        return None
        
    except Exception as e:
        logger.warning(f"Error checking for pause/stop commands: {e}")
        return None