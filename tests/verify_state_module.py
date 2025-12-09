"""
Quick verification script for the state management module.

Run from project root:
    python scripts/verify_state_module.py

This simulates a complete experiment lifecycle to verify
all state management components work correctly.
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.state import (
    ExperimentPhase,
    ExperimentState,
    ExperimentConfig,
    EpochMetrics,
    CycleMetrics,
    QueriedImage,
    ProbeImage,
    UserAnnotation,
    AnnotationSubmission,
    DatasetInfo,
    StateManager,
    ExperimentManager,
)
from src.config import Config


def create_test_config():
    """Create a sample experiment config."""
    return ExperimentConfig(
        model_name="resnet18",
        pretrained=True,
        num_classes=4,
        class_names=["bus", "car", "motorcycle", "truck"],
        epochs_per_cycle=20,
        batch_size=32,
        learning_rate=0.001,
        optimizer="adam",
        weight_decay=0.0001,
        early_stopping_patience=5,
        num_cycles=3,
        sampling_strategy="uncertainty",
        uncertainty_method="least_confidence",
        initial_pool_size=50,
        batch_size_al=20,
        reset_mode="pretrained",
        seed=42,
        data_dir="./data/raw/kaggle-vehicle/",
        val_split=0.15,
        test_split=0.15,
        augmentation=True,
    )


def simulate_experiment_lifecycle(base_dir: Path):
    """Simulate a complete experiment lifecycle."""
    
    print("\n" + "="*60)
    print("STATE MODULE VERIFICATION")
    print("="*60)
    
    exp_manager = ExperimentManager(base_dir)
    
    print("\n[1] Creating experiment...")
    exp_dir = exp_manager.create_experiment("verification_test")
    print(f"    Created: {exp_dir}")
    
    state_manager = StateManager(exp_dir)
    
    print("\n[2] Initializing experiment state...")
    state = state_manager.initialize_state(
        experiment_id=exp_dir.name,
        experiment_name="Verification Test Experiment"
    )
    print(f"    State initialized with phase: {state.phase}")
    
    print("\n[3] Setting experiment as active...")
    exp_manager.set_active(exp_dir)
    active = exp_manager.get_active()
    print(f"    Active experiment: {active.experiment_id}")
    
    print("\n[4] Simulating INITIALIZING phase...")
    config = create_test_config()
    dataset_info = DatasetInfo(
        total_images=400,
        num_classes=4,
        class_names=["bus", "car", "motorcycle", "truck"],
        class_counts={"bus": 100, "car": 100, "motorcycle": 100, "truck": 100},
        train_samples=280,
        val_samples=60,
        test_samples=60,
    )
    
    state = state_manager.update_state(
        phase=ExperimentPhase.INITIALIZING,
        worker_pid=12345,
        config=config,
        dataset_info=dataset_info,
        total_cycles=config.num_cycles,
        total_train_samples=280,
        labeled_count=50,
        unlabeled_count=230,
    )
    print(f"    Phase: {state.phase}")
    print(f"    Config loaded: {state.config.model_name}")
    
    print("\n[5] Setting up probe images...")
    probes = [
        ProbeImage(
            image_id=10,
            image_path="/data/bus_010.jpg",
            display_path=str(exp_dir / "probes/probe_10_easy.jpg"),
            true_class="bus",
            true_class_idx=0,
            probe_type="easy",
            predictions_by_cycle={0: {"bus": 0.85, "car": 0.05, "motorcycle": 0.05, "truck": 0.05}}
        ),
        ProbeImage(
            image_id=150,
            image_path="/data/car_050.jpg",
            display_path=str(exp_dir / "probes/probe_150_hard.jpg"),
            true_class="car",
            true_class_idx=1,
            probe_type="hard",
            predictions_by_cycle={0: {"bus": 0.30, "car": 0.25, "motorcycle": 0.20, "truck": 0.25}}
        ),
    ]
    state_manager.update_state(probe_images=probes)
    print(f"    Added {len(probes)} probe images")
    
    for cycle in range(1, 3):
        print(f"\n[6.{cycle}] Simulating CYCLE {cycle}...")
        
        state_manager.update_state(
            phase=ExperimentPhase.TRAINING,
            current_cycle=cycle,
            current_epoch=0,
            epochs_in_cycle=20,
        )
        print(f"    Phase: TRAINING")
        
        for epoch in range(1, 6):
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=1.5 - (epoch * 0.1) - (cycle * 0.2),
                train_accuracy=0.4 + (epoch * 0.05) + (cycle * 0.1),
                val_loss=1.6 - (epoch * 0.08) - (cycle * 0.15),
                val_accuracy=0.35 + (epoch * 0.04) + (cycle * 0.08),
            )
            state_manager.add_epoch_metrics(metrics)
        
        state = state_manager.read_state()
        print(f"    Trained {len(state.current_cycle_epochs)} epochs")
        print(f"    Last train_acc: {state.current_cycle_epochs[-1].train_accuracy:.3f}")
        
        state_manager.update_state(phase=ExperimentPhase.VALIDATING)
        state_manager.update_heartbeat()
        
        state_manager.update_state(phase=ExperimentPhase.EVALUATING)
        
        cycle_metrics = CycleMetrics(
            cycle=cycle,
            labeled_pool_size=50 + (cycle - 1) * 20,
            unlabeled_pool_size=230 - (cycle - 1) * 20,
            epochs_trained=5,
            best_val_accuracy=0.35 + (5 * 0.04) + (cycle * 0.08),
            best_epoch=5,
            test_accuracy=0.50 + (cycle * 0.1),
            test_f1=0.48 + (cycle * 0.1),
            test_precision=0.52 + (cycle * 0.1),
            test_recall=0.47 + (cycle * 0.1),
        )
        state_manager.finalize_cycle(cycle_metrics)
        print(f"    Test accuracy: {cycle_metrics.test_accuracy:.3f}")
        
        if cycle < 2:
            print(f"\n    Simulating QUERYING phase...")
            state_manager.update_state(phase=ExperimentPhase.QUERYING)
            
            queried = [
                QueriedImage(
                    image_id=100 + i,
                    image_path=f"/data/image_{100+i}.jpg",
                    display_path=str(exp_dir / f"queries/cycle_{cycle}/{100+i}.jpg"),
                    ground_truth=i % 4,
                    ground_truth_name=["bus", "car", "motorcycle", "truck"][i % 4],
                    model_probabilities={"bus": 0.25, "car": 0.25, "motorcycle": 0.25, "truck": 0.25},
                    predicted_class=["bus", "car", "motorcycle", "truck"][i % 4],
                    predicted_confidence=0.25 + (i * 0.02),
                    uncertainty_score=0.75 - (i * 0.02),
                    selection_reason=f"Low confidence: {25 + i*2}%"
                )
                for i in range(5)
            ]
            state_manager.set_queried_images(queried)
            print(f"    Queried {len(queried)} images, awaiting annotation...")
            
            state = state_manager.read_state()
            assert state.phase == ExperimentPhase.AWAITING_ANNOTATION
            
            print(f"\n    Simulating user annotation...")
            annotations = AnnotationSubmission(
                experiment_id=exp_dir.name,
                cycle=cycle,
                annotations=[
                    UserAnnotation(
                        image_id=img.image_id,
                        user_label=img.ground_truth,
                        user_label_name=img.ground_truth_name,
                        timestamp=datetime.now(),
                        was_correct=True
                    )
                    for img in queried
                ],
                submitted_at=datetime.now()
            )
            state_manager.write_annotations(annotations)
            
            assert state_manager.annotations_pending()
            loaded_annotations = state_manager.read_annotations()
            print(f"    User submitted {len(loaded_annotations.annotations)} annotations")
            
            state_manager.update_state(phase=ExperimentPhase.ANNOTATIONS_SUBMITTED)
            state_manager.clear_annotations()
            
            state_manager.update_state(
                labeled_count=50 + cycle * 20,
                unlabeled_count=230 - cycle * 20,
                queried_images=[],
            )
    
    print("\n[7] Completing experiment...")
    state_manager.update_state(phase=ExperimentPhase.COMPLETED)
    
    print("\n[8] Verifying final state...")
    final_state = state_manager.read_state()
    
    assert final_state.phase == ExperimentPhase.COMPLETED
    assert len(final_state.cycle_results) == 2
    assert final_state.cycle_results[0].test_accuracy < final_state.cycle_results[1].test_accuracy
    
    print(f"    Final phase: {final_state.phase}")
    print(f"    Cycles completed: {len(final_state.cycle_results)}")
    print(f"    Cycle 1 test acc: {final_state.cycle_results[0].test_accuracy:.3f}")
    print(f"    Cycle 2 test acc: {final_state.cycle_results[1].test_accuracy:.3f}")
    
    print("\n[9] Testing ExperimentManager listing...")
    experiments = exp_manager.list_experiments()
    print(f"    Found {len(experiments)} experiment(s)")
    
    print("\n[10] Testing worker alive detection...")
    state_manager.update_heartbeat()
    assert state_manager.is_worker_alive(timeout_seconds=30)
    print("    Worker heartbeat: alive")
    
    print("\n" + "="*60)
    print("ALL VERIFICATIONS PASSED")
    print("="*60)
    
    return True


def main():
    """Run verification."""
    test_dir = Path("./test_state_verification")
    
    try:
        if test_dir.exists():
            shutil.rmtree(test_dir)
        
        success = simulate_experiment_lifecycle(test_dir)
        
        if success:
            print("\nState module is ready for use.")
            print(f"\nTest artifacts created in: {test_dir}")
            print("You can inspect the JSON files to see the state structure.")
            print("\nTo clean up, run:")
            print(f"    rm -rf {test_dir}")
        
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())