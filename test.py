#!/usr/bin/env python3
"""
Test Script for MVC Architecture - Phases 1, 2, 3

This script tests:
- Phase 1: Event system
- Phase 2: Model layer (WorldState, DatabaseManager, Schemas)
- Phase 3: Service layer (ActiveLearningService with mock mode)

Run with:
    python test_phases_1_2_3.py
"""

import sys
import time
import tempfile
from pathlib import Path
from datetime import datetime
from multiprocessing import Process, Pipe
import threading

# Add the mvc_implementation directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("MVC Architecture Test - Phases 1, 2, 3")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 TESTS: Event System
# ═══════════════════════════════════════════════════════════════════════════

def test_phase1_events():
    """Test the event system."""
    print("\n" + "─" * 70)
    print("PHASE 1: Event System Tests")
    print("─" * 70)
    
    from controller.events import (
        Event, EventType, create_event,
        start_cycle_event, epoch_complete_event, error_event
    )
    
    # Test 1: Create basic event
    print("\n[Test 1.1] Create basic event...")
    event = Event(EventType.START_CYCLE)
    assert event.type == EventType.START_CYCLE
    assert event.payload == {}
    assert event.source == "unknown"
    print(f"  ✓ Created: {event}")
    
    # Test 2: Create event with payload
    print("\n[Test 1.2] Create event with payload...")
    event = Event(
        EventType.EPOCH_COMPLETE,
        payload={"epoch": 5, "train_loss": 0.123}
    )
    assert event.payload["epoch"] == 5
    assert event.payload["train_loss"] == 0.123
    print(f"  ✓ Created: {event}")
    
    # Test 3: Convenience functions
    print("\n[Test 1.3] Convenience functions...")
    event = start_cycle_event(source="test")
    assert event.type == EventType.START_CYCLE
    assert event.source == "test"
    print(f"  ✓ start_cycle_event: {event}")
    
    event = epoch_complete_event(
        epoch=3, train_loss=0.5, train_accuracy=0.8,
        val_loss=0.6, val_accuracy=0.75, learning_rate=0.001
    )
    assert event.payload["epoch"] == 3
    print(f"  ✓ epoch_complete_event: {event}")
    
    event = error_event("Test error", traceback="...", recoverable=True)
    assert event.type == EventType.SERVICE_ERROR
    assert event.payload["message"] == "Test error"
    print(f"  ✓ error_event: {event}")
    
    # Test 4: Serialization round-trip
    print("\n[Test 1.4] Serialization round-trip...")
    original = Event(
        EventType.QUERY_READY,
        payload={"queried_images": [{"id": 1}, {"id": 2}]},
        source="service"
    )
    serialized = original.to_dict()
    restored = Event.from_dict(serialized)
    
    assert restored.type == original.type
    assert restored.payload == original.payload
    assert restored.source == original.source
    print(f"  ✓ Original:  {original}")
    print(f"  ✓ Restored:  {restored}")
    
    # Test 5: All event types exist
    print("\n[Test 1.5] Event type enumeration...")
    view_events = [e for e in EventType if not e.name.startswith("CMD_") and not e.name.startswith("SERVICE_")]
    cmd_events = [e for e in EventType if e.name.startswith("CMD_")]
    service_events = [e for e in EventType if e.name.startswith("SERVICE_") or e.name in ["EPOCH_COMPLETE", "QUERY_READY", "CYCLE_COMPLETE"]]
    
    print(f"  ✓ View→Controller events: {len(view_events)}")
    print(f"  ✓ Controller→Service commands: {len(cmd_events)}")
    print(f"  ✓ Service→Controller events: {len(service_events)}")
    
    print("\n✅ PHASE 1 PASSED: Event system working correctly")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 TESTS: Model Layer
# ═══════════════════════════════════════════════════════════════════════════

def test_phase2_schemas():
    """Test data schemas."""
    print("\n" + "─" * 70)
    print("PHASE 2.1: Schema Tests")
    print("─" * 70)
    
    from model.schemas import (
        ExperimentPhase, EpochMetrics, CycleSummary,
        ExperimentConfig, QueriedImage, ValidationResult
    )
    
    # Test EpochMetrics
    print("\n[Test 2.1.1] EpochMetrics...")
    metrics = EpochMetrics(
        epoch=5,
        train_loss=0.234,
        train_accuracy=0.89,
        val_loss=0.345,
        val_accuracy=0.85,
        learning_rate=0.001
    )
    
    # Serialize and deserialize
    d = metrics.to_dict()
    restored = EpochMetrics.from_dict(d)
    assert restored.epoch == metrics.epoch
    assert restored.train_loss == metrics.train_loss
    print(f"  ✓ Created and serialized: epoch={metrics.epoch}, val_acc={metrics.val_accuracy}")
    
    # Test CycleSummary
    print("\n[Test 2.1.2] CycleSummary...")
    summary = CycleSummary(
        cycle=1,
        labeled_count=100,
        unlabeled_count=300,
        epochs_trained=10,
        best_val_accuracy=0.92,
        best_epoch=8,
        test_accuracy=0.88,
        test_f1=0.86,
        started_at=datetime.now()
    )
    
    d = summary.to_dict()
    restored = CycleSummary.from_dict(d)
    assert restored.cycle == summary.cycle
    assert restored.test_accuracy == summary.test_accuracy
    print(f"  ✓ Created: cycle={summary.cycle}, test_acc={summary.test_accuracy}")
    
    # Test ExperimentConfig
    print("\n[Test 2.1.3] ExperimentConfig...")
    config = ExperimentConfig(
        model_name="resnet18",
        num_cycles=5,
        epochs_per_cycle=10,
        sampling_strategy="uncertainty"
    )
    
    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)
    assert restored.model_name == config.model_name
    assert restored.num_cycles == config.num_cycles
    print(f"  ✓ Created: model={config.model_name}, cycles={config.num_cycles}")
    
    # Test ValidationResult
    print("\n[Test 2.1.4] ValidationResult...")
    result = ValidationResult(is_valid=True)
    result.add_warning("Consider increasing batch size")
    assert result.is_valid == True
    assert len(result.warnings) == 1
    
    result.add_error("Data directory not found")
    assert result.is_valid == False
    print(f"  ✓ Validation: is_valid={result.is_valid}, errors={result.errors}")
    
    print("\n✅ PHASE 2.1 PASSED: Schemas working correctly")
    return True


def test_phase2_worldstate():
    """Test WorldState."""
    print("\n" + "─" * 70)
    print("PHASE 2.2: WorldState Tests")
    print("─" * 70)
    
    from model.world_state import WorldState
    from model.schemas import ExperimentPhase, ExperimentConfig, EpochMetrics
    
    # Test 1: Create empty WorldState
    print("\n[Test 2.2.1] Create empty WorldState...")
    ws = WorldState()
    assert ws.experiment_id is None
    assert ws.phase == ExperimentPhase.IDLE
    assert ws.is_initialized == False
    print(f"  ✓ Created: {ws}")
    
    # Test 2: Initialize experiment
    print("\n[Test 2.2.2] Initialize experiment...")
    config = ExperimentConfig(
        model_name="resnet18",
        num_cycles=5,
        epochs_per_cycle=10
    )
    ws.initialize(
        experiment_id="exp_001",
        experiment_name="Test Experiment",
        config=config
    )
    assert ws.experiment_id == "exp_001"
    assert ws.is_initialized == True
    assert ws.total_cycles == 5
    assert ws.pending_updates == True
    print(f"  ✓ Initialized: {ws}")
    
    # Test 3: Phase transitions
    print("\n[Test 2.2.3] Phase transitions...")
    ws.set_phase(ExperimentPhase.TRAINING)
    assert ws.phase == ExperimentPhase.TRAINING
    assert ws.is_running == True
    print(f"  ✓ Phase: {ws.phase.value}")
    
    ws.set_error("Test error message")
    assert ws.phase == ExperimentPhase.ERROR
    assert ws.has_error == True
    assert ws.error_message == "Test error message"
    print(f"  ✓ Error state: {ws.error_message}")
    
    # Test 4: Cycle tracking
    print("\n[Test 2.2.4] Cycle tracking...")
    ws.set_phase(ExperimentPhase.TRAINING)
    ws.start_cycle(cycle=1, labeled_count=50, unlabeled_count=350)
    assert ws.current_cycle == 1
    assert ws.labeled_count == 50
    print(f"  ✓ Started cycle: {ws.current_cycle}, labeled={ws.labeled_count}")
    
    # Test 5: Epoch updates
    print("\n[Test 2.2.5] Epoch updates...")
    for epoch in range(1, 4):
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=1.0 - (epoch * 0.1),
            train_accuracy=0.5 + (epoch * 0.1),
            val_loss=1.1 - (epoch * 0.1),
            val_accuracy=0.45 + (epoch * 0.1)
        )
        ws.update_epoch(metrics)
    
    assert ws.current_epoch == 3
    assert len(ws.epoch_history) == 3
    assert ws.current_metrics.epoch == 3
    print(f"  ✓ Epochs recorded: {len(ws.epoch_history)}")
    
    # Test 6: Progress calculation
    print("\n[Test 2.2.6] Progress calculation...")
    progress = ws.progress_percentage
    print(f"  ✓ Progress: {progress:.1f}%")
    
    # Test 7: Thread safety
    print("\n[Test 2.2.7] Thread safety...")
    ws.clear_pending_updates()
    
    def update_from_thread():
        with ws.lock:
            ws.current_epoch = 999
            ws.pending_updates = True
    
    thread = threading.Thread(target=update_from_thread)
    thread.start()
    thread.join()
    
    assert ws.current_epoch == 999
    assert ws.pending_updates == True
    print(f"  ✓ Thread update successful: epoch={ws.current_epoch}")
    
    # Test 8: Reset
    print("\n[Test 2.2.8] Reset...")
    ws.reset()
    assert ws.experiment_id is None
    assert ws.phase == ExperimentPhase.IDLE
    assert ws.is_initialized == False
    print(f"  ✓ Reset successful: {ws}")
    
    print("\n✅ PHASE 2.2 PASSED: WorldState working correctly")
    return True


def test_phase2_database():
    """Test DatabaseManager."""
    print("\n" + "─" * 70)
    print("PHASE 2.3: DatabaseManager Tests")
    print("─" * 70)
    
    from model.database import DatabaseManager
    from model.schemas import EpochMetrics, CycleSummary
    
    # Use in-memory database for testing
    print("\n[Test 2.3.1] Create in-memory database...")
    db = DatabaseManager(":memory:")
    stats = db.get_statistics()
    print(f"  ✓ Database created: {stats}")
    
    # Test 2: Insert experiment
    print("\n[Test 2.3.2] Insert experiment...")
    db.insert_experiment(
        experiment_id="exp_001",
        experiment_name="Test Experiment",
        config={"model_name": "resnet18", "num_cycles": 5}
    )
    
    exp = db.get_experiment("exp_001")
    assert exp is not None
    assert exp["experiment_name"] == "Test Experiment"
    assert exp["config"]["model_name"] == "resnet18"
    print(f"  ✓ Inserted: {exp['experiment_id']}")
    
    # Test 3: Active experiment
    print("\n[Test 2.3.3] Active experiment...")
    active = db.get_active_experiment()
    assert active is not None
    assert active["experiment_id"] == "exp_001"
    print(f"  ✓ Active: {active['experiment_id']}")
    
    # Test 4: Update phase
    print("\n[Test 2.3.4] Update phase...")
    db.update_experiment_phase("exp_001", "TRAINING")
    exp = db.get_experiment("exp_001")
    assert exp["phase"] == "TRAINING"
    print(f"  ✓ Phase updated: {exp['phase']}")
    
    # Test 5: Insert epoch metrics
    print("\n[Test 2.3.5] Insert epoch metrics...")
    for epoch in range(1, 11):
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=1.0 - (epoch * 0.05),
            train_accuracy=0.5 + (epoch * 0.03),
            val_loss=1.1 - (epoch * 0.04),
            val_accuracy=0.45 + (epoch * 0.035)
        )
        db.insert_epoch_metrics("exp_001", cycle=1, metrics=metrics)
    
    all_metrics = db.get_epoch_metrics("exp_001", cycle=1)
    assert len(all_metrics) == 10
    print(f"  ✓ Inserted 10 epochs, retrieved {len(all_metrics)}")
    
    # Test 6: Paginated metrics
    print("\n[Test 2.3.6] Paginated metrics...")
    page1, total = db.get_epoch_metrics_paginated("exp_001", cycle=1, page=1, limit=5)
    assert len(page1) == 5
    assert total == 10
    assert page1[0].epoch == 1
    
    page2, _ = db.get_epoch_metrics_paginated("exp_001", cycle=1, page=2, limit=5)
    assert len(page2) == 5
    assert page2[0].epoch == 6
    print(f"  ✓ Page 1: epochs {[m.epoch for m in page1]}")
    print(f"  ✓ Page 2: epochs {[m.epoch for m in page2]}")
    
    # Test 7: Insert cycle summary
    print("\n[Test 2.3.7] Insert cycle summary...")
    summary = CycleSummary(
        cycle=1,
        labeled_count=60,
        unlabeled_count=340,
        epochs_trained=10,
        best_val_accuracy=0.89,
        best_epoch=8,
        test_accuracy=0.85,
        test_f1=0.83,
        completed_at=datetime.now()
    )
    db.insert_cycle_summary("exp_001", summary)
    
    summaries = db.get_cycle_summaries("exp_001")
    assert len(summaries) == 1
    assert summaries[0].test_accuracy == 0.85
    print(f"  ✓ Cycle summary: test_acc={summaries[0].test_accuracy}")
    
    # Test 8: List experiments
    print("\n[Test 2.3.8] List experiments...")
    # Add another experiment
    db.insert_experiment("exp_002", "Second Experiment", {"model_name": "resnet50"})
    
    experiments, total = db.list_experiments()
    assert total == 2
    print(f"  ✓ Listed {total} experiments")
    
    # Test 9: Statistics
    print("\n[Test 2.3.9] Statistics...")
    stats = db.get_statistics()
    assert stats["experiments"] == 2
    assert stats["cycles"] == 1
    assert stats["epochs"] == 10
    print(f"  ✓ Stats: {stats}")
    
    print("\n✅ PHASE 2.3 PASSED: DatabaseManager working correctly")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 TESTS: Service Layer
# ═══════════════════════════════════════════════════════════════════════════

def test_phase3_service():
    """Test ActiveLearningService with mock mode."""
    print("\n" + "─" * 70)
    print("PHASE 3: Service Layer Tests")
    print("─" * 70)
    
    from controller.events import Event, EventType
    from services.al_service import run_active_learning_service
    
    # Test 1: Service starts and sends SERVICE_READY
    print("\n[Test 3.1] Service startup...")
    
    parent_conn, child_conn = Pipe()
    
    config = {
        "experiment_dir": "/tmp/test_exp",
        "model_name": "resnet18",
        "num_cycles": 3,
        "epochs_per_cycle": 3,  # Short for testing
        "batch_size_al": 5
    }
    
    # Start service process
    process = Process(target=run_active_learning_service, args=(child_conn, config))
    process.start()
    
    # Wait for SERVICE_READY
    print("  Waiting for SERVICE_READY...")
    if parent_conn.poll(timeout=10):
        event_dict = parent_conn.recv()
        event = Event.from_dict(event_dict)
        assert event.type == EventType.SERVICE_READY
        print(f"  ✓ Received: {event.type.name}")
    else:
        print("  ✗ Timeout waiting for SERVICE_READY")
        process.terminate()
        return False
    
    # Test 2: Send CMD_START_CYCLE and receive progress
    print("\n[Test 3.2] Training cycle (mock mode)...")
    
    start_event = Event(EventType.CMD_START_CYCLE, payload={"cycle": 1}, source="test")
    parent_conn.send(start_event.to_dict())
    
    epoch_events = []
    training_complete = False
    query_ready = False
    
    print("  Receiving events...")
    timeout = time.time() + 30  # 30 second timeout
    
    while time.time() < timeout:
        if parent_conn.poll(timeout=1):
            event_dict = parent_conn.recv()
            event = Event.from_dict(event_dict)
            
            if event.type == EventType.EPOCH_COMPLETE:
                epoch_events.append(event)
                print(f"    → EPOCH_COMPLETE: epoch={event.payload['epoch']}, "
                      f"train_loss={event.payload['train_loss']:.3f}")
                      
            elif event.type == EventType.TRAINING_COMPLETE:
                training_complete = True
                print(f"    → TRAINING_COMPLETE")
                
            elif event.type == EventType.EVALUATION_COMPLETE:
                print(f"    → EVALUATION_COMPLETE: test_acc={event.payload['test_accuracy']:.3f}")
                
            elif event.type == EventType.QUERY_READY:
                query_ready = True
                print(f"    → QUERY_READY: {event.payload['count']} images")
                break
                
            elif event.type == EventType.SERVICE_ERROR:
                print(f"    ✗ SERVICE_ERROR: {event.payload['message']}")
                break
    
    assert len(epoch_events) == 3, f"Expected 3 epochs, got {len(epoch_events)}"
    assert training_complete, "Training did not complete"
    assert query_ready, "Query not ready"
    print(f"  ✓ Received {len(epoch_events)} epoch events")
    
    # Test 3: Send CMD_ANNOTATIONS
    print("\n[Test 3.3] Process annotations...")
    
    annotations_event = Event(
        EventType.CMD_ANNOTATIONS,
        payload={
            "annotations": [
                {"image_id": 0, "label": 0},
                {"image_id": 1, "label": 1},
            ]
        },
        source="test"
    )
    parent_conn.send(annotations_event.to_dict())
    
    cycle_complete = False
    timeout = time.time() + 10
    
    while time.time() < timeout:
        if parent_conn.poll(timeout=1):
            event_dict = parent_conn.recv()
            event = Event.from_dict(event_dict)
            
            if event.type == EventType.CYCLE_COMPLETE:
                cycle_complete = True
                print(f"  ✓ CYCLE_COMPLETE: {event.payload}")
                break
    
    assert cycle_complete, "Cycle did not complete after annotations"
    
    # Test 4: Graceful shutdown
    print("\n[Test 3.4] Graceful shutdown...")
    
    shutdown_event = Event(EventType.CMD_SHUTDOWN, source="test")
    parent_conn.send(shutdown_event.to_dict())
    
    process.join(timeout=5)
    
    if process.is_alive():
        print("  ✗ Process did not terminate, forcing...")
        process.terminate()
        process.join(timeout=2)
    else:
        print(f"  ✓ Process terminated gracefully (exit code: {process.exitcode})")
    
    parent_conn.close()
    
    print("\n✅ PHASE 3 PASSED: Service layer working correctly")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_integration():
    """Test integration between layers."""
    print("\n" + "─" * 70)
    print("INTEGRATION TEST: WorldState + Database + Events")
    print("─" * 70)
    
    from model import WorldState, DatabaseManager, ExperimentPhase, ExperimentConfig, EpochMetrics
    from controller.events import Event, EventType
    
    # Simulate the flow: Initialize → Train → Update State → Persist
    
    print("\n[Integration] Simulating full flow...")
    
    # 1. Create components
    ws = WorldState()
    db = DatabaseManager(":memory:")
    
    # 2. Initialize experiment
    config = ExperimentConfig(
        model_name="resnet18",
        num_cycles=3,
        epochs_per_cycle=5
    )
    
    ws.initialize("int_exp_001", "Integration Test", config)
    db.insert_experiment("int_exp_001", "Integration Test", config.to_dict())
    
    print(f"  ✓ Initialized experiment: {ws.experiment_id}")
    
    # 3. Simulate training cycle
    ws.set_phase(ExperimentPhase.TRAINING)
    db.update_experiment_phase("int_exp_001", "TRAINING")
    
    ws.start_cycle(1, labeled_count=40, unlabeled_count=360)
    
    for epoch in range(1, 6):
        # Simulate event from service
        event = Event(
            EventType.EPOCH_COMPLETE,
            payload={
                "epoch": epoch,
                "train_loss": 1.0 - (epoch * 0.1),
                "train_accuracy": 0.5 + (epoch * 0.05),
                "val_loss": 1.1 - (epoch * 0.08),
                "val_accuracy": 0.45 + (epoch * 0.06),
                "learning_rate": 0.001
            }
        )
        
        # Update WorldState (as Controller would)
        metrics = EpochMetrics(
            epoch=event.payload["epoch"],
            train_loss=event.payload["train_loss"],
            train_accuracy=event.payload["train_accuracy"],
            val_loss=event.payload["val_loss"],
            val_accuracy=event.payload["val_accuracy"],
            learning_rate=event.payload["learning_rate"]
        )
        ws.update_epoch(metrics)
        
        # Persist to database
        db.insert_epoch_metrics("int_exp_001", cycle=1, metrics=metrics)
        
        print(f"    Epoch {epoch}: train_loss={metrics.train_loss:.3f}, val_acc={metrics.val_accuracy:.3f}")
    
    # 4. Verify state
    assert ws.current_epoch == 5
    assert len(ws.epoch_history) == 5
    assert ws.pending_updates == True
    
    db_metrics = db.get_epoch_metrics("int_exp_001", cycle=1)
    assert len(db_metrics) == 5
    
    print(f"\n  ✓ WorldState: {len(ws.epoch_history)} epochs in memory")
    print(f"  ✓ Database: {len(db_metrics)} epochs persisted")
    print(f"  ✓ Progress: {ws.progress_percentage:.1f}%")
    
    print("\n✅ INTEGRATION TEST PASSED")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run all tests."""
    results = []
    
    # Phase 1
    try:
        results.append(("Phase 1: Events", test_phase1_events()))
    except Exception as e:
        print(f"\n❌ Phase 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Phase 1: Events", False))
    
    # Phase 2
    try:
        results.append(("Phase 2.1: Schemas", test_phase2_schemas()))
    except Exception as e:
        print(f"\n❌ Phase 2.1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Phase 2.1: Schemas", False))
    
    try:
        results.append(("Phase 2.2: WorldState", test_phase2_worldstate()))
    except Exception as e:
        print(f"\n❌ Phase 2.2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Phase 2.2: WorldState", False))
    
    try:
        results.append(("Phase 2.3: Database", test_phase2_database()))
    except Exception as e:
        print(f"\n❌ Phase 2.3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Phase 2.3: Database", False))
    
    # Phase 3
    try:
        results.append(("Phase 3: Service", test_phase3_service()))
    except Exception as e:
        print(f"\n❌ Phase 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Phase 3: Service", False))
    
    # Integration
    try:
        results.append(("Integration", test_integration()))
    except Exception as e:
        print(f"\n❌ Integration FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Integration", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
