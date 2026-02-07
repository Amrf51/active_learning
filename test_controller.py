"""
Simple test to verify controller.py implementation.
"""

import multiprocessing as mp
from controller import Controller, AppState
from protocol import create_event_dict


def test_controller_initialization():
    """Test that controller initializes correctly."""
    mp_context = mp.get_context('spawn')
    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()
    events = create_event_dict(mp_context)
    
    controller = Controller(task_queue, result_queue, events)
    
    assert controller.get_state() == AppState.IDLE
    assert controller.current_cycle == 0
    assert controller.current_epoch == 0
    assert len(controller.metrics_history) == 0
    print("✓ Controller initialization test passed")


def test_state_transitions():
    """Test state transition validation."""
    mp_context = mp.get_context('spawn')
    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()
    events = create_event_dict(mp_context)
    
    controller = Controller(task_queue, result_queue, events)
    
    # Test valid transition: IDLE -> INITIALIZING
    assert controller._can_transition_to(AppState.INITIALIZING)
    
    # Test invalid transition: IDLE -> TRAINING
    assert not controller._can_transition_to(AppState.TRAINING)
    
    # Test invalid transition: IDLE -> QUERYING
    assert not controller._can_transition_to(AppState.QUERYING)
    
    print("✓ State transition validation test passed")


def test_dispatch_methods():
    """Test that dispatch methods work without errors."""
    mp_context = mp.get_context('spawn')
    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()
    events = create_event_dict(mp_context)
    
    controller = Controller(task_queue, result_queue, events)
    
    # Test dispatch_stop (should work from any state)
    controller.dispatch_stop()
    assert events['stop_requested'].is_set()
    controller.clear_stop()
    assert not events['stop_requested'].is_set()
    
    print("✓ Dispatch methods test passed")


def test_state_persistence():
    """Test save_state and load_state."""
    mp_context = mp.get_context('spawn')
    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()
    events = create_event_dict(mp_context)
    
    controller = Controller(task_queue, result_queue, events, state_file="test_state.json")
    
    # Set some state
    controller.current_cycle = 5
    controller.total_cycles = 10
    controller.metrics_history = [{"cycle": 1, "accuracy": 0.8}]
    
    # Save state
    controller.save_state()
    
    # Create new controller and load state
    controller2 = Controller(task_queue, result_queue, events, state_file="test_state.json")
    loaded = controller2.load_state()
    
    assert loaded
    assert controller2.current_cycle == 5
    assert controller2.total_cycles == 10
    assert len(controller2.metrics_history) == 1
    
    # Cleanup
    import os
    os.remove("test_state.json")
    
    print("✓ State persistence test passed")


def test_utility_methods():
    """Test utility methods."""
    mp_context = mp.get_context('spawn')
    task_queue = mp_context.Queue()
    result_queue = mp_context.Queue()
    events = create_event_dict(mp_context)
    
    controller = Controller(task_queue, result_queue, events)
    
    # Test get_progress
    progress = controller.get_progress()
    assert progress["state"] == "idle"
    assert progress["current_cycle"] == 0
    
    # Test is_busy
    assert not controller.is_busy()
    
    # Test get_last_error
    assert controller.get_last_error() is None
    
    print("✓ Utility methods test passed")


if __name__ == "__main__":
    print("Running controller.py tests...\n")
    
    test_controller_initialization()
    test_state_transitions()
    test_dispatch_methods()
    test_state_persistence()
    test_utility_methods()
    
    print("\n✅ All tests passed!")
