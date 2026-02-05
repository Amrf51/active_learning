"""Property-based tests for multiprocessing infrastructure.

These tests verify the correctness properties of the multiprocessing
architecture using Hypothesis for property-based testing.

Properties tested:
- Property 16: Service Process Lifecycle
- Property 17: Pipe Communication Integrity
- Property 18: State Push After Event

Run with: python -m pytest tests/test_multiprocessing_properties.py -v

Validates: Requirements 3.1, 7.2, 8.2
"""

import pickle
import time
import tempfile
from pathlib import Path
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Dict, Any, Optional

import pytest
from hypothesis import given, strategies as st, settings, assume

from controller.events import Event, EventType
from controller.background_worker import BackgroundWorker
from model.world_state import WorldState, Phase


# =============================================================================
# Test Strategies (Hypothesis generators)
# =============================================================================

@st.composite
def event_type_strategy(draw):
    """Generate random EventType values."""
    return draw(st.sampled_from(list(EventType)))


@st.composite
def simple_payload_strategy(draw):
    """Generate simple picklable payloads for events."""
    # Generate a dictionary with simple types that are always picklable
    return {
        'string_field': draw(st.text(min_size=0, max_size=50, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'S'),
            whitelist_characters=' '
        ))),
        'int_field': draw(st.integers(min_value=-1000, max_value=1000)),
        'float_field': draw(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False)),
        'bool_field': draw(st.booleans()),
        'list_field': draw(st.lists(st.integers(min_value=-100, max_value=100), max_size=10)),
    }


@st.composite
def event_strategy(draw):
    """Generate random Event objects."""
    event_type = draw(event_type_strategy())
    payload = draw(simple_payload_strategy())
    return Event(type=event_type, payload=payload)


@st.composite
def world_state_strategy(draw):
    """Generate random WorldState objects."""
    phase = draw(st.sampled_from(list(Phase)))
    
    return WorldState(
        experiment_id=draw(st.one_of(st.none(), st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=('L', 'N'),
        )))),
        experiment_name=draw(st.one_of(st.none(), st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=('L', 'N', 'P', 'S'),
            whitelist_characters=' '
        )))),
        phase=phase,
        current_cycle=draw(st.integers(min_value=0, max_value=100)),
        total_cycles=draw(st.integers(min_value=0, max_value=100)),
        current_epoch=draw(st.integers(min_value=0, max_value=100)),
        epochs_per_cycle=draw(st.integers(min_value=0, max_value=100)),
        labeled_count=draw(st.integers(min_value=0, max_value=10000)),
        unlabeled_count=draw(st.integers(min_value=0, max_value=10000)),
        class_distribution=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L',))),
            values=st.integers(min_value=0, max_value=1000),
            max_size=5
        )),
        error_message=draw(st.one_of(st.none(), st.text(min_size=1, max_size=100))),
    )


# =============================================================================
# Helper Functions for Testing
# =============================================================================

def simple_echo_service(pipe: Connection, experiments_dir: Path) -> None:
    """A simple service that echoes events back as WorldState updates.
    
    This is used for testing pipe communication without the full
    ActiveLearningService complexity.
    """
    running = True
    state = WorldState()
    
    while running:
        try:
            if pipe.poll(timeout=0.1):
                event = pipe.recv()
                
                if event.type == EventType.SHUTDOWN:
                    running = False
                    continue
                
                # Echo the event by updating state
                state.experiment_name = f"Received: {event.type.value}"
                state.touch()
                pipe.send(state)
                
        except EOFError:
            running = False
        except Exception:
            running = False
    
    pipe.close()


def state_push_service(pipe: Connection, experiments_dir: Path) -> None:
    """A service that pushes state after every event.
    
    Used to test Property 18: State Push After Event.
    """
    running = True
    state = WorldState()
    events_received = 0
    
    while running:
        try:
            if pipe.poll(timeout=0.1):
                event = pipe.recv()
                
                if event.type == EventType.SHUTDOWN:
                    running = False
                    continue
                
                # Update state based on event
                events_received += 1
                state.current_cycle = events_received
                state.experiment_name = f"Event_{events_received}_{event.type.value}"
                state.touch()
                
                # Always push state after processing event
                pipe.send(state)
                
        except EOFError:
            running = False
        except Exception:
            running = False
    
    pipe.close()


# =============================================================================
# Property 16: Service Process Lifecycle
# =============================================================================

class TestServiceProcessLifecycle:
    """
    Feature: mvc-active-learning-dashboard, Property 16: Service Process Lifecycle
    
    *For any* controller initialization, the service process SHALL be alive 
    after start() and not alive after shutdown().
    
    Validates: Requirements 3.1 (background process management)
    """
    
    @settings(max_examples=100, deadline=10000)
    @given(st.integers(min_value=1, max_value=5))
    def test_service_alive_after_start(self, _iteration: int):
        """
        **Validates: Requirements 3.1**
        
        Property: After start(), the service process SHALL be alive.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                # Start the worker with a simple echo service
                worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
                
                # Give process time to start
                time.sleep(0.1)
                
                # Property: Service SHALL be alive after start()
                assert worker.is_alive(), "Service process should be alive after start()"
                assert worker._is_started, "Worker should be marked as started"
                assert worker.process_id is not None, "Process ID should be available"
                
            finally:
                worker.shutdown(timeout=2.0)
    
    @settings(max_examples=100, deadline=10000)
    @given(st.integers(min_value=1, max_value=5))
    def test_service_not_alive_after_shutdown(self, _iteration: int):
        """
        **Validates: Requirements 3.1**
        
        Property: After shutdown(), the service process SHALL NOT be alive.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            # Start the worker
            worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
            time.sleep(0.1)
            
            # Verify it's alive first
            assert worker.is_alive(), "Service should be alive before shutdown"
            
            # Shutdown
            graceful = worker.shutdown(timeout=2.0)
            
            # Property: Service SHALL NOT be alive after shutdown()
            assert not worker.is_alive(), "Service process should not be alive after shutdown()"
            assert not worker._is_started, "Worker should be marked as not started"
            assert worker.process_id is None, "Process ID should be None after shutdown"
    
    @settings(max_examples=50, deadline=15000)
    @given(st.integers(min_value=1, max_value=3))
    def test_multiple_start_shutdown_cycles(self, num_cycles: int):
        """
        **Validates: Requirements 3.1**
        
        Property: Service lifecycle is consistent across multiple start/shutdown cycles.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            for cycle in range(num_cycles):
                # Start
                worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
                time.sleep(0.1)
                
                assert worker.is_alive(), f"Cycle {cycle}: Service should be alive after start"
                
                # Shutdown
                worker.shutdown(timeout=2.0)
                
                assert not worker.is_alive(), f"Cycle {cycle}: Service should not be alive after shutdown"
    
    def test_double_start_is_idempotent(self):
        """
        **Validates: Requirements 3.1**
        
        Property: Calling start() twice should not create duplicate processes.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                # First start
                worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
                time.sleep(0.1)
                first_pid = worker.process_id
                
                # Second start (should be no-op)
                worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
                second_pid = worker.process_id
                
                # Property: Same process should be running
                assert first_pid == second_pid, "Double start should not create new process"
                assert worker.is_alive(), "Service should still be alive"
                
            finally:
                worker.shutdown(timeout=2.0)


# =============================================================================
# Property 17: Pipe Communication Integrity
# =============================================================================

class TestPipeCommunicationIntegrity:
    """
    Feature: mvc-active-learning-dashboard, Property 17: Pipe Communication Integrity
    
    *For any* event sent via pipe, the service SHALL receive an equivalent 
    event (serialization round-trip).
    
    Validates: Requirements 8.2 (event routing)
    """
    
    @settings(max_examples=100, deadline=5000)
    @given(event=event_strategy())
    def test_event_serialization_round_trip(self, event: Event):
        """
        **Validates: Requirements 8.2**
        
        Property: Any event can be serialized and deserialized without data loss.
        """
        # Pickle round-trip
        pickled = pickle.dumps(event)
        unpickled = pickle.loads(pickled)
        
        # Property: Deserialized event SHALL be equivalent
        assert unpickled.type == event.type, "Event type should match after round-trip"
        assert unpickled.payload == event.payload, "Event payload should match after round-trip"
        assert isinstance(unpickled.timestamp, float), "Timestamp should be preserved"
    
    @settings(max_examples=100, deadline=5000)
    @given(state=world_state_strategy())
    def test_world_state_serialization_round_trip(self, state: WorldState):
        """
        **Validates: Requirements 7.2**
        
        Property: Any WorldState can be serialized and deserialized without data loss.
        """
        # Pickle round-trip
        pickled = pickle.dumps(state)
        unpickled = pickle.loads(pickled)
        
        # Property: Deserialized state SHALL be equivalent
        assert unpickled.experiment_id == state.experiment_id
        assert unpickled.experiment_name == state.experiment_name
        assert unpickled.phase == state.phase
        assert unpickled.current_cycle == state.current_cycle
        assert unpickled.total_cycles == state.total_cycles
        assert unpickled.current_epoch == state.current_epoch
        assert unpickled.labeled_count == state.labeled_count
        assert unpickled.unlabeled_count == state.unlabeled_count
        assert unpickled.class_distribution == state.class_distribution
        assert unpickled.error_message == state.error_message
    
    @settings(max_examples=50, deadline=10000)
    @given(event_type=event_type_strategy())
    def test_pipe_event_transmission(self, event_type: EventType):
        """
        **Validates: Requirements 8.2**
        
        Property: Events sent through pipe are received correctly by service.
        """
        # Skip SHUTDOWN as it terminates the service
        assume(event_type != EventType.SHUTDOWN)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
                time.sleep(0.1)
                
                # Send event
                event = Event(type=event_type, payload={'test': 'data'})
                success = worker.send_event(event)
                
                assert success, "Event should be sent successfully"
                
                # Wait for response
                time.sleep(0.2)
                state = worker.poll_state(timeout=1.0)
                
                # Property: Service should have received and processed the event
                assert state is not None, "Should receive state update after event"
                assert event_type.value in state.experiment_name, \
                    f"State should reflect received event type: {event_type.value}"
                
            finally:
                worker.shutdown(timeout=2.0)
    
    @settings(max_examples=50, deadline=15000)
    @given(events=st.lists(event_type_strategy(), min_size=1, max_size=5))
    def test_multiple_events_transmission(self, events: list):
        """
        **Validates: Requirements 8.2**
        
        Property: Multiple events sent in sequence are all received correctly.
        """
        # Filter out SHUTDOWN events
        events = [e for e in events if e != EventType.SHUTDOWN]
        assume(len(events) > 0)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=state_push_service)
                time.sleep(0.1)
                
                # Send all events
                for event_type in events:
                    event = Event(type=event_type)
                    worker.send_event(event)
                    time.sleep(0.05)  # Small delay between events
                
                # Wait for processing
                time.sleep(0.5)
                
                # Drain all states
                state = worker.drain_all_states(timeout=1.0)
                
                # Property: All events should have been processed
                assert state is not None, "Should receive state updates"
                assert state.current_cycle == len(events), \
                    f"Should have processed {len(events)} events, got {state.current_cycle}"
                
            finally:
                worker.shutdown(timeout=2.0)


# =============================================================================
# Property 18: State Push After Event
# =============================================================================

class TestStatePushAfterEvent:
    """
    Feature: mvc-active-learning-dashboard, Property 18: State Push After Event
    
    *For any* event processed by the service, a WorldState update SHALL be 
    pushed to the controller via pipe.
    
    Validates: Requirements 7.2 (state synchronization)
    """
    
    @settings(max_examples=100, deadline=10000)
    @given(event_type=event_type_strategy())
    def test_state_pushed_after_single_event(self, event_type: EventType):
        """
        **Validates: Requirements 7.2, 8.2**
        
        Property: After processing any event, service SHALL push WorldState.
        """
        # Skip SHUTDOWN as it terminates without pushing state
        assume(event_type != EventType.SHUTDOWN)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=state_push_service)
                time.sleep(0.1)
                
                # Send event
                event = Event(type=event_type)
                worker.send_event(event)
                
                # Wait for state push
                time.sleep(0.3)
                state = worker.poll_state(timeout=1.0)
                
                # Property: State SHALL be pushed after event
                assert state is not None, \
                    f"WorldState should be pushed after {event_type.value} event"
                assert isinstance(state, WorldState), \
                    "Pushed data should be WorldState instance"
                assert state.current_cycle >= 1, \
                    "State should reflect event was processed"
                
            finally:
                worker.shutdown(timeout=2.0)
    
    @settings(max_examples=50, deadline=15000)
    @given(num_events=st.integers(min_value=1, max_value=10))
    def test_state_pushed_after_each_event(self, num_events: int):
        """
        **Validates: Requirements 7.2**
        
        Property: Each event processed results in a state push.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=state_push_service)
                time.sleep(0.1)
                
                states_received = []
                
                # Send events one by one and collect states
                for i in range(num_events):
                    event = Event(type=EventType.PAUSE)  # Use PAUSE as it's safe
                    worker.send_event(event)
                    
                    # Wait for state
                    time.sleep(0.2)
                    state = worker.poll_state(timeout=0.5)
                    
                    if state is not None:
                        states_received.append(state)
                
                # Property: Should receive state for each event
                assert len(states_received) == num_events, \
                    f"Should receive {num_events} states, got {len(states_received)}"
                
                # Property: States should show progression
                for i, state in enumerate(states_received):
                    assert state.current_cycle == i + 1, \
                        f"State {i} should have cycle={i+1}, got {state.current_cycle}"
                
            finally:
                worker.shutdown(timeout=2.0)
    
    @settings(max_examples=50, deadline=10000)
    @given(event_type=event_type_strategy())
    def test_state_timestamp_updated_on_push(self, event_type: EventType):
        """
        **Validates: Requirements 7.2**
        
        Property: Pushed state SHALL have updated timestamp.
        """
        assume(event_type != EventType.SHUTDOWN)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=state_push_service)
                time.sleep(0.1)
                
                before_send = time.time()
                
                # Send event
                event = Event(type=event_type)
                worker.send_event(event)
                
                # Wait for state
                time.sleep(0.3)
                state = worker.poll_state(timeout=1.0)
                
                after_receive = time.time()
                
                # Property: State timestamp should be within the time window
                assert state is not None, "Should receive state"
                assert state.updated_at >= before_send, \
                    "State timestamp should be after event was sent"
                assert state.updated_at <= after_receive, \
                    "State timestamp should be before we received it"
                
            finally:
                worker.shutdown(timeout=2.0)


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestMultiprocessingIntegration:
    """Additional integration tests for multiprocessing infrastructure."""
    
    def test_graceful_shutdown_sends_shutdown_event(self):
        """Test that shutdown() sends SHUTDOWN event to service."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            worker.start(Path(tmp_dir), service_entry_point=simple_echo_service)
            time.sleep(0.1)
            
            assert worker.is_alive(), "Service should be alive"
            
            # Graceful shutdown
            graceful = worker.shutdown(timeout=2.0)
            
            assert graceful, "Shutdown should be graceful"
            assert not worker.is_alive(), "Service should not be alive after shutdown"
    
    def test_worker_handles_service_crash(self):
        """Test that worker correctly reports when service is not alive."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            # Don't start the worker
            assert not worker.is_alive(), "Worker should not be alive before start"
            
            # Try to send event (should fail gracefully)
            success = worker.send_event(Event(type=EventType.PAUSE))
            assert not success, "Should not be able to send event when not started"
            
            # Poll should return None
            state = worker.poll_state()
            assert state is None, "Should not receive state when not started"
    
    def test_drain_all_states_returns_latest(self):
        """Test that drain_all_states returns the most recent state."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            worker = BackgroundWorker()
            
            try:
                worker.start(Path(tmp_dir), service_entry_point=state_push_service)
                time.sleep(0.1)
                
                # Send multiple events quickly
                for i in range(5):
                    worker.send_event(Event(type=EventType.PAUSE))
                
                # Wait for all to be processed
                time.sleep(0.5)
                
                # Drain all states
                latest = worker.drain_all_states(timeout=1.0)
                
                # Should get the latest state
                assert latest is not None, "Should receive latest state"
                assert latest.current_cycle == 5, "Should have processed all 5 events"
                
                # No more states should be available
                next_state = worker.poll_state(timeout=0.1)
                assert next_state is None, "No more states should be available after drain"
                
            finally:
                worker.shutdown(timeout=2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
