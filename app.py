"""
app.py — Entry point for the Active Learning Framework with Streamlit UI.

This module initializes the MVC architecture with multiprocessing:
- Sets up multiprocessing context (spawn method for CUDA compatibility)
- Creates queues and events for inter-process communication
- Initializes the Controller (state machine)
- Spawns the worker process (Model)
- Launches the Streamlit UI (View)

Usage:
    streamlit run app.py
"""

import multiprocessing as mp
import streamlit as st
import atexit
import logging
from pathlib import Path

# Import protocol for event creation
from protocol import create_event_dict

# Import Controller and Worker (Phase 4)
from controller import Controller
from worker import worker_main

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# MULTIPROCESSING INITIALIZATION
# ============================================================================

def init_multiprocessing():
    """
    Initialize multiprocessing context with spawn method for CUDA compatibility.
    
    The 'spawn' method is required when using CUDA/GPU in worker processes
    to avoid issues with forked processes and CUDA context.
    
    Returns:
        Multiprocessing context object
    """
    # Use spawn method for CUDA compatibility
    mp_context = mp.get_context('spawn')
    logger.info("Multiprocessing context initialized with 'spawn' method")
    return mp_context


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """
    Initialize Streamlit session state with queues, events, and controller.
    
    This function is called once when the app first loads. It sets up:
    - Multiprocessing context
    - Task and result queues for communication
    - Event dictionary for process synchronization
    - Controller (placeholder for Phase 4)
    - Worker process (placeholder for Phase 4)
    """
    # Check if already initialized
    if 'initialized' in st.session_state:
        return
    
    logger.info("Initializing session state...")
    
    # Subtask 6.1: Initialize multiprocessing context
    mp_context = init_multiprocessing()
    st.session_state.mp_context = mp_context
    
    # Subtask 6.2: Create mp.Event dict for all events using protocol helper
    events = create_event_dict(mp_context)
    st.session_state.events = events
    logger.info(f"Created {len(events)} multiprocessing events")
    
    # Subtask 6.3: Create task_queue and result_queue
    task_queue = mp_context.Queue(maxsize=10)
    result_queue = mp_context.Queue(maxsize=100)
    st.session_state.task_queue = task_queue
    st.session_state.result_queue = result_queue
    logger.info("Created task_queue (maxsize=10) and result_queue (maxsize=100)")
    
    # Load config at startup (single source of truth)
    from config import load_config
    config = load_config()  # Loads default.yaml, validates, resolves device
    config_dict = config.to_dict()

    # Store config in session state for UI access
    st.session_state.config = config

    # Subtask 9.1: Initialize Controller with queues and events
    controller = Controller(task_queue, result_queue, events, config)
    st.session_state.controller = controller
    logger.info("Controller initialized")
    
    # Start worker process with config passed at spawn time
    worker = mp_context.Process(
        target=worker_main,
        args=(task_queue, result_queue, events, config_dict),
        daemon=True,
        name="ALWorker"
    )
    worker.start()
    st.session_state.worker = worker
    logger.info(f"Worker process started (PID: {worker.pid})")

    # Wait for worker to initialize AL components (blocking with timeout)
    if not events['worker_initialized'].wait(timeout=60):
        logger.error("Worker failed to initialize within 60 seconds")
        st.error("❌ Worker initialization timeout. Check logs for details.")
        raise RuntimeError("Worker initialization timeout")

    logger.info("Worker initialized and ready")
    
    # Subtask 6.6: Store queues and events in st.session_state (already done above)
    # Additional session state for application state
    st.session_state.app_state = "IDLE"  # IDLE, TRAINING, QUERYING, ANNOTATING, ERROR (no INITIALIZING - worker ready at startup)
    st.session_state.current_cycle = 0
    st.session_state.current_epoch = 0
    st.session_state.metrics_history = []
    st.session_state.queried_images = []
    st.session_state.experiment_config = {}
    st.session_state.experiment_history = []  # For strategy comparison
    
    # Mark as initialized
    st.session_state.initialized = True
    logger.info("Session state initialization complete")


# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

def shutdown_handler():
    """
    Subtask 6.8: Handle graceful shutdown on app close.
    
    This function is called when the app is closing to ensure:
    - Worker process is terminated cleanly
    - Queues are closed
    - Events are cleared
    - Resources are released
    """
    logger.info("Shutdown handler called")
    
    if 'initialized' not in st.session_state:
        return
    
    try:
        # Signal worker to stop if it exists
        if st.session_state.worker is not None and st.session_state.worker.is_alive():
            logger.info("Terminating worker process...")
            
            # Send shutdown message to worker
            from protocol import build_shutdown_message
            try:
                st.session_state.task_queue.put(build_shutdown_message(), timeout=1)
                # Wait for shutdown to complete
                st.session_state.events['shutdown_complete'].wait(timeout=5)
            except:
                pass  # Timeout or queue full, proceed with termination
            
            # Terminate worker process
            if st.session_state.worker.is_alive():
                st.session_state.worker.terminate()
                st.session_state.worker.join(timeout=2)
                logger.info("Worker process terminated")
        
        # Close queues
        if hasattr(st.session_state, 'task_queue'):
            st.session_state.task_queue.close()
            logger.info("Task queue closed")
        
        if hasattr(st.session_state, 'result_queue'):
            st.session_state.result_queue.close()
            logger.info("Result queue closed")
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Register shutdown handler
atexit.register(shutdown_handler)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application entry point.
    
    Subtask 6.7: Basic Streamlit page config and layout structure.
    """
    # Streamlit page configuration
    st.set_page_config(
        page_title="Active Learning Framework",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state (runs once)
    init_session_state()
    
    # Main layout structure
    st.title("🎯 Active Learning Framework")
    st.caption("Visual and Interactive Active Learning for Vehicle Image Classification")
    
    # Sidebar placeholder (to be implemented in Phase 5)
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Configuration controls will be added in Phase 5")
        
        # Debug info
        with st.expander("🔧 Debug Info"):
            st.write(f"App State: {st.session_state.app_state}")
            st.write(f"Current Cycle: {st.session_state.current_cycle}")
            st.write(f"Controller: {'✅ Ready' if st.session_state.controller else '❌ Not initialized'}")
            st.write(f"Worker: {'✅ Running (PID: ' + str(st.session_state.worker.pid) + ')' if st.session_state.worker and st.session_state.worker.is_alive() else '❌ Not running'}")
            
            # Show queue sizes
            if st.session_state.controller:
                st.write(f"Task Queue: {st.session_state.task_queue.qsize()} messages")
                st.write(f"Result Queue: {st.session_state.result_queue.qsize()} messages")
    
    # Main content area
    st.divider()
    
    # Subtask 9.4: Call view.render() for UI
    # TODO: Phase 5 - Implement views and replace placeholder
    # from views.router import render
    # render(st.session_state.controller, st.session_state.app_state)
    
    # Placeholder for view rendering (to be implemented in Phase 5)
    st.info("📋 **Status:** MVC Core Components Wired Up")
    st.write("**Progress:**")
    st.markdown("""
    - ✅ Phase 1: Configuration & Protocol Infrastructure (Complete)
    - ✅ Phase 2: Backend Modifications (Complete)
    - ✅ Phase 3: Application Skeleton (Complete)
    - ✅ Phase 4: MVC Core Components (Complete - Task 9)
    - ⏳ Phase 5: Streamlit Views (Next)
    """)
    
    # Show controller status
    if st.session_state.controller:
        st.success("✅ Controller initialized and ready")
        progress_info = st.session_state.controller.get_progress()
        st.json(progress_info)
    
    # TODO: Phase 5 - Replace with actual view rendering
    # from views.router import render
    # render(st.session_state.controller, st.session_state.app_state)
    
    st.divider()
    st.caption("Active Learning Framework v1.0 | Bachelor Thesis Project")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
