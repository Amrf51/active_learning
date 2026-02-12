"""
View Router for Active Learning Framework.

This module provides the main render() function that dispatches to state-specific
views based on the current application state from the Controller.

State-to-View Mapping:
- IDLE -> render_idle_view()
- TRAINING -> render_training_view()
- QUERYING -> render_querying_view()
- ANNOTATING -> render_annotating_view()
- ERROR -> render_error_view()

The router polls the Controller for updates and ensures the UI reflects the
current state of the AL experiment.
"""

import streamlit as st
from typing import Optional
from controller import AppState, Controller


def render() -> None:
    """
    Main view dispatcher that routes to state-specific views.
    
    This function:
    1. Gets the controller from session state
    2. Determines the current application state
    3. Dispatches to the appropriate view renderer
    
    Note: Polling is now handled by the live_update_fragment() in app.py
    which runs every 2 seconds automatically.
    
    Called from app.py fragment on every auto-rerun.
    """
    # Get controller from session state
    controller: Optional[Controller] = st.session_state.get('controller')
    
    if controller is None:
        st.error("❌ Controller not initialized. Please restart the application.")
        return
    
    # Get current application state
    current_state = controller.get_state()
    
    # Dispatch to state-specific view
    if current_state == AppState.IDLE:
        render_idle_view(controller)
    elif current_state == AppState.TRAINING:
        render_training_view(controller)
    elif current_state == AppState.QUERYING:
        render_querying_view(controller)
    elif current_state == AppState.ANNOTATING:
        render_annotating_view(controller)
    elif current_state == AppState.ERROR:
        render_error_view(controller)
    else:
        st.error(f"❌ Unknown application state: {current_state}")


# ============================================================================
# STATE-SPECIFIC VIEW RENDERERS (Placeholders for Phase 5)
# ============================================================================

def render_idle_view(controller: Controller) -> None:
    """
    Render the IDLE state view.
    
    This view is shown when:
    - App first starts (no experiment running)
    - After an experiment completes
    - After an error is cleared
    
    Displays:
    - Welcome message
    - Instructions to configure and start experiment
    - Previous experiment results (if any)
    
    To be implemented in Phase 5 (Task 11+).
    """
    # Check if we have completed cycles to show results
    if controller.metrics_history:
        # Show tabs: Welcome + Results
        tab1, tab2 = st.tabs(["🏁 Welcome", "📈 Results"])
        
        with tab1:
            st.info("🏁 **IDLE State** — Ready to start a new experiment")
            st.write("Configure your experiment in the sidebar and click **Start Experiment**.")
            
            # Show basic controller info
            with st.expander("📊 Controller Status"):
                st.write(f"Current Cycle: {controller.current_cycle}")
                st.write(f"Total Cycles: {len(controller.metrics_history)}")
                st.write(f"Queried Images: {len(controller.queried_images)}")
        
        with tab2:
            # Import and render results view
            from views.results import render_results_view
            render_results_view(controller)
    else:
        # No results yet, just show welcome
        st.info("🏁 **IDLE State** — Ready to start a new experiment")
        st.write("Configure your experiment in the sidebar and click **Start Experiment**.")
        
        # Show basic controller info
        with st.expander("📊 Controller Status"):
            st.write(f"Current Cycle: {controller.current_cycle}")
            st.write(f"Total Cycles: {len(controller.metrics_history)}")
            st.write(f"Queried Images: {len(controller.queried_images)}")


def render_training_view(controller: Controller) -> None:
    """
    Render the TRAINING state view.
    
    This view is shown when:
    - Model is training on labeled data
    
    Displays:
    - Progress bar for current epoch
    - Cycle progress indicator
    - Live loss/accuracy charts
    - Pool statistics
    - Current metrics
    
    Implemented in Phase 5 (Task 12).
    """
    # Import the training view module
    from views.training import render_training_view as render_training
    
    # Check if we have completed cycles to show results alongside training
    if controller.metrics_history:
        # Show tabs: Training + Results
        tab1, tab2 = st.tabs(["🔄 Training", "📈 Results"])
        
        with tab1:
            # Delegate to the training view
            render_training(controller)
        
        with tab2:
            # Import and render results view
            from views.results import render_results_view
            render_results_view(controller)
    else:
        # No results yet, just show training
        render_training(controller)


def render_querying_view(controller: Controller) -> None:
    """
    Render the QUERYING state view.
    
    This view is shown when:
    - Worker is running inference on unlabeled pool
    - Applying sampling strategy to select uncertain samples
    
    Displays:
    - Progress indicator
    - Strategy being used
    - Query size
    
    This is typically a brief transitional state.
    
    To be implemented in Phase 5 (Task 13).
    """
    st.info("🔍 **QUERYING State** — Selecting uncertain samples...")
    
    st.write(f"**Cycle:** {controller.current_cycle}")
    st.write("Applying sampling strategy to unlabeled pool...")
    
    # Show spinner while querying
    with st.spinner("Running inference and computing uncertainty scores..."):
        st.caption("This typically takes a few seconds")


def render_annotating_view(controller: Controller) -> None:
    """
    Render the ANNOTATING state view.
    
    This view is shown when:
    - Query is complete and samples are ready for annotation
    - Waiting for user to provide labels
    
    Displays:
    - Gallery of Uncertainty (image grid)
    - Per-image: thumbnail, prediction, confidence, uncertainty score
    - Annotation controls (dropdowns or auto-label button)
    - Submit button to send annotations back to worker
    
    Implemented in Phase 5 (Task 13).
    """
    # Import the gallery view module
    from views.gallery import render_gallery_view
    
    # Delegate to the gallery view
    render_gallery_view(controller)


def render_error_view(controller: Controller) -> None:
    """
    Render the ERROR state view.
    
    This view is shown when:
    - Worker encounters an error
    - Invalid state transition attempted
    - Timeout or communication failure
    
    Displays:
    - Error message and details
    - Stack trace (if available)
    - Reset button to return to IDLE
    
    To be implemented in Phase 5 (Task 20 - deferred).
    """
    st.error("❌ **ERROR State** — An error occurred")
    
    # Get error details from controller
    error_info = controller.get_last_error()
    
    if error_info:
        st.write(f"**Error Type:** {error_info.get('type', 'Unknown')}")
        st.write(f"**Message:** {error_info.get('message', 'No details available')}")
        
        # Show traceback if available
        if 'traceback' in error_info:
            with st.expander("📋 Stack Trace"):
                st.code(error_info['traceback'], language='python')
    else:
        st.write("No error details available")
    
    # Reset button
    if st.button("🔄 Reset to IDLE"):
        controller.reset_state()
        st.rerun()
