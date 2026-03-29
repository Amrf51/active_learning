"""
View router for the threaded Active Learning UI.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import streamlit as st

from controller import Controller
from experiment_state import AppState


STALE_HEARTBEAT_SECONDS = 60.0


def render(snap: Optional[Dict] = None) -> None:
    """Route to the correct state view using one atomic snapshot."""
    controller: Optional[Controller] = st.session_state.get("controller")
    if controller is None:
        st.error("Controller not initialized. Please restart the application.")
        return

    if snap is None:
        snap = controller.get_snapshot()
    current_state = snap["app_state"]

    if current_state in {AppState.INITIALIZING, AppState.QUERYING} and snap["thread_alive"]:
        age = time.time() - float(snap["heartbeat_ts"])
        if age > STALE_HEARTBEAT_SECONDS:
            st.warning(
                f"Backend thread may be stalled (last heartbeat {age:.1f}s ago). "
                "Try Stop and restart the experiment."
            )

    main_tab, results_tab, compare_tab, explorer_tab = st.tabs(
        ["Main", "Results", "Compare Runs", "Dataset Explorer"]
    )
    with main_tab:
        _render_state_view(controller, snap, current_state)
    with results_tab:
        from views.results import render_results_view

        render_results_view(controller, snap)
    with compare_tab:
        from views.results import render_comparison_view

        render_comparison_view(controller, snap)
    with explorer_tab:
        from views.explorer import render_explorer_view

        render_explorer_view(controller, snap)


def _render_state_view(controller: Controller, snap: dict, current_state: AppState) -> None:
    """Render state-specific main page content."""
    if current_state == AppState.IDLE:
        render_idle_view(controller, snap)
    elif current_state == AppState.INITIALIZING:
        render_initializing_view(snap)
    elif current_state == AppState.TRAINING:
        render_training_view(controller, snap)
    elif current_state == AppState.QUERYING:
        render_querying_view(snap)
    elif current_state == AppState.ANNOTATING:
        render_annotating_view(controller, snap)
    elif current_state == AppState.WAITING_STEP:
        render_waiting_step_view(controller, snap)
    elif current_state == AppState.STOPPING:
        render_stopping_view(snap)
    elif current_state == AppState.ERROR:
        render_error_view(controller, snap)
    elif current_state == AppState.FINISHED:
        render_finished_view(controller, snap)
    else:
        st.error(f"Unknown application state: {current_state}")


def render_idle_view(controller: Controller, snap: dict) -> None:
    _ = controller
    st.info("IDLE - Ready to start a new experiment.")
    st.write("Configure your experiment in the sidebar and click Start Experiment.")
    with st.expander("Controller Status"):
        st.write(f"Current Cycle: {snap['current_cycle']}")
        st.write(f"Completed Cycles: {len(snap['metrics_history'])}")
        st.write(f"Queried Images: {len(snap['queried_images'])}")


def render_finished_view(controller: Controller, snap: dict) -> None:
    _ = controller
    st.success("Experiment finished.")
    st.caption(snap.get("progress_detail", ""))


def render_initializing_view(snap: dict) -> None:
    st.info("Initializing backend components...")
    st.caption(snap.get("progress_detail", "Preparing run..."))
    with st.spinner("Building model and dataloaders..."):
        st.progress(0.05)


def render_training_view(controller: Controller, snap: dict) -> None:
    from views.training import render_training_view as render_training

    render_training(controller, snap)


def render_querying_view(snap: dict) -> None:
    st.info("QUERYING - Selecting uncertain samples...")
    st.write(f"Cycle: {snap['current_cycle']}/{snap['total_cycles']}")
    with st.spinner(snap.get("progress_detail", "Running uncertainty query...")):
        st.caption("This may take a few seconds")


def render_annotating_view(controller: Controller, snap: dict) -> None:
    from views.gallery import render_gallery_view

    render_gallery_view(controller, snap)


def render_waiting_step_view(controller: Controller, snap: dict) -> None:
    _ = controller
    st.info("Paused in step mode.")
    st.caption(snap.get("progress_detail", "Click Next Step in the sidebar to continue."))


def render_stopping_view(snap: dict) -> None:
    st.warning("Stopping backend thread...")
    st.caption(snap.get("progress_detail", "Stop requested..."))


def render_error_view(controller: Controller, snap: dict) -> None:
    st.error("ERROR - An error occurred")
    error_info = snap.get("last_error")
    if error_info:
        st.write(f"Error Type: {error_info.get('type', 'Unknown')}")
        st.write(f"Message: {error_info.get('message', 'No details available')}")
        if error_info.get("traceback"):
            with st.expander("Stack Trace"):
                st.code(error_info["traceback"], language="python")
    else:
        st.write("No error details available")

    if st.button("Reset to IDLE"):
        controller.reset_to_idle(clear_history=False)
        st.rerun()
