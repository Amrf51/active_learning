"""
app.py - Streamlit entrypoint for the threaded Active Learning UI.
"""

import atexit
import logging
from typing import List

import streamlit as st

from controller import Controller
from events import Event, EventType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize config + controller once per Streamlit session."""
    if "initialized" in st.session_state:
        return

    from config import load_config

    config = load_config()
    controller = Controller(config)

    st.session_state.config = config
    st.session_state.controller = controller
    st.session_state.experiment_history = []
    st.session_state.last_event_version = -1
    st.session_state.current_cycle_id = None
    st.session_state.annotations = {}
    st.session_state.initialized = True


def shutdown_handler() -> None:
    """Best-effort cleanup on process exit."""
    try:
        controller = st.session_state.get("controller")
        if controller is not None:
            controller.stop_experiment(join_timeout=2.0)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Error during shutdown")


atexit.register(shutdown_handler)


def _handle_ui_effects(events: List[Event]) -> None:
    """Apply local UI/session cleanup triggered by lifecycle events."""
    for event in events:
        if event.type == EventType.CYCLE_STARTED:
            st.session_state.annotations = {}
            st.session_state.pop("last_annotation_feedback", None)
            st.session_state["current_cycle_id"] = (event.run_id, event.cycle)
        elif event.type == EventType.NEW_IMAGES:
            st.session_state.annotations = {}
        elif event.type in (EventType.RUN_FINISHED, EventType.RUN_ERROR, EventType.RUN_STOPPED):
            st.session_state.annotations = {}
            st.session_state.pop("last_annotation_feedback", None)


@st.fragment(run_every="0.5s")
def live_update_fragment() -> None:
    """Render main routed view on a fast fragment refresh cadence."""
    from views.router import render

    controller = st.session_state.controller
    last_ver = st.session_state.get("last_event_version", -1)

    events, new_ver = controller.process_inbox(last_ver)
    if events:
        _handle_ui_effects(events)
    st.session_state.last_event_version = new_ver

    render()


def main() -> None:
    st.set_page_config(
        page_title="Active Learning Framework",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    controller = st.session_state.controller

    from views.sidebar import render_sidebar

    render_sidebar(controller)

    st.title("Active Learning Framework")
    st.caption("Visual and Interactive Active Learning for Vehicle Image Classification")
    st.divider()

    live_update_fragment()

    st.divider()
    st.caption("Active Learning Framework v1.0 | Bachelor Thesis Project")


if __name__ == "__main__":
    main()
