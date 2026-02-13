"""
app.py - Streamlit entrypoint for the threaded Active Learning UI.
"""

import atexit
import logging

import streamlit as st

from controller import Controller

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


@st.fragment(run_every="0.5s")
def live_update_fragment() -> None:
    """Render main routed view on a fast fragment refresh cadence."""
    from views.router import render

    render()


def main() -> None:
    st.set_page_config(
        page_title="Active Learning Framework",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    controller = st.session_state.controller

    from views.sidebar import render_sidebar

    render_sidebar(controller)

    st.title("🎯 Active Learning Framework")
    st.caption("Visual and Interactive Active Learning for Vehicle Image Classification")
    st.divider()

    live_update_fragment()

    st.divider()
    st.caption("Active Learning Framework v1.0 | Bachelor Thesis Project")


if __name__ == "__main__":
    main()
