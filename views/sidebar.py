"""
Sidebar configuration controls for Active Learning Framework.

This module provides the sidebar UI for configuring experiments:
- Model selection (curated families)
- Strategy selection
- Training hyperparameters
- Active learning settings
- Experiment controls (Start/Stop)

The sidebar collects user configuration and dispatches commands to the Controller.
"""

import streamlit as st
from typing import Dict, Any, Optional
import logging
from controller import Controller, AppState
from models import get_model_families, get_model_card

logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 11.1: Model selection dropdown (curated families)
# ============================================================================

def render_model_selection() -> str:
    """
    Render model selection dropdown with curated families.
    
    Returns:
        Selected model name
    """
    st.sidebar.markdown("### 🤖 Model Configuration")
    
    # Get curated model families
    model_families = get_model_families()
    
    # Flatten families into a list with group labels
    model_options = []
    model_display_names = []
    
    for family_name, models in model_families.items():
        for model in models:
            model_options.append(model)
            model_display_names.append(f"{family_name}: {model}")
    
    # Create selectbox with grouped display
    selected_idx = st.sidebar.selectbox(
        "Model Architecture",
        range(len(model_options)),
        format_func=lambda i: model_display_names[i],
        help="Select a pretrained model architecture. ResNet-50 is recommended for most tasks."
    )
    
    selected_model = model_options[selected_idx]
    
    # Show model info card
    with st.sidebar.expander("📊 Model Info"):
        model_info = get_model_card(selected_model)
        if "error" in model_info:
            st.warning(f"⚠️ Could not load model info: {model_info['error']}")
        else:
            st.write(f"**Parameters:** {model_info['parameters_human']}")
            st.write(f"**Pretrained:** {'✅ Yes' if model_info['has_pretrained'] else '❌ No'}")
    
    return selected_model


# ============================================================================
# SUBTASK 11.2: Strategy dropdown
# ============================================================================

def render_strategy_selection() -> str:
    """
    Render sampling strategy dropdown.
    
    Returns:
        Selected strategy name
    """
    st.sidebar.markdown("### 🎯 Sampling Strategy")
    
    strategies = {
        "entropy": "Entropy (Recommended)",
        "margin": "Margin Sampling",
        "least_confidence": "Least Confidence",
        "random": "Random (Baseline)"
    }
    
    strategy_descriptions = {
        "entropy": "Selects samples with highest prediction entropy (uncertainty spread across classes)",
        "margin": "Selects samples with smallest margin between top-2 predictions",
        "least_confidence": "Selects samples with lowest confidence in top prediction",
        "random": "Random selection (baseline for comparison)"
    }
    
    selected_strategy = st.sidebar.selectbox(
        "Strategy",
        list(strategies.keys()),
        format_func=lambda x: strategies[x],
        help="Active learning sampling strategy for selecting uncertain samples"
    )
    
    # Show strategy description
    st.sidebar.caption(strategy_descriptions[selected_strategy])
    
    return selected_strategy


# ============================================================================
# SUBTASK 11.3: Training hyperparameters
# ============================================================================

def render_training_hyperparameters() -> Dict[str, Any]:
    """
    Render training hyperparameter controls.
    
    Returns:
        Dictionary with training hyperparameters
    """
    st.sidebar.markdown("### ⚙️ Training Settings")
    
    epochs = st.sidebar.slider(
        "Epochs per Cycle",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of training epochs per active learning cycle"
    )
    
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[8, 16, 32, 64, 128],
        value=32,
        help="Training batch size (larger = faster but more memory)"
    )
    
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        value=1e-4,
        format_func=lambda x: f"{x:.0e}",
        help="Optimizer learning rate"
    )
    
    # Advanced settings in expander
    with st.sidebar.expander("🔧 Advanced Settings"):
        weight_decay = st.number_input(
            "Weight Decay",
            min_value=0.0,
            max_value=1e-2,
            value=1e-4,
            format="%.0e",
            help="L2 regularization strength"
        )
        
        optimizer = st.selectbox(
            "Optimizer",
            ["adamw", "adam", "sgd"],
            help="Optimization algorithm"
        )
        
        early_stopping = st.number_input(
            "Early Stopping Patience",
            min_value=1,
            max_value=10,
            value=3,
            help="Stop training if no improvement for N epochs"
        )
    
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optimizer": optimizer,
        "early_stopping_patience": early_stopping
    }


# ============================================================================
# SUBTASK 11.4: AL settings
# ============================================================================

def render_al_settings() -> Dict[str, Any]:
    """
    Render active learning settings controls.
    
    Returns:
        Dictionary with AL settings
    """
    st.sidebar.markdown("### 🔄 Active Learning Settings")
    
    num_cycles = st.sidebar.slider(
        "Number of Cycles",
        min_value=1,
        max_value=20,
        value=10,
        help="Total number of active learning cycles to run"
    )
    
    query_size = st.sidebar.slider(
        "Query Size",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of samples to query per cycle"
    )
    
    # Auto-annotate toggle
    auto_annotate = st.sidebar.checkbox(
        "Auto-Annotate (Ground Truth)",
        value=True,
        help="If checked, images are auto-labeled with ground truth (simulation). "
             "Uncheck for manual human-in-the-loop annotation."
    )

    # Advanced AL settings
    with st.sidebar.expander("Advanced AL Settings"):
        initial_pool_size = st.number_input(
            "Initial Pool Size",
            min_value=10,
            max_value=500,
            value=100,
            help="Number of labeled samples to start with"
        )

        reset_mode = st.selectbox(
            "Model Reset Mode",
            ["pretrained", "head_only", "none"],
            help="How to reset model weights between cycles:\n"
                 "- pretrained: Reset to pretrained weights\n"
                 "- head_only: Reset only classification head\n"
                 "- none: Continue training from previous cycle"
        )

    return {
        "num_cycles": num_cycles,
        "batch_size_al": query_size,
        "initial_pool_size": initial_pool_size,
        "reset_mode": reset_mode,
        "auto_annotate": auto_annotate,
    }


# ============================================================================
# SUBTASK 11.5 & 11.6: Start/Stop buttons
# ============================================================================

def render_experiment_controls(controller: Controller) -> None:
    """
    Render experiment control buttons (Start/Stop).
    
    Args:
        controller: Controller instance for dispatching commands
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎮 Experiment Controls")
    
    current_state = controller.get_state()
    
    # Show current state
    state_emoji = {
        AppState.IDLE: "🏁",
        AppState.TRAINING: "🔄",
        AppState.QUERYING: "🔍",
        AppState.ANNOTATING: "🏷️",
        AppState.ERROR: "❌"
    }
    
    st.sidebar.info(
        f"{state_emoji.get(current_state, '❓')} **Status:** {current_state.value.upper()}"
    )
    
    # Start button (only enabled in IDLE state)
    start_disabled = current_state != AppState.IDLE
    
    if st.sidebar.button(
        "Start Experiment",
        disabled=start_disabled,
        help="Start the first active learning cycle" if not start_disabled else "Cannot start from current state"
    ):
        try:
            # Apply config overrides before starting
            config_overrides = st.session_state.get('config_overrides', {})
            if config_overrides:
                from config import load_config
                new_config = load_config(overrides=config_overrides)
                st.session_state.config = new_config
                controller.config = new_config
                controller.experiment_config = new_config.to_dict()
                controller.total_cycles = new_config.active_learning.num_cycles
                logger.info("Applied config overrides from sidebar")

                # Respawn worker with new config
                spawn_fn = st.session_state.get('spawn_worker')
                if spawn_fn:
                    spawn_fn(new_config)
                    logger.info("Worker respawned with new config")

            # Reset controller state for fresh experiment
            controller.current_cycle = 0
            controller.current_epoch = 0
            controller.metrics_history = []
            controller.epoch_metrics = []
            controller.queried_images = []
            controller.last_error = None
            controller.unlabeled_pool_size = 0

            # Dispatch first cycle
            controller.dispatch_run_cycle(cycle_num=1)
            st.sidebar.success("Experiment started!")
            logger.info("User started experiment (Cycle 1)")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Failed to start: {e}")
            logger.error(f"Failed to start experiment: {e}")
    
    # Stop button (enabled when busy)
    stop_disabled = not controller.is_busy()
    
    if st.sidebar.button(
        "⏹️ Stop",
        disabled=stop_disabled,
        help="Stop the current operation" if not stop_disabled else "Nothing to stop"
    ):
        try:
            controller.dispatch_stop()
            st.sidebar.warning("⚠️ Stop requested...")
            logger.info("User requested stop")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Failed to stop: {e}")
            logger.error(f"Failed to stop: {e}")
    
    # Show progress if experiment is running
    if current_state != AppState.IDLE:
        progress = controller.get_progress()
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Progress")
        st.sidebar.write(f"**Cycle:** {progress['current_cycle']} / {progress['total_cycles']}")
        st.sidebar.write(f"**Completed Cycles:** {progress['cycles_completed']}")


# ============================================================================
# MAIN SIDEBAR RENDER FUNCTION
# ============================================================================

def render_sidebar(controller: Controller) -> Dict[str, Any]:
    """
    Main sidebar render function.
    
    Renders all sidebar components and collects configuration.
    
    Args:
        controller: Controller instance
        
    Returns:
        Dictionary with all configuration overrides
    """
    st.sidebar.title("🎯 Active Learning Framework")
    st.sidebar.markdown("Configure your experiment below")
    
    # Render all sections
    model_name = render_model_selection()
    strategy = render_strategy_selection()
    training_params = render_training_hyperparameters()
    al_params = render_al_settings()
    
    # Render controls
    render_experiment_controls(controller)
    
    # Build config overrides dictionary
    config_overrides = {
        "model.name": model_name,
        "active_learning.sampling_strategy": strategy,
        "training.epochs": training_params["epochs"],
        "training.batch_size": training_params["batch_size"],
        "training.learning_rate": training_params["learning_rate"],
        "training.weight_decay": training_params["weight_decay"],
        "training.optimizer": training_params["optimizer"],
        "training.early_stopping_patience": training_params["early_stopping_patience"],
        "active_learning.num_cycles": al_params["num_cycles"],
        "active_learning.batch_size_al": al_params["batch_size_al"],
        "active_learning.initial_pool_size": al_params["initial_pool_size"],
        "active_learning.reset_mode": al_params["reset_mode"],
        "active_learning.auto_annotate": al_params["auto_annotate"],
    }
    
    # Store in session state for access by other views
    st.session_state['config_overrides'] = config_overrides
    
    return config_overrides
