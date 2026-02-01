"""
Active Learning Control Page - Interactive AL Cycle Management

This page provides the main interface for controlling and monitoring Active Learning experiments.
It implements the Controller-Worker pattern where the dashboard (controller) sends commands
to the worker process and monitors state through polling.

Features:
- Cycle progress display (cycle number, pool sizes)
- Control buttons (Start Cycle, Pause, Stop) using Command Pattern
- Live training visualization with state polling
- Prediction monitor for reference images
- Query visualization and annotation interface

The page NEVER calls training functions directly - it only reads state and writes commands.
The worker process (run_worker.py) is responsible for executing all training operations.
"""

import streamlit as st
from pathlib import Path
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NEW: MVC imports
from views.controller_factory import get_controller, update_session_heartbeat
from controller.events import Event, EventType
from model.schemas import ExperimentPhase, EpochMetrics, QueriedImage

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Active Learning Control - AL Dashboard",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS for AL Control page
st.markdown("""
    <style>
    /* 1. CONTROL PANEL */
    .control-panel {
        background-color: #112240;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* 2. PHASE STATUS INDICATORS */
    .phase-idle { color: #8892b0; font-weight: bold; }
    .phase-training { color: #4ade80; font-weight: bold; }
    .phase-evaluating { color: #fbbf24; font-weight: bold; }
    .phase-querying { color: #f97316; font-weight: bold; }
    .phase-awaiting { color: #a855f7; font-weight: bold; }
    .phase-error { color: #ef4444; font-weight: bold; }
    .phase-completed { color: #10b981; font-weight: bold; }
    
    /* 3. TRAINING CHARTS CONTAINER */
    .training-charts {
        background-color: #0f1419;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #233554;
    }
    
    /* 4. PREDICTION MONITOR */
    .prediction-monitor {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #4a4a6a;
    }
    
    /* 5. QUERY SECTION */
    .query-section {
        background-color: #0f3d3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #20c997;
    }
    
    /* 6. METRIC CARDS */
    .metric-card {
        background-color: #112240;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #233554;
        color: #e6f1ff;
    }
    
    /* 7. EARLY STOPPING INDICATOR */
    .early-stop-marker {
        color: #fbbf24;
        font-weight: bold;
        background-color: #332b00;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
    }
    
    /* 8. WORKER STATUS */
    .worker-active { color: #4ade80; }
    .worker-inactive { color: #ef4444; }
    
    /* 9. IMAGE CARD for queried images */
    .image-card {
        background-color: #1a1a2e;
        padding: 0.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for AL Control page (MVC version)."""
    # NEW: Update session heartbeat
    update_session_heartbeat()
    
    # AL Control specific state
    if "training_active" not in st.session_state:
        st.session_state.training_active = False
    
    if "last_poll_time" not in st.session_state:
        st.session_state.last_poll_time = None
    
    if "poll_interval" not in st.session_state:
        st.session_state.poll_interval = 1.0  # seconds
    
    # Manual annotation state
    if "manual_annotations" not in st.session_state:
        st.session_state.manual_annotations = {}


def check_experiment_selected():
    """Check if an experiment is selected and controller is available (MVC version)."""
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        return status.get('experiment_id') is not None
    except Exception:
        st.warning("⚠️ No experiment selected. Please select an experiment from the sidebar or create one in the Configuration page.")
        return False


def get_phase_display(phase: ExperimentPhase) -> tuple:
    """Get display information for a phase."""
    phase_info = {
        ExperimentPhase.IDLE: ("🔵", "IDLE", "phase-idle"),
        ExperimentPhase.INITIALIZING: ("🟡", "INITIALIZING", "phase-evaluating"),
        ExperimentPhase.PREPARING: ("🟡", "PREPARING", "phase-evaluating"),
        ExperimentPhase.TRAINING: ("🟢", "TRAINING", "phase-training"),
        ExperimentPhase.VALIDATING: ("🟡", "VALIDATING", "phase-evaluating"),
        ExperimentPhase.EVALUATING: ("🟡", "EVALUATING", "phase-evaluating"),
        ExperimentPhase.QUERYING: ("🟠", "QUERYING", "phase-querying"),
        ExperimentPhase.AWAITING_ANNOTATION: ("🟣", "AWAITING ANNOTATION", "phase-awaiting"),
        ExperimentPhase.ANNOTATIONS_SUBMITTED: ("🟣", "PROCESSING ANNOTATIONS", "phase-awaiting"),
        ExperimentPhase.COMPLETED: ("✅", "COMPLETED", "phase-completed"),
        ExperimentPhase.ERROR: ("❌", "ERROR", "phase-error"),
        ExperimentPhase.ABORT: ("🛑", "ABORTED", "phase-error")
    }
    
    return phase_info.get(phase, ("⚪", str(phase), "phase-idle"))


def display_cycle_progress():
    """Display current cycle progress and pool sizes (MVC version)."""
    if not check_experiment_selected():
        return None
    
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("📊 Cycle Progress")
        
        # Progress metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_cycles = status.get('total_cycles', 0)
            current_cycle = status.get('current_cycle', 0)
            if total_cycles > 0:
                progress = current_cycle / total_cycles
                st.metric("Current Cycle", f"{current_cycle}/{total_cycles}")
                st.progress(progress)
            else:
                st.metric("Current Cycle", "Not Started")
        
        with col2:
            labeled_count = status.get('labeled_count', 0)
            st.metric("Labeled Pool", f"{labeled_count:,}")
        
        with col3:
            unlabeled_count = status.get('unlabeled_count', 0)
            st.metric("Unlabeled Pool", f"{unlabeled_count:,}")
        
        with col4:
            # TODO: Get latest test accuracy from controller
            st.metric("Latest Test Acc", "N/A")
        
        # Phase and service status
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            phase = status.get('phase', 'UNKNOWN')
            icon, phase_text, css_class = get_phase_display(ExperimentPhase(phase))
            st.markdown(f"**Current Phase:** {icon} <span class='{css_class}'>{phase_text}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            if ctrl.is_service_alive():
                st.markdown("**Service Status:** <span class='worker-active'>🟢 Active</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown("**Service Status:** <span class='worker-inactive'>🔴 Inactive</span>", 
                           unsafe_allow_html=True)
        
        # Error message if present
        error_message = status.get('error_message')
        if error_message:
            st.error(f"❌ {error_message}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return status
    
    except Exception as e:
        st.error(f"❌ Error reading experiment status: {str(e)}")
        return None


def display_control_buttons(state):
    """
    Display control buttons for AL cycle management using Command Pattern.
    
    This function implements the Command Pattern where buttons ONLY write commands
    to state and NEVER call training functions directly. The worker process is
    responsible for executing all commands.
    """
    if not state:
        return
    
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.subheader("🎮 Cycle Control")
    
    # Display current phase prominently
    icon, phase_text, css_class = get_phase_display(state.phase)
    st.markdown(f"**Current Phase:** {icon} <span class='{css_class}'>{phase_text}</span>", 
               unsafe_allow_html=True)
    
    # Show worker status
    if st.session_state.state_manager.is_worker_alive():
        st.markdown("**Worker Status:** <span class='worker-active'>🟢 Active</span>", 
                   unsafe_allow_html=True)
    else:
        st.markdown("**Worker Status:** <span class='worker-inactive'>🔴 Inactive</span>", 
                   unsafe_allow_html=True)
        st.warning("⚠️ Worker is not active. Start the worker process to execute commands.")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Determine button states based on current phase
    can_start = state.phase in [ExperimentPhase.IDLE, ExperimentPhase.AWAITING_ANNOTATION]
    can_pause = state.phase in [ExperimentPhase.TRAINING, ExperimentPhase.EVALUATING, ExperimentPhase.QUERYING]
    can_stop = state.phase not in [ExperimentPhase.IDLE, ExperimentPhase.COMPLETED, ExperimentPhase.ERROR]
    can_continue = state.phase == ExperimentPhase.AWAITING_ANNOTATION
    
    # START CYCLE Button - Command Pattern Implementation
    with col1:
        start_button_type = "primary" if can_start else "secondary"
        start_help = get_button_help_text("start", state.phase)
        
        if st.button(
            "🚀 Start Cycle",
            disabled=not can_start,
            use_container_width=True,
            type=start_button_type,
            help=start_help
        ):
            try:
                # COMMAND PATTERN: Only write command, never call training functions
                st.session_state.state_manager.set_command(Command.START_CYCLE)
                st.success("✅ START_CYCLE command sent to worker")
                logger.info(f"START_CYCLE command issued for experiment {state.experiment_id}")
                
                # Brief pause for command to be processed, then refresh
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to send START_CYCLE command: {str(e)}")
                logger.error(f"Failed to send START_CYCLE command: {e}")
    
    # PAUSE Button - Command Pattern Implementation
    with col2:
        pause_help = get_button_help_text("pause", state.phase)
        
        if st.button(
            "⏸️ Pause",
            disabled=not can_pause,
            use_container_width=True,
            help=pause_help
        ):
            try:
                # COMMAND PATTERN: Only write command, never call training functions
                st.session_state.state_manager.set_command(Command.PAUSE)
                st.warning("⏸️ PAUSE command sent to worker")
                logger.info(f"PAUSE command issued for experiment {state.experiment_id}")
                
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to send PAUSE command: {str(e)}")
                logger.error(f"Failed to send PAUSE command: {e}")
    
    # STOP Button - Command Pattern Implementation
    with col3:
        stop_help = get_button_help_text("stop", state.phase)
        
        if st.button(
            "🛑 Stop",
            disabled=not can_stop,
            use_container_width=True,
            help=stop_help
        ):
            try:
                # COMMAND PATTERN: Only write command, never call training functions
                st.session_state.state_manager.set_command(Command.STOP)
                st.error("🛑 STOP command sent to worker")
                logger.info(f"STOP command issued for experiment {state.experiment_id}")
                
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to send STOP command: {str(e)}")
                logger.error(f"Failed to send STOP command: {e}")
    
    # CONTINUE Button - Command Pattern Implementation
    with col4:
        continue_button_type = "primary" if can_continue else "secondary"
        continue_help = get_button_help_text("continue", state.phase)
        
        # Check if annotations have been submitted
        annotations_ready = st.session_state.state_manager.annotations_pending()
        
        if st.button(
            "▶️ Continue",
            disabled=not can_continue or not annotations_ready,
            use_container_width=True,
            type=continue_button_type,
            help=continue_help if annotations_ready else "Submit annotations first before continuing"
        ):
            try:
                # COMMAND PATTERN: Only write command, never call training functions
                st.session_state.state_manager.set_command(Command.CONTINUE)
                st.info("▶️ CONTINUE command sent to worker")
                logger.info(f"CONTINUE command issued for experiment {state.experiment_id}")
                
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to send CONTINUE command: {str(e)}")
                logger.error(f"Failed to send CONTINUE command: {e}")
    
    # Display phase-specific guidance
    display_phase_guidance(state.phase)
    
    # Show current command if any
    if state.command:
        st.info(f"🔄 Pending command: **{state.command.value}**")
    
    # Show annotation status if in awaiting phase
    if state.phase == ExperimentPhase.AWAITING_ANNOTATION:
        if annotations_ready:
            st.success("✅ Annotations submitted. Click **Continue** to proceed to the next cycle.")
        else:
            st.warning("⏳ Waiting for annotations. Submit annotations below, then click **Continue**.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def get_button_help_text(button_type: str, phase: ExperimentPhase) -> str:
    """Get contextual help text for control buttons based on current phase."""
    help_texts = {
        "start": {
            ExperimentPhase.IDLE: "Start the next Active Learning cycle",
            ExperimentPhase.AWAITING_ANNOTATION: "Start the next cycle after annotations are confirmed",
            "default": "Cannot start cycle in current phase"
        },
        "pause": {
            ExperimentPhase.TRAINING: "Pause training after current epoch completes",
            ExperimentPhase.EVALUATING: "Pause evaluation process",
            ExperimentPhase.QUERYING: "Pause query selection process",
            "default": "Cannot pause in current phase"
        },
        "stop": {
            ExperimentPhase.TRAINING: "Stop training and save current progress",
            ExperimentPhase.EVALUATING: "Stop evaluation and save progress",
            ExperimentPhase.QUERYING: "Stop querying process",
            "default": "Stop current operation and save progress"
        },
        "continue": {
            ExperimentPhase.AWAITING_ANNOTATION: "Continue to next cycle with confirmed annotations",
            "default": "Cannot continue in current phase"
        }
    }
    
    button_help = help_texts.get(button_type, {})
    return button_help.get(phase, button_help.get("default", ""))


def display_phase_guidance(phase: ExperimentPhase):
    """Display contextual guidance based on current phase."""
    guidance_messages = {
        ExperimentPhase.IDLE: {
            "message": "💡 Click **Start Cycle** to begin the next Active Learning cycle",
            "type": "info"
        },
        ExperimentPhase.TRAINING: {
            "message": "🔄 Training in progress. Monitor live charts above. You can **Pause** or **Stop** at any time",
            "type": "info"
        },
        ExperimentPhase.EVALUATING: {
            "message": "📊 Model evaluation in progress. This usually takes a few moments",
            "type": "info"
        },
        ExperimentPhase.QUERYING: {
            "message": "🎯 Selecting most informative samples for annotation",
            "type": "info"
        },
        ExperimentPhase.AWAITING_ANNOTATION: {
            "message": "📝 Review and annotate the queried images below, then click **Continue** to proceed",
            "type": "info"
        },
        ExperimentPhase.COMPLETED: {
            "message": "🎉 All cycles completed! Check the Results page for detailed analysis",
            "type": "success"
        },
        ExperimentPhase.ERROR: {
            "message": "❌ An error occurred. Check the error message above and restart if needed",
            "type": "error"
        }
    }
    
    guidance = guidance_messages.get(phase)
    if guidance:
        message_type = guidance["type"]
        message_text = guidance["message"]
        
        if message_type == "info":
            st.info(message_text)
        elif message_type == "success":
            st.success(message_text)
        elif message_type == "error":
            st.error(message_text)
        else:
            st.write(message_text)


def create_training_charts_containers():
    """Create containers for live training charts."""
    st.markdown('<div class="training-charts">', unsafe_allow_html=True)
    st.subheader("📈 Live Training Progress")
    
    # Create containers that will be updated during polling
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**Loss Curves**")
        loss_chart_container = st.empty()
    
    with chart_col2:
        st.markdown("**Accuracy Curves**")
        accuracy_chart_container = st.empty()
    
    # Progress and metrics containers
    progress_container = st.empty()
    metrics_container = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        "loss_chart": loss_chart_container,
        "accuracy_chart": accuracy_chart_container,
        "progress": progress_container,
        "metrics": metrics_container
    }


def create_prediction_monitor_section():
    """Create section for prediction monitor display."""
    st.markdown('<div class="prediction-monitor">', unsafe_allow_html=True)
    st.subheader("🔍 Prediction Monitor")
    st.markdown("Track how model predictions change across cycles on reference images")
    
    # Container for probe images
    probe_images_container = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return probe_images_container


def create_queried_images_section():
    """Create section for queried images display - returns None, content rendered directly."""
    # Note: We return None here because queried images need to render 
    # directly in main(), not in an st.empty() container which can only
    # hold one element at a time.
    return None


def main():
    """Main Active Learning Control page."""
    initialize_session_state()
    
    st.title("🎯 Active Learning Control")
    st.markdown("Interactive control interface for managing Active Learning cycles")
    
    # Check if experiment is selected
    if not check_experiment_selected():
        return
    
    # Display cycle progress and get current state
    state = display_cycle_progress()
    
    if not state:
        return
    
    # Display control buttons
    display_control_buttons(state)
    
    # Create containers for different sections
    training_containers = create_training_charts_containers()
    probe_container = create_prediction_monitor_section()
    
    # Show appropriate content based on current phase
    if state.phase == ExperimentPhase.TRAINING:
        # Show live training visualization with polling
        display_live_training_visualization(training_containers, state)
        
        # Auto-refresh for live updates during training
        time.sleep(st.session_state.poll_interval)
        st.rerun()
        
    elif state.phase in [ExperimentPhase.EVALUATING, ExperimentPhase.QUERYING]:
        # Show last training results
        display_last_training_results(training_containers, state)
        
        # Continue polling during evaluation/querying phases
        time.sleep(st.session_state.poll_interval)
        st.rerun()
        
    elif state.phase == ExperimentPhase.AWAITING_ANNOTATION:
        # Show queried images for annotation (renders directly, not in container)
        display_queried_images(state)
    
    # Always show prediction monitor if probe images exist
    display_prediction_monitor(probe_container, state)


def check_early_stopping(state) -> tuple:
    """Check if early stopping occurred by analyzing epoch metrics."""
    if not state.current_cycle_epochs or not state.config:
        return False, 0, ""
    
    current_epoch = len(state.current_cycle_epochs)
    total_epochs = state.config.epochs_per_cycle
    
    if current_epoch < total_epochs and state.phase != ExperimentPhase.TRAINING:
        val_metrics = [m for m in state.current_cycle_epochs if m.val_accuracy is not None]
        
        if len(val_metrics) >= 3:
            recent_val_accs = [m.val_accuracy for m in val_metrics[-3:]]
            
            if len(set([round(acc, 4) for acc in recent_val_accs])) <= 2:
                return True, current_epoch, "Validation accuracy plateau detected"
        
        return False, current_epoch, "Training stopped manually"
    
    return False, 0, ""


def display_early_stopping_indicator(containers, state):
    """Display early stopping indicator if early stopping was triggered."""
    early_stopped, stop_epoch, reason = check_early_stopping(state)
    
    if early_stopped:
        with containers["loss_chart"]:
            if state.current_cycle_epochs:
                st.markdown(
                    f'<div class="early-stop-marker">⏹️ Early Stop at Epoch {stop_epoch}</div>',
                    unsafe_allow_html=True
                )
        
        with containers["accuracy_chart"]:
            if state.current_cycle_epochs:
                st.markdown(
                    f'<div class="early-stop-marker">📊 {reason}</div>',
                    unsafe_allow_html=True
                )
        
        with containers["metrics"]:
            st.warning(f"⏹️ **Early Stopping Triggered**\n\n"
                      f"Training stopped at epoch {stop_epoch}/{state.config.epochs_per_cycle}\n\n"
                      f"Reason: {reason}")
    
    return early_stopped


def display_live_training_visualization(containers, state):
    """Display live training visualization with state polling."""
    early_stopped = display_early_stopping_indicator(containers, state)
    
    # Update progress bar
    with containers["progress"]:
        if state.current_cycle_epochs:
            current_epoch = len(state.current_cycle_epochs)
            total_epochs = state.config.epochs_per_cycle if state.config else 10
            
            progress = min(current_epoch / total_epochs, 1.0)
            st.progress(progress)
            
            if st.session_state.last_poll_time:
                time_since_last = datetime.now() - st.session_state.last_poll_time
                epoch_text = f"Epoch {current_epoch}/{total_epochs} (Updated {time_since_last.seconds}s ago)"
            else:
                epoch_text = f"Epoch {current_epoch}/{total_epochs}"
            
            if early_stopped:
                epoch_text += " ⏹️ EARLY STOPPED"
            
            st.write(epoch_text)
        else:
            st.info("🔄 Training starting... Waiting for first epoch data")
    
    # Update current epoch metrics
    if not early_stopped:
        with containers["metrics"]:
            if state.current_cycle_epochs:
                latest_metrics = state.current_cycle_epochs[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Loss", f"{latest_metrics.train_loss:.4f}")
                with col2:
                    st.metric("Train Acc", f"{latest_metrics.train_accuracy:.3f}")
                with col3:
                    if latest_metrics.val_loss is not None:
                        st.metric("Val Loss", f"{latest_metrics.val_loss:.4f}")
                    else:
                        st.metric("Val Loss", "N/A")
                with col4:
                    if latest_metrics.val_accuracy is not None:
                        st.metric("Val Acc", f"{latest_metrics.val_accuracy:.3f}")
                    else:
                        st.metric("Val Acc", "N/A")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train Loss", "Waiting...")
                with col2:
                    st.metric("Train Acc", "Waiting...")
                with col3:
                    st.metric("Val Loss", "Waiting...")
                with col4:
                    st.metric("Val Acc", "Waiting...")
    
    # Update loss curve chart
    with containers["loss_chart"]:
        if state.current_cycle_epochs and len(state.current_cycle_epochs) >= 1:
            epochs = [m.epoch for m in state.current_cycle_epochs]
            train_losses = [m.train_loss for m in state.current_cycle_epochs]
            
            chart_data = pd.DataFrame({
                "Epoch": epochs,
                "Train Loss": train_losses
            })
            
            val_losses = [m.val_loss for m in state.current_cycle_epochs if m.val_loss is not None]
            if val_losses and len(val_losses) == len(epochs):
                chart_data["Val Loss"] = val_losses
            
            st.line_chart(chart_data.set_index("Epoch"))
            
            if len(train_losses) >= 2:
                trend = "📉 Decreasing" if train_losses[-1] < train_losses[-2] else "📈 Increasing"
                st.caption(f"Loss trend: {trend}")
        else:
            st.info("📊 Waiting for training data to plot loss curves...")
    
    # Update accuracy curve chart
    with containers["accuracy_chart"]:
        if state.current_cycle_epochs and len(state.current_cycle_epochs) >= 1:
            epochs = [m.epoch for m in state.current_cycle_epochs]
            train_accs = [m.train_accuracy for m in state.current_cycle_epochs]
            
            chart_data = pd.DataFrame({
                "Epoch": epochs,
                "Train Accuracy": train_accs
            })
            
            val_accs = [m.val_accuracy for m in state.current_cycle_epochs if m.val_accuracy is not None]
            if val_accs and len(val_accs) == len(epochs):
                chart_data["Val Accuracy"] = val_accs
            
            st.line_chart(chart_data.set_index("Epoch"))
            
            if train_accs:
                best_acc = max(train_accs)
                best_epoch = epochs[train_accs.index(best_acc)]
                st.caption(f"Best train accuracy: {best_acc:.3f} (epoch {best_epoch})")
        else:
            st.info("📊 Waiting for training data to plot accuracy curves...")
    
    st.session_state.last_poll_time = datetime.now()


def display_last_training_results(containers, state):
    """Display results from the last completed training."""
    with containers["progress"]:
        if state.current_cycle_epochs:
            st.success(f"✅ Training completed - {len(state.current_cycle_epochs)} epochs")
        else:
            st.info("No training data available")
    
    display_live_training_visualization(containers, state)


def display_prediction_monitor(container, state):
    """Display prediction monitor with probe images."""
    with container:
        if not state.probe_images:
            st.info("🔍 No probe images available. They will be initialized when the first cycle starts.")
            return
        
        st.markdown("**Reference Images - Prediction History**")
        st.markdown("Track how model predictions evolve across Active Learning cycles")
        
        num_images = min(12, len(state.probe_images))
        cols_per_row = 4
        
        for i in range(0, num_images, cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j in range(cols_per_row):
                idx = i + j
                if idx >= num_images:
                    break
                
                probe_img = state.probe_images[idx]
                
                with cols[j]:
                    st.markdown(f"**Image {probe_img.image_id}**")
                    st.markdown(f"**True Class:** {probe_img.true_class}")
                    st.markdown("🖼️ *[Image placeholder]*")
                    
                    if probe_img.predictions_by_cycle:
                        st.markdown("**Prediction History:**")
                        
                        sorted_cycles = sorted(probe_img.predictions_by_cycle.items())
                        
                        previous_pred = None
                        for cycle, pred_data in sorted_cycles:
                            pred_class = pred_data.get("predicted_class", "Unknown")
                            confidence = pred_data.get("confidence", 0.0)
                            
                            is_correct = pred_class == probe_img.true_class
                            correctness_icon = "✅" if is_correct else "❌"
                            
                            change_icon = ""
                            if previous_pred and previous_pred != pred_class:
                                change_icon = " 🔄"
                            
                            confidence_pct = f"{confidence:.0%}" if confidence > 0 else "N/A"
                            
                            pred_text = f"C{cycle}: **{pred_class}** ({confidence_pct}) {correctness_icon}{change_icon}"
                            
                            if is_correct:
                                st.markdown(f'<div style="color: #4ade80;">{pred_text}</div>', 
                                          unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div style="color: #ef4444;">{pred_text}</div>', 
                                          unsafe_allow_html=True)
                            
                            previous_pred = pred_class
                        
                        # Summary statistics
                        total_predictions = len(sorted_cycles)
                        correct_predictions = sum(1 for _, pred in sorted_cycles 
                                                if pred.get("predicted_class") == probe_img.true_class)
                        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                        
                        st.markdown(f"**Accuracy:** {accuracy:.0%} ({correct_predictions}/{total_predictions})")
                        
                        pred_classes = [pred.get("predicted_class") for _, pred in sorted_cycles]
                        unique_predictions = len(set(pred_classes))
                        if unique_predictions > 1:
                            st.markdown(f"🔄 **{unique_predictions} different predictions**")
                    
                    else:
                        st.markdown("*No predictions yet*")
                        st.info("Predictions will appear after the first cycle completes")


# ============================================================================
# FIXED: Queried Images Display with Actual Image Thumbnails and Annotation Submission
# ============================================================================

def display_queried_images(state):
    """
    Display queried images awaiting annotation with actual image thumbnails.
    
    This function renders directly (not in a container) because st.empty()
    can only hold one element, and we need to display multiple images.
    """
    if not state.queried_images:
        st.info("No queried images available.")
        return
    
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    st.subheader("🎯 Queried Images")
    st.markdown(f"### 📋 {len(state.queried_images)} images selected for annotation")
    
    # Annotation mode selection
    col_mode1, col_mode2 = st.columns([2, 1])
    
    with col_mode1:
        use_ground_truth = st.checkbox(
            "Use Ground Truth Labels (Simulated Mode)",
            value=True,
            help="Automatically use ground truth labels for annotation. Uncheck for manual labeling."
        )
    
    # Get class names from state config
    if state.config and state.config.class_names:
        class_names = state.config.class_names
    else:
        # Fallback: extract from queried images
        class_names = list(set(img.ground_truth_name for img in state.queried_images))
    
    st.markdown("---")
    
    # Display queried images in a grid (2 columns for better image visibility)
    cols_per_row = 2
    
    for row_start in range(0, len(state.queried_images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            img_idx = row_start + col_idx
            if img_idx >= len(state.queried_images):
                break
            
            queried_img = state.queried_images[img_idx]
            
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**Image {queried_img.image_id}**")
                    
                    # Display actual image thumbnail
                    display_image_thumbnail(queried_img)
                    
                    # Model predictions
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.markdown(f"**Predicted:** {queried_img.predicted_class}")
                        st.markdown(f"**Confidence:** {queried_img.predicted_confidence:.1%}")
                    with col_pred2:
                        st.markdown(f"**Uncertainty:** {queried_img.uncertainty_score:.3f}")
                        st.caption(f"Reason: {queried_img.selection_reason}")
                    
                    # Ground truth or manual selection
                    if use_ground_truth:
                        # Show ground truth (simulated mode)
                        is_correct = queried_img.predicted_class == queried_img.ground_truth_name
                        icon = "✅" if is_correct else "❌"
                        st.markdown(f"**Ground Truth:** {queried_img.ground_truth_name} {icon}")
                    else:
                        # Manual label selection
                        current_selection = st.session_state.manual_annotations.get(
                            queried_img.image_id, 
                            queried_img.ground_truth_name  # Default to ground truth
                        )
                        
                        selected_label = st.selectbox(
                            "Select Label",
                            options=class_names,
                            index=class_names.index(current_selection) if current_selection in class_names else 0,
                            key=f"label_{queried_img.image_id}"
                        )
                        
                        st.session_state.manual_annotations[queried_img.image_id] = selected_label
                    
                    # Expandable probability distribution
                    with st.expander("View Probabilities"):
                        if queried_img.model_probabilities:
                            prob_df = pd.DataFrame({
                                "Class": list(queried_img.model_probabilities.keys()),
                                "Probability": list(queried_img.model_probabilities.values())
                            }).sort_values("Probability", ascending=False)
                            st.bar_chart(prob_df.set_index("Class"))
                        else:
                            st.write("No probability data available")
                    
                    st.markdown("---")
    
    # Annotation submission section
    st.markdown("### ✅ Submit Annotations")
    
    # Show summary
    if use_ground_truth:
        # Calculate accuracy preview
        correct_count = sum(
            1 for img in state.queried_images 
            if img.predicted_class == img.ground_truth_name
        )
        total = len(state.queried_images)
        st.info(f"📊 Using ground truth labels. Model predicted {correct_count}/{total} ({correct_count/total:.0%}) correctly.")
    else:
        annotated_count = len(st.session_state.manual_annotations)
        st.info(f"📝 Manual annotation mode. {annotated_count} labels assigned.")
    
    # Check if annotations already submitted
    annotations_pending = st.session_state.state_manager.annotations_pending()
    
    if annotations_pending:
        st.success("✅ Annotations already submitted! Click **Continue** in the control panel to proceed.")
    else:
        # Submit button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("✅ Confirm & Submit Annotations", type="primary", use_container_width=True):
                success = submit_annotations(state, use_ground_truth, class_names)
                if success:
                    st.success("✅ Annotations submitted successfully! Click **Continue** in the control panel to proceed.")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_image_thumbnail(queried_img):
    """Display image thumbnail for a queried image."""
    image_path = None
    
    # Try display_path first, then fall back to image_path
    if queried_img.display_path and Path(queried_img.display_path).exists():
        image_path = queried_img.display_path
    elif queried_img.image_path and Path(queried_img.image_path).exists():
        image_path = queried_img.image_path
    
    if image_path:
        try:
            st.image(image_path, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Could not load image: {e}")
            st.markdown("🖼️ *[Image unavailable]*")
    else:
        # Show placeholder with path info for debugging
        st.markdown("🖼️ *[Image not found]*")
        with st.expander("Debug Info"):
            st.caption(f"display_path: {queried_img.display_path}")
            st.caption(f"image_path: {queried_img.image_path}")


def submit_annotations(state, use_ground_truth: bool, class_names: list) -> bool:
    """
    Submit annotations to the state file for the worker to process.
    
    This writes the AnnotationSubmission to user_annotations.json which
    the worker will read and process.
    """
    try:
        annotations = []
        
        for queried_img in state.queried_images:
            if use_ground_truth:
                # Use ground truth label
                user_label = queried_img.ground_truth
                user_label_name = queried_img.ground_truth_name
            else:
                # Use manual annotation from session state
                user_label_name = st.session_state.manual_annotations.get(
                    queried_img.image_id,
                    queried_img.ground_truth_name  # Default to ground truth
                )
                # Convert name to index
                if class_names and user_label_name in class_names:
                    user_label = class_names.index(user_label_name)
                else:
                    user_label = queried_img.ground_truth  # Fallback
            
            # Check if annotation is correct
            was_correct = (user_label == queried_img.ground_truth)
            
            annotation = UserAnnotation(
                image_id=queried_img.image_id,
                user_label=user_label,
                user_label_name=user_label_name,
                timestamp=datetime.now(),
                was_correct=was_correct
            )
            annotations.append(annotation)
        
        # Create submission
        submission = AnnotationSubmission(
            experiment_id=state.experiment_id,
            cycle=state.current_cycle,
            annotations=annotations,
            submitted_at=datetime.now()
        )
        
        # Write to state file
        st.session_state.state_manager.write_annotations(submission)
        
        # Log summary
        correct_count = sum(1 for a in annotations if a.was_correct)
        total = len(annotations)
        
        st.info(f"📊 Annotation accuracy: {correct_count}/{total} ({correct_count/total:.0%}) correct")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to submit annotations: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False


if __name__ == "__main__":
    main()