"""
Active Learning Control Page - Interactive AL Cycle Management

This page provides the main interface for controlling and monitoring Active Learning experiments.
It implements the MVC pattern where the dashboard (controller) sends commands
to the service process and monitors state through polling.

Features:
- Cycle progress display (cycle number, pool sizes)
- Control buttons (Start Cycle, Pause, Stop) using Command Pattern
- Live training visualization with @st.fragment for efficient auto-refresh
- Prediction monitor for reference images
- Query visualization and annotation interface

The page NEVER calls training functions directly - it only reads state and dispatches events.
The service process is responsible for executing all training operations.
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
from types import SimpleNamespace
from emoji_sanitizer import EmojiSanitizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MVC imports
from controller.controller_factory import get_controller, update_session_heartbeat
from controller.events import Event, EventType
from model.schemas import (
    ExperimentPhase, EpochMetrics, QueriedImage,
    UserAnnotation, AnnotationSubmission
)

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Active Learning Control - AL Dashboard",
    layout="wide"
)
EmojiSanitizer(st).patch()

# Custom CSS for AL Control page
st.markdown("""
    <style>
    .control-panel {
        background-color: #112240;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .phase-idle { color: #8892b0; font-weight: bold; }
    .phase-training { color: #4ade80; font-weight: bold; }
    .phase-evaluating { color: #fbbf24; font-weight: bold; }
    .phase-querying { color: #f97316; font-weight: bold; }
    .phase-awaiting { color: #a855f7; font-weight: bold; }
    .phase-error { color: #ef4444; font-weight: bold; }
    .phase-completed { color: #10b981; font-weight: bold; }
    .training-charts {
        background-color: #0f1419;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #233554;
    }
    .prediction-monitor {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #4a4a6a;
    }
    .query-section {
        background-color: #0f3d3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #20c997;
    }
    .worker-active { color: #4ade80; }
    .worker-inactive { color: #ef4444; }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for AL Control page (MVC version)."""
    update_session_heartbeat()
    if "training_active" not in st.session_state:
        st.session_state.training_active = False
    if "last_poll_time" not in st.session_state:
        st.session_state.last_poll_time = None
    if "poll_interval" not in st.session_state:
        st.session_state.poll_interval = 1.0
    if "manual_annotations" not in st.session_state:
        st.session_state.manual_annotations = {}


def check_experiment_selected():
    """Check if an experiment is selected and controller is available."""
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        return status.get('experiment_id') is not None
    except Exception:
        st.warning("No experiment selected. Please create one in the Configuration page.")
        return False


def get_phase_display(phase: ExperimentPhase) -> tuple:
    """Get display information for a phase."""
    phase_info = {
        ExperimentPhase.IDLE: ("IDLE", "phase-idle"),
        ExperimentPhase.INITIALIZING: ("INITIALIZING", "phase-evaluating"),
        ExperimentPhase.PREPARING: ("PREPARING", "phase-evaluating"),
        ExperimentPhase.TRAINING: ("TRAINING", "phase-training"),
        ExperimentPhase.VALIDATING: ("VALIDATING", "phase-evaluating"),
        ExperimentPhase.EVALUATING: ("EVALUATING", "phase-evaluating"),
        ExperimentPhase.QUERYING: ("QUERYING", "phase-querying"),
        ExperimentPhase.AWAITING_ANNOTATION: ("AWAITING ANNOTATION", "phase-awaiting"),
        ExperimentPhase.ANNOTATIONS_SUBMITTED: ("PROCESSING", "phase-awaiting"),
        ExperimentPhase.COMPLETED: ("COMPLETED", "phase-completed"),
        ExperimentPhase.ERROR: ("ERROR", "phase-error"),
        ExperimentPhase.ABORT: ("ABORTED", "phase-error")
    }
    info = phase_info.get(phase, (str(phase), "phase-idle"))
    return ("", info[0], info[1])



def display_cycle_progress():
    """Display current cycle progress and pool sizes (MVC version)."""
    if not check_experiment_selected():
        return None
    
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("Cycle Progress")
        
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
            st.metric("Latest Test Acc", "N/A")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            phase = status.get('phase', 'UNKNOWN')
            _, phase_text, css_class = get_phase_display(ExperimentPhase(phase))
            st.markdown(f"**Current Phase:** <span class='{css_class}'>{phase_text}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            if ctrl.is_service_alive():
                st.markdown("**Service Status:** <span class='worker-active'>Active</span>", 
                           unsafe_allow_html=True)
            else:
                st.markdown("**Service Status:** <span class='worker-inactive'>Inactive</span>", 
                           unsafe_allow_html=True)
        
        error_message = status.get('error_message')
        if error_message:
            st.error(f"Error: {error_message}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return SimpleNamespace(
            phase=ExperimentPhase(status.get('phase', 'IDLE')),
            experiment_id=status.get('experiment_id'),
            experiment_name=status.get('experiment_name'),
            current_cycle=status.get('current_cycle', 0),
            total_cycles=status.get('total_cycles', 0),
            current_epoch=status.get('current_epoch', 0),
            epochs_per_cycle=status.get('epochs_per_cycle', 0),
            labeled_count=status.get('labeled_count', 0),
            unlabeled_count=status.get('unlabeled_count', 0),
            error_message=status.get('error_message'),
            config=status.get('config'),
            current_cycle_epochs=status.get('current_cycle_epochs', []),
            queried_images=status.get('queried_images', []),
            probe_images=status.get('probe_images', []),
            command=None,
            _raw_status=status
        )
    
    except Exception as e:
        st.error(f"Error reading experiment status: {str(e)}")
        return None


def display_control_buttons(state):
    """
    Display control buttons for AL cycle management using Command Pattern.
    
    Buttons dispatch events to the service process via the controller.
    Training runs in the service process, not in the UI thread.
    
    Requirements: 3.1, 5.1, 5.2, 8.4
    """
    if not state:
        return
    
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.subheader("Cycle Control")
    
    _, phase_text, css_class = get_phase_display(state.phase)
    st.markdown(f"**Current Phase:** <span class='{css_class}'>{phase_text}</span>", 
               unsafe_allow_html=True)
    
    try:
        ctrl = get_controller()
        service_alive = ctrl.is_service_alive()
    except Exception:
        service_alive = False
    
    if service_alive:
        st.markdown("**Service Status:** <span class='worker-active'>Active</span>", 
                   unsafe_allow_html=True)
    else:
        st.markdown("**Service Status:** <span class='worker-inactive'>Inactive</span>", 
                   unsafe_allow_html=True)
        st.warning("Service is not active. Start the service process to execute commands.")
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    can_start = state.phase in [ExperimentPhase.IDLE, ExperimentPhase.AWAITING_ANNOTATION]
    can_pause = state.phase in [ExperimentPhase.TRAINING, ExperimentPhase.EVALUATING, ExperimentPhase.QUERYING]
    can_stop = state.phase not in [ExperimentPhase.IDLE, ExperimentPhase.COMPLETED, ExperimentPhase.ERROR]
    can_continue = state.phase == ExperimentPhase.AWAITING_ANNOTATION
    
    # START CYCLE Button - dispatches START_CYCLE event
    with col1:
        if st.button("Start Cycle", disabled=not can_start, use_container_width=True,
                    type="primary" if can_start else "secondary"):
            try:
                ctrl = get_controller()
                event = Event(EventType.START_CYCLE, payload={})
                ctrl.dispatch(event)
                st.success("START_CYCLE event dispatched")
                logger.info(f"START_CYCLE event dispatched for experiment {state.experiment_id}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to dispatch START_CYCLE event: {str(e)}")
    
    # PAUSE Button - dispatches PAUSE event
    with col2:
        if st.button("Pause", disabled=not can_pause, use_container_width=True):
            try:
                ctrl = get_controller()
                event = Event(EventType.PAUSE, payload={})
                ctrl.dispatch(event)
                st.warning("PAUSE event dispatched")
                logger.info(f"PAUSE event dispatched for experiment {state.experiment_id}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to dispatch PAUSE event: {str(e)}")
    
    # STOP Button - dispatches STOP event
    with col3:
        if st.button("Stop", disabled=not can_stop, use_container_width=True):
            try:
                ctrl = get_controller()
                event = Event(EventType.STOP, payload={})
                ctrl.dispatch(event)
                st.error("STOP event dispatched")
                logger.info(f"STOP event dispatched for experiment {state.experiment_id}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to dispatch STOP event: {str(e)}")
    
    # CONTINUE Button - dispatches CONTINUE event
    with col4:
        annotations_ready = len(st.session_state.get("manual_annotations", {})) > 0
        if st.button("Continue", disabled=not can_continue or not annotations_ready, 
                    use_container_width=True, type="primary" if can_continue else "secondary"):
            try:
                ctrl = get_controller()
                event = Event(EventType.CONTINUE, payload={})
                ctrl.dispatch(event)
                st.info("CONTINUE event dispatched")
                logger.info(f"CONTINUE event dispatched for experiment {state.experiment_id}")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to dispatch CONTINUE event: {str(e)}")
    
    # Phase-specific guidance
    if state.phase == ExperimentPhase.IDLE:
        st.info("Click **Start Cycle** to begin the next Active Learning cycle")
    elif state.phase == ExperimentPhase.TRAINING:
        st.info("Training in progress. Monitor live charts below. You can **Pause** or **Stop** at any time")
    elif state.phase == ExperimentPhase.AWAITING_ANNOTATION:
        if annotations_ready:
            st.success("Annotations submitted. Click **Continue** to proceed to the next cycle.")
        else:
            st.warning("Waiting for annotations. Submit annotations below, then click **Continue**.")
    elif state.phase == ExperimentPhase.COMPLETED:
        st.success("All cycles completed! Check the Results page for detailed analysis")
    elif state.phase == ExperimentPhase.ERROR:
        st.error("An error occurred. Check the error message above and restart if needed")
    
    st.markdown('</div>', unsafe_allow_html=True)



@st.fragment(run_every=2.0)
def live_metrics_fragment():
    """Live metrics display fragment with auto-refresh during training.
    
    This fragment auto-refreshes every 2 seconds during training, updating
    only the metrics section without causing full page flicker.
    
    Uses @st.fragment(run_every=2.0) for efficient partial page updates.
    Requirements: 3.2, 3.6
    """
    try:
        ctrl = get_controller()
        
        # Poll for updates from service process before reading state
        ctrl.poll_updates()
        status = ctrl.get_status()
        
        phase = ExperimentPhase(status.get('phase', 'idle'))
        current_cycle = status.get('current_cycle', 0)
        total_cycles = status.get('total_cycles', 0)
        current_epoch = status.get('current_epoch', 0)
        epochs_per_cycle = status.get('epochs_per_cycle', 0)
        epoch_metrics = status.get('current_cycle_epochs', [])
        
        st.markdown('<div class="training-charts">', unsafe_allow_html=True)
        st.subheader("Live Training Progress")
        
        # Progress metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if total_cycles > 0:
                st.metric("Cycle", f"{current_cycle}/{total_cycles}")
            else:
                st.metric("Cycle", "Not Started")
        
        with col2:
            if epochs_per_cycle > 0 and phase == ExperimentPhase.TRAINING:
                st.metric("Epoch", f"{current_epoch}/{epochs_per_cycle}")
            else:
                st.metric("Epoch", "-")
        
        with col3:
            _, phase_text, _ = get_phase_display(phase)
            st.metric("Phase", phase_text)
        
        # Show progress bar during training
        if phase == ExperimentPhase.TRAINING and epochs_per_cycle > 0:
            progress = min(current_epoch / epochs_per_cycle, 1.0)
            st.progress(progress)
        
        # Current epoch metrics
        if epoch_metrics and len(epoch_metrics) > 0:
            latest = epoch_metrics[-1]
            
            st.markdown("**Current Epoch Metrics**")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric("Train Loss", f"{latest.train_loss:.4f}")
            with metric_cols[1]:
                st.metric("Train Acc", f"{latest.train_accuracy:.3f}")
            with metric_cols[2]:
                if latest.val_loss is not None:
                    st.metric("Val Loss", f"{latest.val_loss:.4f}")
                else:
                    st.metric("Val Loss", "N/A")
            with metric_cols[3]:
                if latest.val_accuracy is not None:
                    st.metric("Val Acc", f"{latest.val_accuracy:.3f}")
                else:
                    st.metric("Val Acc", "N/A")
            
            # Loss and accuracy charts
            st.markdown("---")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("**Loss Curves**")
                if len(epoch_metrics) >= 1:
                    epochs = [m.epoch for m in epoch_metrics]
                    train_losses = [m.train_loss for m in epoch_metrics]
                    
                    chart_data = pd.DataFrame({
                        "Epoch": epochs,
                        "Train Loss": train_losses
                    })
                    
                    val_losses = [m.val_loss for m in epoch_metrics if m.val_loss is not None]
                    if val_losses and len(val_losses) == len(epochs):
                        chart_data["Val Loss"] = val_losses
                    
                    st.line_chart(chart_data.set_index("Epoch"))
                    
                    if len(train_losses) >= 2:
                        trend = "Decreasing" if train_losses[-1] < train_losses[-2] else "Increasing"
                        st.caption(f"Loss trend: {trend}")
                else:
                    st.info("Waiting for training data...")
            
            with chart_col2:
                st.markdown("**Accuracy Curves**")
                if len(epoch_metrics) >= 1:
                    epochs = [m.epoch for m in epoch_metrics]
                    train_accs = [m.train_accuracy for m in epoch_metrics]
                    
                    chart_data = pd.DataFrame({
                        "Epoch": epochs,
                        "Train Accuracy": train_accs
                    })
                    
                    val_accs = [m.val_accuracy for m in epoch_metrics if m.val_accuracy is not None]
                    if val_accs and len(val_accs) == len(epochs):
                        chart_data["Val Accuracy"] = val_accs
                    
                    st.line_chart(chart_data.set_index("Epoch"))
                    
                    if train_accs:
                        best_acc = max(train_accs)
                        best_epoch = epochs[train_accs.index(best_acc)]
                        st.caption(f"Best train accuracy: {best_acc:.3f} (epoch {best_epoch})")
                else:
                    st.info("Waiting for training data...")
        
        elif phase == ExperimentPhase.TRAINING:
            st.info("Training starting... Waiting for first epoch data")
        
        elif phase in [ExperimentPhase.EVALUATING, ExperimentPhase.QUERYING]:
            st.info(f"{phase.value.replace('_', ' ').title()} in progress...")
        
        elif phase == ExperimentPhase.IDLE:
            st.info("Click **Start Cycle** to begin training")
        
        elif phase == ExperimentPhase.COMPLETED:
            st.success("All cycles completed! Check the Results page for detailed analysis.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.session_state.last_poll_time = datetime.now()
        
    except Exception as e:
        st.error(f"Error updating metrics: {str(e)}")
        logger.error(f"Error in live_metrics_fragment: {e}")



def display_queried_images(state):
    """
    Display queried images awaiting annotation with image thumbnails.
    
    Displays queried_images from WorldState in a grid and collects
    annotations to dispatch SUBMIT_ANNOTATIONS event.
    
    Requirements: 4.1, 4.5
    """
    queried_images = state.queried_images if hasattr(state, 'queried_images') else []
    
    if not queried_images:
        st.info("No queried images available.")
        return
    
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    st.subheader("Queried Images")
    st.markdown(f"### {len(queried_images)} images selected for annotation")
    
    col_mode1, col_mode2 = st.columns([2, 1])
    
    with col_mode1:
        use_ground_truth = st.checkbox(
            "Use Ground Truth Labels (Simulated Mode)",
            value=True,
            help="Automatically use ground truth labels for annotation. Uncheck for manual labeling."
        )
    
    # Get class names from queried images
    class_names = list(set(img.ground_truth_name for img in queried_images if hasattr(img, 'ground_truth_name')))
    if not class_names:
        class_names = ["Unknown"]
    
    st.markdown("---")
    
    # Display queried images in a grid
    cols_per_row = 2
    
    for row_start in range(0, len(queried_images), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            img_idx = row_start + col_idx
            if img_idx >= len(queried_images):
                break
            
            queried_img = queried_images[img_idx]
            
            with cols[col_idx]:
                with st.container():
                    st.markdown(f"**Image {queried_img.image_id}**")
                    
                    # Display image thumbnail
                    image_path = None
                    if hasattr(queried_img, 'display_path') and queried_img.display_path:
                        if Path(queried_img.display_path).exists():
                            image_path = queried_img.display_path
                    if not image_path and hasattr(queried_img, 'image_path') and queried_img.image_path:
                        if Path(queried_img.image_path).exists():
                            image_path = queried_img.image_path
                    
                    if image_path:
                        try:
                            st.image(image_path, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not load image: {e}")
                    else:
                        st.markdown("*[Image not found]*")
                    
                    # Model predictions
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        pred_class = getattr(queried_img, 'predicted_class', 'N/A')
                        pred_conf = getattr(queried_img, 'predicted_confidence', 0.0)
                        st.markdown(f"**Predicted:** {pred_class}")
                        st.markdown(f"**Confidence:** {pred_conf:.1%}")
                    with col_pred2:
                        uncertainty = getattr(queried_img, 'uncertainty_score', 0.0)
                        reason = getattr(queried_img, 'selection_reason', '')
                        st.markdown(f"**Uncertainty:** {uncertainty:.3f}")
                        st.caption(f"Reason: {reason}")
                    
                    # Ground truth or manual selection
                    gt_name = getattr(queried_img, 'ground_truth_name', 'Unknown')
                    if use_ground_truth:
                        is_correct = pred_class == gt_name
                        icon = "OK" if is_correct else "X"
                        st.markdown(f"**Ground Truth:** {gt_name} ({icon})")
                    else:
                        current_selection = st.session_state.manual_annotations.get(
                            queried_img.image_id, gt_name
                        )
                        
                        selected_label = st.selectbox(
                            "Select Label",
                            options=class_names,
                            index=class_names.index(current_selection) if current_selection in class_names else 0,
                            key=f"label_{queried_img.image_id}"
                        )
                        
                        st.session_state.manual_annotations[queried_img.image_id] = selected_label
                    
                    # Expandable probability distribution
                    probs = getattr(queried_img, 'model_probabilities', {})
                    if probs:
                        with st.expander("View Probabilities"):
                            prob_df = pd.DataFrame({
                                "Class": list(probs.keys()),
                                "Probability": list(probs.values())
                            }).sort_values("Probability", ascending=False)
                            st.bar_chart(prob_df.set_index("Class"))
                    
                    st.markdown("---")
    
    # Annotation submission section
    st.markdown("### Submit Annotations")
    
    if use_ground_truth:
        correct_count = sum(
            1 for img in queried_images 
            if getattr(img, 'predicted_class', '') == getattr(img, 'ground_truth_name', '')
        )
        total = len(queried_images)
        st.info(f"Using ground truth labels. Model predicted {correct_count}/{total} ({correct_count/total:.0%}) correctly.")
    else:
        annotated_count = len(st.session_state.manual_annotations)
        st.info(f"Manual annotation mode. {annotated_count} labels assigned.")
    
    annotations_pending = len(st.session_state.get("manual_annotations", {})) > 0
    
    if annotations_pending:
        st.success("Annotations already submitted! Click **Continue** in the control panel to proceed.")
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("Confirm & Submit Annotations", type="primary", use_container_width=True):
                success = submit_annotations(state, use_ground_truth, class_names)
                if success:
                    st.success("Annotations submitted successfully! Click **Continue** in the control panel to proceed.")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def submit_annotations(state, use_ground_truth: bool, class_names: list) -> bool:
    """
    Submit annotations by dispatching SUBMIT_ANNOTATIONS event.
    
    Requirements: 4.1, 4.5
    """
    try:
        annotations = []
        queried_images = state.queried_images if hasattr(state, 'queried_images') else []
        
        for queried_img in queried_images:
            gt = getattr(queried_img, 'ground_truth', -1)
            gt_name = getattr(queried_img, 'ground_truth_name', 'Unknown')
            
            if use_ground_truth:
                user_label = gt
                user_label_name = gt_name
            else:
                user_label_name = st.session_state.manual_annotations.get(
                    queried_img.image_id, gt_name
                )
                if class_names and user_label_name in class_names:
                    user_label = class_names.index(user_label_name)
                else:
                    user_label = gt
            
            was_correct = (user_label == gt)
            
            annotation = UserAnnotation(
                image_id=queried_img.image_id,
                user_label=user_label,
                user_label_name=user_label_name,
                timestamp=datetime.now(),
                was_correct=was_correct
            )
            annotations.append(annotation)
        
        submission = AnnotationSubmission(
            experiment_id=state.experiment_id,
            cycle=state.current_cycle,
            annotations=annotations,
            submitted_at=datetime.now()
        )
        
        # Dispatch SUBMIT_ANNOTATIONS event
        ctrl = get_controller()
        event = Event(EventType.SUBMIT_ANNOTATIONS, payload={"annotations": submission})
        ctrl.dispatch(event)
        
        correct_count = sum(1 for a in annotations if a.was_correct)
        total = len(annotations)
        
        st.info(f"Annotation accuracy: {correct_count}/{total} ({correct_count/total:.0%}) correct")
        
        # Mark annotations as submitted
        st.session_state.manual_annotations = {"_submitted": True}
        
        return True
        
    except Exception as e:
        st.error(f"Failed to submit annotations: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return False



def display_prediction_monitor(state):
    """Display prediction monitor with probe images."""
    probe_images = state.probe_images if hasattr(state, 'probe_images') else []
    
    st.markdown('<div class="prediction-monitor">', unsafe_allow_html=True)
    st.subheader("Prediction Monitor")
    
    if not probe_images:
        st.info("No probe images available. They will be initialized when the first cycle starts.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.markdown("**Reference Images - Prediction History**")
    st.markdown("Track how model predictions evolve across Active Learning cycles")
    
    num_images = min(12, len(probe_images))
    cols_per_row = 4
    
    for i in range(0, num_images, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx >= num_images:
                break
            
            probe_img = probe_images[idx]
            
            with cols[j]:
                img_id = getattr(probe_img, 'image_id', f'img_{idx}')
                true_class = getattr(probe_img, 'true_class', 'Unknown')
                predictions = getattr(probe_img, 'predictions_by_cycle', {})
                
                st.markdown(f"**Image {img_id}**")
                st.markdown(f"**True Class:** {true_class}")
                st.markdown("*[Image placeholder]*")
                
                if predictions:
                    st.markdown("**Prediction History:**")
                    
                    sorted_cycles = sorted(predictions.items())
                    
                    previous_pred = None
                    for cycle, pred_data in sorted_cycles:
                        pred_class = pred_data.get("predicted_class", "Unknown")
                        confidence = pred_data.get("confidence", 0.0)
                        
                        is_correct = pred_class == true_class
                        correctness_icon = "OK" if is_correct else "X"
                        
                        change_icon = ""
                        if previous_pred and previous_pred != pred_class:
                            change_icon = " (changed)"
                        
                        confidence_pct = f"{confidence:.0%}" if confidence > 0 else "N/A"
                        
                        pred_text = f"C{cycle}: **{pred_class}** ({confidence_pct}) {correctness_icon}{change_icon}"
                        
                        if is_correct:
                            st.markdown(f'<div style="color: #4ade80;">{pred_text}</div>', 
                                      unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div style="color: #ef4444;">{pred_text}</div>', 
                                      unsafe_allow_html=True)
                        
                        previous_pred = pred_class
                    
                    total_predictions = len(sorted_cycles)
                    correct_predictions = sum(1 for _, pred in sorted_cycles 
                                            if pred.get("predicted_class") == true_class)
                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                    
                    st.markdown(f"**Accuracy:** {accuracy:.0%} ({correct_predictions}/{total_predictions})")
                    
                    pred_classes = [pred.get("predicted_class") for _, pred in sorted_cycles]
                    unique_predictions = len(set(pred_classes))
                    if unique_predictions > 1:
                        st.markdown(f"**{unique_predictions} different predictions**")
                
                else:
                    st.markdown("*No predictions yet*")
                    st.info("Predictions will appear after the first cycle completes")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main Active Learning Control page."""
    initialize_session_state()
    
    st.title("Active Learning Control")
    st.markdown("Interactive control interface for managing Active Learning cycles")
    
    if not check_experiment_selected():
        return
    
    # Display cycle progress and get current state
    state = display_cycle_progress()
    
    if not state:
        return
    
    # Display control buttons (dispatches events, no direct training calls)
    display_control_buttons(state)
    
    # Live metrics display with @st.fragment for efficient auto-refresh
    # The fragment auto-refreshes every 2 seconds during training
    live_metrics_fragment()
    
    # Show queried images for annotation when in AWAITING_ANNOTATION phase
    if state.phase == ExperimentPhase.AWAITING_ANNOTATION:
        display_queried_images(state)
    
    # Always show prediction monitor
    display_prediction_monitor(state)


if __name__ == "__main__":
    main()
