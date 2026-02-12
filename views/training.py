"""
Training visualization view for Active Learning Framework.

This module provides real-time training visualization during the TRAINING state:
- Progress bar for current epoch within cycle
- Cycle progress indicator
- Live loss/accuracy line charts
- Pool statistics display
- Current metrics display

The view polls the Controller for progress updates and renders charts dynamically.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from controller import Controller

logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 12.1: Progress bar for current epoch within cycle
# ============================================================================

def render_epoch_progress(controller: Controller, config_overrides: Dict[str, Any]) -> None:
    """
    Render progress bar for current epoch within cycle.
    
    Args:
        controller: Controller instance
        config_overrides: Configuration overrides from sidebar
    """
    # Get progress info from controller
    progress = controller.get_progress()
    current_epoch = progress.get('current_epoch', 0)
    
    # Get total epochs from config
    total_epochs = config_overrides.get('training.epochs', 5)
    
    # Calculate progress percentage
    if total_epochs > 0:
        epoch_progress = current_epoch / total_epochs
    else:
        epoch_progress = 0.0
    
    # Render progress bar
    st.progress(
        epoch_progress,
        text=f"Epoch {current_epoch} / {total_epochs}"
    )


# ============================================================================
# SUBTASK 12.2: Cycle progress indicator (Cycle X/N)
# ============================================================================

def render_cycle_progress(controller: Controller) -> None:
    """
    Render cycle progress indicator.
    
    Args:
        controller: Controller instance
    """
    progress = controller.get_progress()
    current_cycle = progress.get('current_cycle', 0)
    total_cycles = progress.get('total_cycles', 0)
    
    # Create columns for cycle info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Current Cycle",
            value=f"{current_cycle} / {total_cycles}"
        )
    
    with col2:
        cycles_completed = progress.get('cycles_completed', 0)
        st.metric(
            label="Completed Cycles",
            value=cycles_completed
        )
    
    with col3:
        if total_cycles > 0:
            completion_pct = (cycles_completed / total_cycles) * 100
            st.metric(
                label="Overall Progress",
                value=f"{completion_pct:.1f}%"
            )


# ============================================================================
# SUBTASK 12.3: Live loss/accuracy line chart
# ============================================================================

def render_training_charts(controller: Controller) -> None:
    """
    Render live loss/accuracy line charts.
    
    Updates each epoch via polling from controller's epoch metrics.
    
    Args:
        controller: Controller instance
    """
    # Get epoch metrics from controller
    epoch_metrics = controller.epoch_metrics
    
    if not epoch_metrics:
        st.info("📊 Training charts will appear once training starts...")
        return
    
    # Create two columns for loss and accuracy charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Training Loss")
        
        # Prepare loss data
        loss_data = {
            'Epoch': [m['epoch'] for m in epoch_metrics if 'train_loss' in m],
            'Train Loss': [m['train_loss'] for m in epoch_metrics if 'train_loss' in m],
        }
        
        # Add validation loss if available
        if any('val_loss' in m and m['val_loss'] is not None for m in epoch_metrics):
            loss_data['Val Loss'] = [
                m.get('val_loss', None) for m in epoch_metrics if 'train_loss' in m
            ]
        
        if loss_data['Epoch']:
            st.line_chart(
                loss_data,
                x='Epoch',
                y=['Train Loss'] + (['Val Loss'] if 'Val Loss' in loss_data else []),
                height=300
            )
        else:
            st.caption("No loss data yet...")
    
    with col2:
        st.markdown("#### Training Accuracy")
        
        # Prepare accuracy data
        acc_data = {
            'Epoch': [m['epoch'] for m in epoch_metrics if 'train_accuracy' in m],
            'Train Acc': [m['train_accuracy'] * 100 for m in epoch_metrics if 'train_accuracy' in m],
        }
        
        # Add validation accuracy if available
        if any('val_accuracy' in m and m['val_accuracy'] is not None for m in epoch_metrics):
            acc_data['Val Acc'] = [
                m.get('val_accuracy', None) * 100 if m.get('val_accuracy') is not None else None
                for m in epoch_metrics if 'train_accuracy' in m
            ]
        
        if acc_data['Epoch']:
            st.line_chart(
                acc_data,
                x='Epoch',
                y=['Train Acc'] + (['Val Acc'] if 'Val Acc' in acc_data else []),
                height=300
            )
        else:
            st.caption("No accuracy data yet...")


# ============================================================================
# SUBTASK 12.4: Pool statistics display
# ============================================================================

def render_pool_statistics(controller: Controller) -> None:
    """
    Render pool statistics display (labeled/unlabeled counts).
    
    Args:
        controller: Controller instance
    """
    st.markdown("### 📊 Data Pool Status")
    
    # Get pool statistics from latest cycle metrics
    metrics_history = controller.metrics_history
    
    if not metrics_history:
        st.info("Pool statistics will appear after the first cycle completes")
        return
    
    # Get latest cycle metrics
    latest_metrics = metrics_history[-1]
    
    labeled_size = latest_metrics.get('labeled_pool_size', 0)
    unlabeled_size = latest_metrics.get('unlabeled_pool_size', 0)
    total_size = labeled_size + unlabeled_size
    
    # Create columns for pool stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Labeled Pool",
            value=labeled_size,
            delta=f"{(labeled_size / total_size * 100):.1f}%" if total_size > 0 else "0%"
        )
    
    with col2:
        st.metric(
            label="Unlabeled Pool",
            value=unlabeled_size,
            delta=f"{(unlabeled_size / total_size * 100):.1f}%" if total_size > 0 else "0%"
        )
    
    with col3:
        st.metric(
            label="Total Samples",
            value=total_size
        )
    
    # Visual progress bar for pool distribution
    if total_size > 0:
        labeled_pct = labeled_size / total_size
        st.progress(labeled_pct, text=f"Labeled: {labeled_pct * 100:.1f}%")


# ============================================================================
# SUBTASK 12.5: Current metrics display
# ============================================================================

def render_current_metrics(controller: Controller) -> None:
    """
    Render current metrics display (train_loss, train_acc, val_acc).
    
    Args:
        controller: Controller instance
    """
    st.markdown("### 📈 Current Metrics")
    
    # Get latest epoch metrics from controller
    epoch_metrics = controller.epoch_metrics
    
    if not epoch_metrics:
        st.info("Metrics will appear once training starts...")
        return
    
    # Get the most recent epoch metrics
    latest_epoch = epoch_metrics[-1]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_loss = latest_epoch.get('train_loss', 0.0)
        st.metric(
            label="Train Loss",
            value=f"{train_loss:.4f}"
        )
    
    with col2:
        train_acc = latest_epoch.get('train_accuracy', 0.0)
        st.metric(
            label="Train Accuracy",
            value=f"{train_acc * 100:.2f}%"
        )
    
    with col3:
        val_loss = latest_epoch.get('val_loss')
        if val_loss is not None:
            st.metric(
                label="Val Loss",
                value=f"{val_loss:.4f}"
            )
        else:
            st.metric(
                label="Val Loss",
                value="N/A"
            )
    
    with col4:
        val_acc = latest_epoch.get('val_accuracy')
        if val_acc is not None:
            st.metric(
                label="Val Accuracy",
                value=f"{val_acc * 100:.2f}%"
            )
        else:
            st.metric(
                label="Val Accuracy",
                value="N/A"
            )
    
    # Show learning rate if available
    if 'learning_rate' in latest_epoch and latest_epoch['learning_rate'] is not None:
        st.caption(f"Learning Rate: {latest_epoch['learning_rate']:.2e}")


# ============================================================================
# MAIN TRAINING VIEW RENDER FUNCTION
# ============================================================================

def render_training_view(controller: Controller) -> None:
    """
    Main render function for TRAINING state view.
    
    This view is shown when the model is training on labeled data.
    
    Displays:
    - Progress bar for current epoch
    - Cycle progress indicator
    - Live loss/accuracy charts
    - Pool statistics
    - Current metrics
    
    Args:
        controller: Controller instance
    """
    # Header
    st.title("🔄 Training in Progress")
    st.markdown("---")
    
    # Get config overrides from session state
    config_overrides = st.session_state.get('config_overrides', {})
    
    # Render cycle progress (Task 12.2)
    render_cycle_progress(controller)
    
    st.markdown("---")
    
    # Render epoch progress bar (Task 12.1)
    st.markdown("### ⏱️ Epoch Progress")
    render_epoch_progress(controller, config_overrides)
    
    st.markdown("---")
    
    # Render training charts (Task 12.3)
    st.markdown("### 📊 Training Metrics")
    render_training_charts(controller)
    
    st.markdown("---")
    
    # Render current metrics (Task 12.5)
    render_current_metrics(controller)
    
    st.markdown("---")
    
    # Render pool statistics (Task 12.4)
    render_pool_statistics(controller)
    
    # Auto-refresh hint
    st.caption("💡 This view updates automatically as training progresses")
