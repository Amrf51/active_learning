"""
Results dashboard view for Active Learning Framework.

This module provides the results dashboard for completed AL experiments:
- Cycle-by-cycle metrics table
- Test accuracy progression line chart
- Best cycle highlight with summary

The view displays comprehensive results after cycles complete.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from controller import Controller
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 14.1: Cycle-by-cycle metrics table
# ============================================================================

def render_metrics_table(controller: Controller) -> None:
    """
    Render cycle-by-cycle metrics table.
    
    Displays a table with columns:
    - Cycle
    - Labeled (pool size)
    - Val Acc (validation accuracy)
    - Test Acc (test accuracy)
    - F1 (F1 score)
    - Precision
    - Recall
    
    Args:
        controller: Controller instance
    """
    st.markdown("### 📊 Cycle Summary")
    
    # Get metrics history from controller
    metrics_history = controller.metrics_history
    
    if not metrics_history:
        st.info("📋 Metrics table will appear after cycles complete")
        return
    
    # Build table data
    table_data = []
    for metrics in metrics_history:
        row = {
            "Cycle": metrics.get("cycle", "N/A"),
            "Labeled": metrics.get("labeled_pool_size", 0),
            "Val Acc": f"{metrics.get('best_val_accuracy', 0) * 100:.2f}%",
            "Test Acc": f"{metrics.get('test_accuracy', 0) * 100:.2f}%",
            "F1": f"{metrics.get('test_f1', 0):.3f}",
            "Precision": f"{metrics.get('test_precision', 0):.3f}",
            "Recall": f"{metrics.get('test_recall', 0):.3f}",
        }
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Display table with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Show additional stats
    if len(metrics_history) > 0:
        total_labeled = metrics_history[-1].get("labeled_pool_size", 0)
        total_unlabeled = metrics_history[-1].get("unlabeled_pool_size", 0)
        total_samples = total_labeled + total_unlabeled
        
        st.caption(
            f"📈 Total samples: {total_samples} | "
            f"Labeled: {total_labeled} ({total_labeled/total_samples*100:.1f}%) | "
            f"Unlabeled: {total_unlabeled} ({total_unlabeled/total_samples*100:.1f}%)"
        )


# ============================================================================
# SUBTASK 14.2: Test accuracy progression line chart
# ============================================================================

def render_accuracy_progression_chart(controller: Controller) -> None:
    """
    Render test accuracy progression line chart.
    
    Shows how test accuracy improves across AL cycles.
    
    Args:
        controller: Controller instance
    """
    st.markdown("### 📈 Test Accuracy Progression")
    
    # Get metrics history from controller
    metrics_history = controller.metrics_history
    
    if not metrics_history:
        st.info("📊 Accuracy chart will appear after cycles complete")
        return
    
    # Prepare chart data
    chart_data = {
        "Cycle": [m.get("cycle", i+1) for i, m in enumerate(metrics_history)],
        "Test Accuracy": [m.get("test_accuracy", 0) * 100 for m in metrics_history],
        "Val Accuracy": [m.get("best_val_accuracy", 0) * 100 for m in metrics_history],
    }
    
    # Create DataFrame for chart
    df = pd.DataFrame(chart_data)
    
    # Display line chart
    st.line_chart(
        df,
        x="Cycle",
        y=["Test Accuracy", "Val Accuracy"],
        height=400
    )
    
    # Show improvement stats
    if len(metrics_history) >= 2:
        first_test_acc = metrics_history[0].get("test_accuracy", 0) * 100
        last_test_acc = metrics_history[-1].get("test_accuracy", 0) * 100
        improvement = last_test_acc - first_test_acc
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Initial Test Acc",
                value=f"{first_test_acc:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Final Test Acc",
                value=f"{last_test_acc:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Improvement",
                value=f"+{improvement:.2f}%",
                delta=f"{improvement:.2f}%"
            )


# ============================================================================
# SUBTASK 14.3: Best cycle highlight with summary
# ============================================================================

def render_best_cycle_summary(controller: Controller) -> None:
    """
    Render best cycle highlight with summary.
    
    Identifies and highlights the cycle with the best test accuracy,
    showing key metrics and insights.
    
    Args:
        controller: Controller instance
    """
    st.markdown("### 🏆 Best Cycle Summary")
    
    # Get metrics history from controller
    metrics_history = controller.metrics_history
    
    if not metrics_history:
        st.info("🎯 Best cycle summary will appear after cycles complete")
        return
    
    # Find best cycle by test accuracy
    best_cycle_idx = 0
    best_test_acc = 0.0
    
    for i, metrics in enumerate(metrics_history):
        test_acc = metrics.get("test_accuracy", 0)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_cycle_idx = i
    
    best_metrics = metrics_history[best_cycle_idx]
    
    # Display best cycle info in a highlighted container
    st.success(f"🏆 **Best Performance: Cycle {best_metrics.get('cycle', best_cycle_idx + 1)}**")
    
    # Create columns for key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{best_metrics.get('test_accuracy', 0) * 100:.2f}%"
        )
    
    with col2:
        st.metric(
            label="F1 Score",
            value=f"{best_metrics.get('test_f1', 0):.3f}"
        )
    
    with col3:
        st.metric(
            label="Precision",
            value=f"{best_metrics.get('test_precision', 0):.3f}"
        )
    
    with col4:
        st.metric(
            label="Recall",
            value=f"{best_metrics.get('test_recall', 0):.3f}"
        )
    
    # Additional details
    st.markdown("#### 📋 Cycle Details")
    
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        st.write(f"**Labeled Pool Size:** {best_metrics.get('labeled_pool_size', 0)}")
    
    with detail_col2:
        st.write(f"**Epochs Trained:** {best_metrics.get('epochs_trained', 0)}")
    
    with detail_col3:
        st.write(f"**Best Epoch:** {best_metrics.get('best_epoch', 0)}")
    
    # Calculate improvement from first cycle
    if len(metrics_history) > 1:
        first_test_acc = metrics_history[0].get("test_accuracy", 0) * 100
        best_test_acc_pct = best_test_acc * 100
        improvement = best_test_acc_pct - first_test_acc
        
        st.info(
            f"📈 **Improvement:** +{improvement:.2f}% from initial cycle "
            f"({first_test_acc:.2f}% → {best_test_acc_pct:.2f}%)"
        )


# ============================================================================
# MAIN RESULTS VIEW RENDER FUNCTION
# ============================================================================

def render_results_view(controller: Controller) -> None:
    """
    Main render function for results dashboard.
    
    This view displays comprehensive results after AL cycles complete.
    Can be shown alongside other views or as a dedicated results page.
    
    Displays:
    - Test accuracy progression line chart
    - Cycle-by-cycle metrics table
    - Best cycle highlight with summary
    
    Args:
        controller: Controller instance
    """
    # Header
    st.title("📈 Results Dashboard")
    st.markdown("---")
    
    # Check if we have any results
    if not controller.metrics_history:
        st.info(
            "📊 Results will appear here after completing Active Learning cycles.\n\n"
            "Start an experiment from the sidebar to begin!"
        )
        return
    
    # Render test accuracy progression chart (Task 14.2)
    render_accuracy_progression_chart(controller)
    
    st.markdown("---")
    
    # Render metrics table (Task 14.1)
    render_metrics_table(controller)
    
    st.markdown("---")
    
    # Render best cycle summary (Task 14.3)
    render_best_cycle_summary(controller)
    
    st.markdown("---")
    
    # Export button (placeholder for future enhancement)
    if st.button("📥 Export Results (JSON)"):
        st.info("Export functionality coming soon!")
        # TODO: Implement JSON export in Phase 7 (Task 19.4)
