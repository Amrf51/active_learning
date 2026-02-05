"""
Results & Analysis Page - Comprehensive results visualization

This page provides:
- Performance over cycles visualization
- Cycle metrics table with detailed statistics
- Multi-experiment comparison
- Per-class performance metrics
- Confusion matrix visualization with filtering
- Results export functionality
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import logging
import json
from typing import List, Dict, Optional, Any, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NEW: MVC imports
from pages.controller_factory import get_controller, update_session_heartbeat
from model.schemas import ExperimentPhase, CycleSummary, ExperimentStatus

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Results & Analysis - AL Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS matching the dashboard theme
st.markdown("""
    <style>
    /* Results page specific styling */
    .results-section {
        background-color: #112240;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        color: #e6f1ff;
    }
    
    .metric-highlight {
        background-color: #0f3d3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #20c997;
        color: #e6f1ff;
    }
    
    .export-section {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #4a4a6a;
        color: #e6f1ff;
    }
    
    .comparison-card {
        background-color: #112240;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #233554;
        margin: 0.5rem 0;
        color: #e6f1ff;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for results page."""
    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = None

    if "selected_experiment_data" not in st.session_state:
        st.session_state.selected_experiment_data = None

    if "comparison_experiments" not in st.session_state:
        st.session_state.comparison_experiments = []


def display_experiment_selector() -> Optional[Tuple[str, List[CycleSummary]]]:
    """Display experiment selection for results analysis.

    Returns:
        Tuple of (experiment_id, cycle_summaries) or None if no experiment selected
    """
    st.sidebar.header("📊 Results Analysis")

    # Get controller to access database
    ctrl = get_controller()
    db_manager = ctrl._model_handler._db_manager

    # Get all experiments from database
    try:
        experiments = db_manager.list_experiments()
    except Exception as e:
        st.sidebar.error(f"Failed to load experiments: {e}")
        return None

    if not experiments:
        st.sidebar.info("No experiments found.")
        return None

    # Filter experiments that have results (completed cycles)
    experiments_with_results = []
    for exp in experiments:
        try:
            cycle_summaries = db_manager.get_cycle_summaries(exp["experiment_id"])
            if cycle_summaries:  # Has completed cycles
                experiments_with_results.append({
                    "experiment_id": exp["experiment_id"],
                    "experiment_name": exp.get("experiment_name", exp["experiment_id"]),
                    "created_at": exp.get("created_at")
                })
        except Exception:
            continue

    if not experiments_with_results:
        st.sidebar.warning("No experiments with results found.")
        return None

    # Experiment selection
    exp_options = ["None"] + [
        f"{exp['experiment_name']} ({exp['experiment_id'][:8]}...)"
        for exp in experiments_with_results
    ]
    exp_ids = ["None"] + [exp["experiment_id"] for exp in experiments_with_results]

    current_selection = st.session_state.selected_experiment
    if current_selection not in exp_ids:
        current_selection = "None"

    selected_idx = st.sidebar.selectbox(
        "Select Experiment",
        range(len(exp_options)),
        format_func=lambda i: exp_options[i],
        index=exp_ids.index(current_selection) if current_selection in exp_ids else 0,
        help="Choose an experiment to analyze"
    )

    selected_id = exp_ids[selected_idx]

    if selected_id != "None" and selected_id != st.session_state.selected_experiment:
        st.session_state.selected_experiment = selected_id
        st.rerun()

    if selected_id == "None":
        return None

    # Get cycle summaries for selected experiment
    try:
        cycle_summaries = db_manager.get_cycle_summaries(selected_id)
        return (selected_id, cycle_summaries)
    except Exception as e:
        st.sidebar.error(f"Failed to load results: {e}")
        return None


def display_export_section(state):
    """Display export section with CSV download functionality."""
    st.markdown('<div class="export-section">', unsafe_allow_html=True)
    st.subheader("📥 Export Results")
    
    if not state.cycle_results:
        st.warning("No results to export.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Prepare cycle metrics data for export
    export_data = []
    for cycle_result in state.cycle_results:
        row = {
            'experiment_id': state.experiment_id,
            'experiment_name': state.experiment_name,
            'cycle': cycle_result.cycle,
            'labeled_pool_size': cycle_result.labeled_count,
            'unlabeled_pool_size': cycle_result.unlabeled_count,
            'epochs_trained': cycle_result.epochs_trained,
            'best_val_accuracy': cycle_result.best_val_accuracy,
            'best_epoch': cycle_result.best_epoch,
            'test_accuracy': cycle_result.test_accuracy,
            'test_precision': cycle_result.test_precision,
            'test_recall': cycle_result.test_recall,
            'test_f1': cycle_result.test_f1,
        }
        
        # Add annotation accuracy if available
        if cycle_result.annotation_accuracy is not None:
            row['annotation_accuracy'] = cycle_result.annotation_accuracy
        
        # Add config info
        if state.config:
            row['model_name'] = state.config.model_name
            row['sampling_strategy'] = state.config.sampling_strategy
            row['learning_rate'] = state.config.learning_rate
            row['batch_size'] = state.config.batch_size
            row['epochs_per_cycle'] = state.config.epochs_per_cycle
            row['initial_pool_size'] = state.config.initial_pool_size
            row['batch_size_al'] = state.config.batch_size_al
            row['reset_mode'] = state.config.reset_mode
            row['seed'] = state.config.seed
        
        export_data.append(row)
    
    export_df = pd.DataFrame(export_data)
    
    # Preview of export data
    st.markdown("#### 📋 Export Preview")
    
    with st.expander("View export data", expanded=False):
        st.dataframe(export_df, use_container_width=True, height=200)
    
    # Export information
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Export Summary:**")
        st.write(f"• Experiment: `{state.experiment_id}`")
        st.write(f"• Cycles: {len(state.cycle_results)}")
        st.write(f"• Metrics included: {len(export_df.columns)} columns")
        
        if state.config:
            st.write(f"• Strategy: {state.config.sampling_strategy}")
            st.write(f"• Model: {state.config.model_name}")
    
    # CSV Export
    with col2:
        csv_data = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"{state.experiment_id}_results.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
            help="Download cycle metrics as CSV file"
        )
    
    # JSON Export (full state)
    with col3:
        # Prepare comprehensive JSON export
        json_export = {
            'experiment_id': state.experiment_id,
            'experiment_name': state.experiment_name,
            'created_at': state.created_at.isoformat() if state.created_at else None,
            'phase': state.phase.value if state.phase else None,
            'total_cycles': state.total_cycles,
            'current_cycle': state.current_cycle,
            'labeled_count': state.labeled_count,
            'unlabeled_count': state.unlabeled_count,
            'config': state.config.model_dump() if state.config else None,
            'cycle_results': [cr.model_dump() for cr in state.cycle_results],
            'dataset_info': state.dataset_info.model_dump() if state.dataset_info else None
        }
        
        json_data = json.dumps(json_export, indent=2, default=str)
        
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"{state.experiment_id}_full_results.json",
            mime="application/json",
            use_container_width=True,
            help="Download complete experiment data as JSON"
        )
    
    # Per-class metrics export if available
    if state.cycle_results and any(cr.per_class_metrics for cr in state.cycle_results):
        st.markdown("---")
        st.markdown("#### 🎯 Per-Class Metrics Export")
        
        # Get latest cycle with per-class metrics
        latest_with_per_class = None
        for cr in reversed(state.cycle_results):
            if cr.per_class_metrics:
                latest_with_per_class = cr
                break
        
        if latest_with_per_class and latest_with_per_class.per_class_metrics:
            # Prepare per-class data
            per_class_data = []
            for class_name, metrics in latest_with_per_class.per_class_metrics.items():
                per_class_data.append({
                    'class': class_name,
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0),
                    'f1_score': metrics.get('f1', metrics.get('f1-score', 0.0)),
                    'support': metrics.get('support', 0)
                })
            
            per_class_df = pd.DataFrame(per_class_data)
            per_class_csv = per_class_df.to_csv(index=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"Per-class metrics from Cycle {latest_with_per_class.cycle}")
                st.write(f"• {len(per_class_data)} classes")
            
            with col2:
                st.download_button(
                    label="📊 Download Per-Class CSV",
                    data=per_class_csv,
                    file_name=f"{state.experiment_id}_per_class_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download per-class metrics"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)



def main():
    """Main results page."""
    # Update heartbeat to maintain session
    update_session_heartbeat()

    initialize_session_state()

    st.title("📊 Results & Analysis")
    st.markdown("Analyze and compare Active Learning experiment results.")

    # Experiment selection - returns (experiment_id, cycle_summaries) or None
    result = display_experiment_selector()

    if not result:
        st.info("👈 Select an experiment from the sidebar to view results.")
        return

    experiment_id, cycle_summaries = result

    if not cycle_summaries:
        st.warning("No completed cycles found for this experiment.")
        return

    try:
        # Get full experiment details including config
        ctrl = get_controller()
        db_manager = ctrl._model_handler._db_manager
        experiment_details = db_manager.get_experiment(experiment_id)

        # Create a simple data structure to pass to display functions
        # This mimics the old state object but uses CycleSummary list
        class ExperimentData:
            def __init__(self, exp_id: str, exp_name: str, cycles: List[CycleSummary], 
                        config, created_at, phase, total_cycles, current_cycle, 
                        labeled_count, unlabeled_count, dataset_info):
                self.experiment_id = exp_id
                self.experiment_name = exp_name
                self.cycle_results = cycles
                self.config = config
                self.created_at = created_at
                self.phase = phase
                self.total_cycles = total_cycles
                self.current_cycle = current_cycle
                self.labeled_count = labeled_count
                self.unlabeled_count = unlabeled_count
                self.dataset_info = dataset_info

        # Parse created_at timestamp
        created_at = None
        if experiment_details and experiment_details.get("created_at"):
            try:
                created_at = datetime.fromisoformat(experiment_details["created_at"])
            except:
                created_at = None

        # Get additional experiment info
        phase = None
        total_cycles = 0
        current_cycle = len(cycle_summaries)
        labeled_count = cycle_summaries[-1].labeled_count if cycle_summaries else 0
        unlabeled_count = cycle_summaries[-1].unlabeled_count if cycle_summaries else 0
        
        # Parse config if available
        config_obj = None
        if experiment_details and experiment_details.get("config"):
            from model.schemas import ExperimentConfig
            try:
                config_dict = experiment_details["config"]
                if isinstance(config_dict, str):
                    config_dict = json.loads(config_dict)
                config_obj = ExperimentConfig.from_dict(config_dict)
                total_cycles = config_obj.num_cycles
            except Exception as e:
                logger.warning(f"Failed to parse config: {e}")

        experiment_data = ExperimentData(
            experiment_id,
            experiment_details.get("experiment_name", experiment_id) if experiment_details else experiment_id,
            cycle_summaries,
            config_obj,
            created_at,
            phase,
            total_cycles,
            current_cycle,
            labeled_count,
            unlabeled_count,
            None  # dataset_info not available from DB
        )

        # Create tabs for different result views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance",
            "📋 Metrics Table",
            "🔄 Comparison",
            "🎯 Per-Class",
            "🔥 Confusion Matrix"
        ])

        with tab1:
            display_performance_visualization(experiment_data)

        with tab2:
            display_metrics_table(experiment_data)

        with tab3:
            display_multi_experiment_comparison()

        with tab4:
            display_per_class_metrics(experiment_data)

        with tab5:
            display_confusion_matrix(experiment_data)

        # Export section at the bottom
        st.markdown("---")
        display_export_section(experiment_data)

    except Exception as e:
        st.error(f"❌ Error loading results: {str(e)}")
        logger.error(f"Results loading failed: {e}")


if __name__ == "__main__":
    main()


def display_performance_visualization(state):
    """Display performance over cycles visualization."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("📈 Performance Over Cycles")
    
    if not state.cycle_results:
        st.warning("No cycle results available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Extract data for visualization
    cycles = []
    labeled_samples = []
    test_accuracies = []
    test_f1_scores = []
    val_accuracies = []
    
    for cycle_result in state.cycle_results:
        cycles.append(cycle_result.cycle)
        labeled_samples.append(cycle_result.labeled_count)
        test_accuracies.append(cycle_result.test_accuracy)
        test_f1_scores.append(cycle_result.test_f1)
        val_accuracies.append(cycle_result.best_val_accuracy)
    
    # Create performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Test Accuracy vs Labeled Samples
        fig_acc = go.Figure()
        
        fig_acc.add_trace(go.Scatter(
            x=labeled_samples,
            y=test_accuracies,
            mode='lines+markers',
            name='Test Accuracy',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, color='#1f77b4')
        ))
        
        fig_acc.add_trace(go.Scatter(
            x=labeled_samples,
            y=val_accuracies,
            mode='lines+markers',
            name='Best Val Accuracy',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, color='#ff7f0e')
        ))
        
        fig_acc.update_layout(
            title="Accuracy vs. Labeled Samples",
            xaxis_title="Number of Labeled Samples",
            yaxis_title="Accuracy",
            template="plotly_dark",
            height=400,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # F1 Score vs Cycles
        fig_f1 = go.Figure()
        
        fig_f1.add_trace(go.Scatter(
            x=cycles,
            y=test_f1_scores,
            mode='lines+markers',
            name='Test F1 Score',
            line=dict(color='#2ca02c', width=3),
            marker=dict(size=8, color='#2ca02c')
        ))
        
        fig_f1.update_layout(
            title="F1 Score Over Cycles",
            xaxis_title="Cycle Number",
            yaxis_title="F1 Score",
            template="plotly_dark",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Performance summary metrics
    st.markdown("#### 📊 Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_acc = max(test_accuracies)
        best_cycle = cycles[test_accuracies.index(best_acc)]
        st.metric(
            "Best Test Accuracy", 
            f"{best_acc:.3f}",
            help=f"Achieved in cycle {best_cycle}"
        )
    
    with col2:
        best_f1 = max(test_f1_scores)
        best_f1_cycle = cycles[test_f1_scores.index(best_f1)]
        st.metric(
            "Best F1 Score", 
            f"{best_f1:.3f}",
            help=f"Achieved in cycle {best_f1_cycle}"
        )
    
    with col3:
        final_acc = test_accuracies[-1]
        initial_acc = test_accuracies[0] if test_accuracies else 0
        improvement = final_acc - initial_acc
        st.metric(
            "Total Improvement", 
            f"{improvement:+.3f}",
            delta=f"{improvement:.3f}"
        )
    
    with col4:
        total_labeled = labeled_samples[-1] if labeled_samples else 0
        st.metric(
            "Total Labeled", 
            f"{total_labeled:,}",
            help="Final number of labeled samples"
        )
    
    # Learning efficiency analysis
    if len(cycles) > 1:
        st.markdown("#### 🎯 Learning Efficiency")
        
        # Calculate accuracy gain per sample
        sample_gains = []
        for i in range(1, len(cycles)):
            acc_gain = test_accuracies[i] - test_accuracies[i-1]
            sample_gain = labeled_samples[i] - labeled_samples[i-1]
            if sample_gain > 0:
                efficiency = acc_gain / sample_gain
                sample_gains.append(efficiency)
        
        if sample_gains:
            avg_efficiency = np.mean(sample_gains)
            st.info(f"📈 **Average Learning Efficiency:** {avg_efficiency:.6f} accuracy gain per labeled sample")
            
            # Show efficiency trend
            efficiency_cycles = cycles[1:len(sample_gains)+1]
            
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Scatter(
                x=efficiency_cycles,
                y=sample_gains,
                mode='lines+markers',
                name='Efficiency',
                line=dict(color='#d62728', width=2),
                marker=dict(size=6, color='#d62728')
            ))
            
            fig_eff.update_layout(
                title="Learning Efficiency Over Cycles",
                xaxis_title="Cycle Number",
                yaxis_title="Accuracy Gain per Sample",
                template="plotly_dark",
                height=300
            )
            
            st.plotly_chart(fig_eff, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_metrics_table(state):
    """Display detailed cycle metrics table."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("📋 Detailed Cycle Metrics")
    
    if not state.cycle_results:
        st.warning("No cycle results available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Create comprehensive metrics DataFrame
    metrics_data = []
    
    for cycle_result in state.cycle_results:
        row = {
            'Cycle': cycle_result.cycle,
            'Labeled Pool': cycle_result.labeled_count,
            'Unlabeled Pool': cycle_result.unlabeled_count,
            'Epochs Trained': cycle_result.epochs_trained,
            'Best Val Acc': cycle_result.best_val_accuracy,
            'Best Epoch': cycle_result.best_epoch,
            'Test Accuracy': cycle_result.test_accuracy,
            'Test Precision': cycle_result.test_precision,
            'Test Recall': cycle_result.test_recall,
            'Test F1': cycle_result.test_f1,
        }
        
        # Add annotation accuracy if available
        if cycle_result.annotation_accuracy is not None:
            row['Annotation Acc'] = cycle_result.annotation_accuracy
        
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Summary Statistics**")
        st.write(f"• Total Cycles: {len(df)}")
        st.write(f"• Avg Test Accuracy: {df['Test Accuracy'].mean():.3f}")
        st.write(f"• Std Test Accuracy: {df['Test Accuracy'].std():.3f}")
    
    with col2:
        st.markdown("**🎯 Best Performance**")
        best_idx = df['Test Accuracy'].idxmax()
        st.write(f"• Best Cycle: {df.loc[best_idx, 'Cycle']}")
        st.write(f"• Best Accuracy: {df.loc[best_idx, 'Test Accuracy']:.3f}")
        st.write(f"• Best F1: {df.loc[best_idx, 'Test F1']:.3f}")
    
    with col3:
        st.markdown("**📈 Progress**")
        first_acc = df['Test Accuracy'].iloc[0]
        last_acc = df['Test Accuracy'].iloc[-1]
        improvement = last_acc - first_acc
        st.write(f"• Initial Accuracy: {first_acc:.3f}")
        st.write(f"• Final Accuracy: {last_acc:.3f}")
        st.write(f"• Total Improvement: {improvement:+.3f}")
    
    # Format DataFrame for display
    display_df = df.copy()
    
    # Format numeric columns
    numeric_cols = ['Best Val Acc', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1']
    if 'Annotation Acc' in display_df.columns:
        numeric_cols.append('Annotation Acc')
    
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    
    # Add formatting for pool sizes
    display_df['Labeled Pool'] = display_df['Labeled Pool'].apply(lambda x: f"{x:,}")
    display_df['Unlabeled Pool'] = display_df['Unlabeled Pool'].apply(lambda x: f"{x:,}")
    
    # Display the table
    st.markdown("#### 📊 Complete Metrics Table")
    
    # Add sorting options
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            options=['Cycle', 'Test Accuracy', 'Test F1', 'Best Val Acc'],
            help="Choose column to sort by"
        )
        
        ascending = st.checkbox("Ascending", value=True)
    
    # Sort DataFrame
    if sort_by in df.columns:
        sorted_indices = df[sort_by].argsort()
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        display_df = display_df.iloc[sorted_indices].reset_index(drop=True)
    
    # Display with highlighting
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )
    
    # Detailed epoch history for selected cycle
    st.markdown("#### 🔍 Epoch-Level Details")
    st.info("📊 Epoch-level training details will be available in a future update. Currently showing cycle-level summaries only.")
    
    st.markdown('</div>', unsafe_allow_html=True)
def display_multi_experiment_comparison():
    """Display multi-experiment comparison interface."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("🔄 Multi-Experiment Comparison")

    # Get controller to access database
    ctrl = get_controller()
    db_manager = ctrl._model_handler._db_manager

    # Get all experiments with results
    try:
        experiments = db_manager.list_experiments()
    except Exception as e:
        st.error(f"Failed to load experiments: {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    experiments_with_results = []

    for exp in experiments:
        try:
            exp_id = exp["experiment_id"]
            cycle_summaries = db_manager.get_cycle_summaries(exp_id)

            if cycle_summaries:
                # Get full experiment details including config
                experiment_details = db_manager.get_experiment(exp_id)

                # Create a simple data structure
                class ExperimentData:
                    def __init__(self, exp_id: str, exp_name: str, cycles: List[CycleSummary], 
                                config, created_at, phase, total_cycles, current_cycle, 
                                labeled_count, unlabeled_count, dataset_info):
                        self.experiment_id = exp_id
                        self.experiment_name = exp_name
                        self.cycle_results = cycles
                        self.config = config
                        self.created_at = created_at
                        self.phase = phase
                        self.total_cycles = total_cycles
                        self.current_cycle = current_cycle
                        self.labeled_count = labeled_count
                        self.unlabeled_count = unlabeled_count
                        self.dataset_info = dataset_info

                # Parse created_at timestamp
                created_at = None
                if experiment_details and experiment_details.get("created_at"):
                    try:
                        created_at = datetime.fromisoformat(experiment_details["created_at"])
                    except:
                        created_at = None

                # Get additional experiment info
                phase = None
                total_cycles = 0
                current_cycle = len(cycle_summaries)
                labeled_count = cycle_summaries[-1].labeled_count if cycle_summaries else 0
                unlabeled_count = cycle_summaries[-1].unlabeled_count if cycle_summaries else 0
                
                # Parse config if available
                config_obj = None
                if experiment_details and experiment_details.get("config"):
                    from model.schemas import ExperimentConfig
                    try:
                        config_dict = experiment_details["config"]
                        if isinstance(config_dict, str):
                            config_dict = json.loads(config_dict)
                        config_obj = ExperimentConfig.from_dict(config_dict)
                        total_cycles = config_obj.num_cycles
                    except Exception as e:
                        logger.warning(f"Failed to parse config: {e}")

                exp_data = ExperimentData(
                    exp_id,
                    experiment_details.get("experiment_name", exp_id) if experiment_details else exp_id,
                    cycle_summaries,
                    config_obj,
                    created_at,
                    phase,
                    total_cycles,
                    current_cycle,
                    labeled_count,
                    unlabeled_count,
                    None  # dataset_info not available from DB
                )

                experiments_with_results.append({
                    'id': exp_id,
                    'name': exp_data.experiment_name,
                    'state': exp_data,
                    'config': exp_data.config
                })
        except Exception as e:
            logger.warning(f"Could not load experiment {exp['experiment_id']}: {e}")
            continue
    
    if len(experiments_with_results) < 2:
        st.warning("Need at least 2 experiments with results for comparison.")
        st.info("Run more experiments to enable comparison features.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Experiment selection for comparison
    st.markdown("#### 📋 Select Experiments to Compare")
    
    exp_options = [f"{exp['id']} ({exp['name']})" for exp in experiments_with_results]
    
    selected_experiments = st.multiselect(
        "Choose experiments:",
        options=exp_options,
        default=exp_options[:min(3, len(exp_options))],  # Default to first 3
        help="Select 2-5 experiments to compare"
    )
    
    if len(selected_experiments) < 2:
        st.warning("Please select at least 2 experiments for comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Extract selected experiment data
    selected_exp_data = []
    for sel_exp_str in selected_experiments:
        exp_id = sel_exp_str.split(' (')[0]
        for exp in experiments_with_results:
            if exp['id'] == exp_id:
                selected_exp_data.append(exp)
                break
    
    # Create comparison visualization
    st.markdown("#### 📈 Performance Comparison")
    
    # Prepare data for plotting
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, exp_data in enumerate(selected_exp_data):
        state = exp_data['state']
        
        # Extract performance data
        labeled_samples = [cr.labeled_count for cr in state.cycle_results]
        test_accuracies = [cr.test_accuracy for cr in state.cycle_results]
        
        # Create experiment label
        exp_label = f"{exp_data['id']}"
        if exp_data['config']:
            strategy = exp_data['config'].sampling_strategy
            model = exp_data['config'].model_name
            exp_label += f" ({strategy}, {model})"
        
        fig.add_trace(go.Scatter(
            x=labeled_samples,
            y=test_accuracies,
            mode='lines+markers',
            name=exp_label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=6, color=colors[i % len(colors)])
        ))
    
    fig.update_layout(
        title="Test Accuracy vs. Labeled Samples - Multi-Experiment Comparison",
        xaxis_title="Number of Labeled Samples",
        yaxis_title="Test Accuracy",
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Comparison table
    st.markdown("#### 📊 Comparison Summary")
    
    comparison_data = []
    for exp_data in selected_exp_data:
        state = exp_data['state']
        config = exp_data['config']
        
        # Calculate summary statistics
        test_accs = [cr.test_accuracy for cr in state.cycle_results]
        test_f1s = [cr.test_f1 for cr in state.cycle_results]
        
        row = {
            'Experiment': exp_data['id'],
            'Strategy': config.sampling_strategy if config else 'Unknown',
            'Model': config.model_name if config else 'Unknown',
            'Cycles': len(state.cycle_results),
            'Best Accuracy': max(test_accs),
            'Final Accuracy': test_accs[-1],
            'Avg Accuracy': np.mean(test_accs),
            'Best F1': max(test_f1s),
            'Final F1': test_f1s[-1],
            'Total Labeled': state.cycle_results[-1].labeled_count if state.cycle_results else 0
        }
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Format for display
    display_comparison_df = comparison_df.copy()
    numeric_cols = ['Best Accuracy', 'Final Accuracy', 'Avg Accuracy', 'Best F1', 'Final F1']
    for col in numeric_cols:
        display_comparison_df[col] = display_comparison_df[col].apply(lambda x: f"{x:.3f}")
    
    display_comparison_df['Total Labeled'] = display_comparison_df['Total Labeled'].apply(lambda x: f"{x:,}")
    
    st.dataframe(display_comparison_df, use_container_width=True)
    
    # Statistical analysis
    st.markdown("#### 📊 Statistical Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Best Performers**")
        
        # Best accuracy
        best_acc_idx = comparison_df['Best Accuracy'].idxmax()
        best_acc_exp = comparison_df.loc[best_acc_idx]
        st.write(f"• **Best Accuracy:** {best_acc_exp['Experiment']}")
        st.write(f"  - Strategy: {best_acc_exp['Strategy']}")
        st.write(f"  - Accuracy: {best_acc_exp['Best Accuracy']:.3f}")
        
        # Best final performance
        best_final_idx = comparison_df['Final Accuracy'].idxmax()
        best_final_exp = comparison_df.loc[best_final_idx]
        st.write(f"• **Best Final:** {best_final_exp['Experiment']}")
        st.write(f"  - Strategy: {best_final_exp['Strategy']}")
        st.write(f"  - Accuracy: {best_final_exp['Final Accuracy']:.3f}")
    
    with col2:
        st.markdown("**📈 Efficiency Analysis**")
        
        # Calculate efficiency (accuracy per sample)
        for i, row in comparison_df.iterrows():
            total_labeled = row['Total Labeled']
            if isinstance(total_labeled, str):
                total_labeled = int(total_labeled.replace(',', ''))
            
            efficiency = row['Final Accuracy'] / total_labeled if total_labeled > 0 else 0
            comparison_df.loc[i, 'Efficiency'] = efficiency
        
        # Most efficient
        most_eff_idx = comparison_df['Efficiency'].idxmax()
        most_eff_exp = comparison_df.loc[most_eff_idx]
        st.write(f"• **Most Efficient:** {most_eff_exp['Experiment']}")
        st.write(f"  - Strategy: {most_eff_exp['Strategy']}")
        st.write(f"  - Efficiency: {most_eff_exp['Efficiency']:.6f}")
        
        # Strategy comparison
        strategy_performance = comparison_df.groupby('Strategy')['Final Accuracy'].mean().sort_values(ascending=False)
        st.write("• **Strategy Ranking:**")
        for strategy, avg_acc in strategy_performance.items():
            st.write(f"  - {strategy}: {avg_acc:.3f}")
    
    # Detailed cycle-by-cycle comparison
    with st.expander("🔍 Detailed Cycle-by-Cycle Comparison", expanded=False):
        
        # Create detailed comparison table
        max_cycles = max(len(exp['state'].cycle_results) for exp in selected_exp_data)
        
        detailed_data = []
        for cycle in range(1, max_cycles + 1):
            row = {'Cycle': cycle}
            
            for exp_data in selected_exp_data:
                exp_id = exp_data['id']
                state = exp_data['state']
                
                if cycle <= len(state.cycle_results):
                    cycle_result = state.cycle_results[cycle - 1]
                    row[f"{exp_id}_Acc"] = f"{cycle_result.test_accuracy:.3f}"
                    row[f"{exp_id}_F1"] = f"{cycle_result.test_f1:.3f}"
                    row[f"{exp_id}_Labeled"] = f"{cycle_result.labeled_count:,}"
                else:
                    row[f"{exp_id}_Acc"] = "N/A"
                    row[f"{exp_id}_F1"] = "N/A"
                    row[f"{exp_id}_Labeled"] = "N/A"
            
            detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        st.dataframe(detailed_df, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
def display_per_class_metrics(state):
    """Display per-class performance metrics."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("🎯 Per-Class Performance Analysis")
    
    if not state.cycle_results:
        st.warning("No cycle results available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Find cycles with per-class metrics
    cycles_with_per_class = []
    for cycle_result in state.cycle_results:
        if cycle_result.per_class_metrics:
            cycles_with_per_class.append(cycle_result)
    
    if not cycles_with_per_class:
        st.warning("No per-class metrics available. Per-class metrics are computed during evaluation.")
        st.info("💡 Per-class metrics will be available after running cycles with detailed evaluation.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Cycle selection for per-class analysis
    cycle_options = [f"Cycle {cr.cycle}" for cr in cycles_with_per_class]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_cycle_str = st.selectbox(
            "Select Cycle:",
            options=cycle_options,
            index=len(cycle_options) - 1,  # Default to latest cycle
            help="Choose cycle to analyze per-class performance"
        )
    
    if not selected_cycle_str:
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Get selected cycle data
    selected_cycle_num = int(selected_cycle_str.split()[1])
    selected_cycle_result = None
    
    for cr in cycles_with_per_class:
        if cr.cycle == selected_cycle_num:
            selected_cycle_result = cr
            break
    
    if not selected_cycle_result or not selected_cycle_result.per_class_metrics:
        st.error("Selected cycle has no per-class metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    per_class_data = selected_cycle_result.per_class_metrics
    
    # Create per-class metrics DataFrame
    class_metrics_data = []
    
    for class_name, metrics in per_class_data.items():
        row = {
            'Class': class_name,
            'Precision': metrics.get('precision', 0.0),
            'Recall': metrics.get('recall', 0.0),
            'F1-Score': metrics.get('f1-score', 0.0),
            'Support': int(metrics.get('support', 0))
        }
        class_metrics_data.append(row)
    
    class_df = pd.DataFrame(class_metrics_data)
    
    # Sort by F1-score (descending)
    class_df = class_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_precision = class_df['Precision'].mean()
        st.metric("Avg Precision", f"{avg_precision:.3f}")
    
    with col2:
        avg_recall = class_df['Recall'].mean()
        st.metric("Avg Recall", f"{avg_recall:.3f}")
    
    with col3:
        avg_f1 = class_df['F1-Score'].mean()
        st.metric("Avg F1-Score", f"{avg_f1:.3f}")
    
    with col4:
        total_support = class_df['Support'].sum()
        st.metric("Total Samples", f"{total_support:,}")
    
    # Performance distribution visualization
    st.markdown("#### 📊 Performance Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # F1-Score distribution
        fig_f1_dist = go.Figure()
        
        fig_f1_dist.add_trace(go.Histogram(
            x=class_df['F1-Score'],
            nbinsx=10,
            name='F1-Score Distribution',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        fig_f1_dist.update_layout(
            title="F1-Score Distribution Across Classes",
            xaxis_title="F1-Score",
            yaxis_title="Number of Classes",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_f1_dist, use_container_width=True)
    
    with col2:
        # Support vs Performance scatter
        fig_support = go.Figure()
        
        fig_support.add_trace(go.Scatter(
            x=class_df['Support'],
            y=class_df['F1-Score'],
            mode='markers',
            text=class_df['Class'],
            marker=dict(
                size=8,
                color=class_df['F1-Score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="F1-Score")
            ),
            name='Classes'
        ))
        
        fig_support.update_layout(
            title="F1-Score vs. Support (Sample Count)",
            xaxis_title="Support (Number of Samples)",
            yaxis_title="F1-Score",
            template="plotly_dark",
            height=300
        )
        
        st.plotly_chart(fig_support, use_container_width=True)
    
    # Detailed per-class table
    st.markdown("#### 📋 Detailed Per-Class Metrics")
    
    # Format DataFrame for display
    display_class_df = class_df.copy()
    display_class_df['Precision'] = display_class_df['Precision'].apply(lambda x: f"{x:.3f}")
    display_class_df['Recall'] = display_class_df['Recall'].apply(lambda x: f"{x:.3f}")
    display_class_df['F1-Score'] = display_class_df['F1-Score'].apply(lambda x: f"{x:.3f}")
    display_class_df['Support'] = display_class_df['Support'].apply(lambda x: f"{x:,}")
    
    # Add performance indicators
    def get_performance_indicator(f1_score_str):
        f1_score = float(f1_score_str)
        if f1_score >= 0.9:
            return "🟢 Excellent"
        elif f1_score >= 0.8:
            return "🟡 Good"
        elif f1_score >= 0.7:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    display_class_df['Performance'] = display_class_df['F1-Score'].apply(get_performance_indicator)
    
    # Reorder columns
    display_class_df = display_class_df[['Class', 'Performance', 'Precision', 'Recall', 'F1-Score', 'Support']]
    
    st.dataframe(display_class_df, use_container_width=True, height=400)
    
    # Performance analysis
    st.markdown("#### 🔍 Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏆 Best Performing Classes**")
        
        top_classes = class_df.nlargest(5, 'F1-Score')
        for _, row in top_classes.iterrows():
            st.write(f"• **{row['Class']}**: F1={row['F1-Score']:.3f} (Support: {row['Support']:,})")
    
    with col2:
        st.markdown("**⚠️ Challenging Classes**")
        
        bottom_classes = class_df.nsmallest(5, 'F1-Score')
        for _, row in bottom_classes.iterrows():
            st.write(f"• **{row['Class']}**: F1={row['F1-Score']:.3f} (Support: {row['Support']:,})")
    
    # Class imbalance analysis
    if len(class_df) > 1:
        st.markdown("#### ⚖️ Class Balance Analysis")
        
        support_values = class_df['Support'].values
        max_support = max(support_values)
        min_support = min(support_values)
        imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Imbalance Ratio", f"{imbalance_ratio:.1f}x")
        
        with col2:
            st.metric("Largest Class", f"{max_support:,}")
        
        with col3:
            st.metric("Smallest Class", f"{min_support:,}")
        
        if imbalance_ratio > 10:
            st.warning("⚠️ **High class imbalance detected!** Consider using class weights or balanced sampling.")
        elif imbalance_ratio > 3:
            st.info("ℹ️ **Moderate class imbalance.** Monitor per-class performance carefully.")
        else:
            st.success("✅ **Well-balanced dataset.** Good class distribution for training.")
    
    # Correlation analysis
    if len(class_df) > 2:
        st.markdown("#### 📈 Metric Correlations")
        
        # Calculate correlations
        corr_data = class_df[['Precision', 'Recall', 'F1-Score', 'Support']].corr()
        
        # Create correlation heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_data.values,
            x=corr_data.columns,
            y=corr_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_data.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Correlation Between Metrics",
            template="plotly_dark",
            height=400,
            width=400
        )
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("**📊 Correlation Insights:**")
            
            # Support vs F1 correlation
            support_f1_corr = corr_data.loc['Support', 'F1-Score']
            if support_f1_corr > 0.3:
                st.write("• 📈 **Positive correlation** between support and F1-score")
                st.write("  Classes with more samples tend to perform better")
            elif support_f1_corr < -0.3:
                st.write("• 📉 **Negative correlation** between support and F1-score")
                st.write("  Smaller classes may be performing better")
            else:
                st.write("• ➡️ **Weak correlation** between support and F1-score")
                st.write("  Performance is not strongly tied to sample count")
            
            # Precision vs Recall correlation
            prec_rec_corr = corr_data.loc['Precision', 'Recall']
            if prec_rec_corr > 0.5:
                st.write("• ✅ **Strong precision-recall correlation**")
                st.write("  Balanced performance across classes")
            else:
                st.write("• ⚖️ **Precision-recall trade-off observed**")
                st.write("  Some classes favor precision or recall")
    
    st.markdown('</div>', unsafe_allow_html=True)
def display_confusion_matrix(state):
    """Display confusion matrix visualization with lazy loading and filtering."""
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.subheader("🔥 Confusion Matrix Analysis")
    
    if not state.cycle_results:
        st.warning("No cycle results available.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Cycle selection
    cycle_options = [f"Cycle {cr.cycle}" for cr in state.cycle_results]
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_cycle_str = st.selectbox(
            "Select Cycle:",
            options=cycle_options,
            index=len(cycle_options) - 1,  # Default to latest cycle
            help="Choose cycle to view confusion matrix"
        )
    
    if not selected_cycle_str:
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    selected_cycle_num = int(selected_cycle_str.split()[1])
    
    # Check for confusion matrix file
    experiment_dir = Path("experiments") / state.experiment_id
    cm_file = experiment_dir / f"cycle_{selected_cycle_num}_confusion_matrix.npy"
    
    if not cm_file.exists():
        st.warning(f"⚠️ Confusion matrix not yet generated for Cycle {selected_cycle_num}")
        st.info("💡 Confusion matrices are saved during evaluation. Run the worker to generate them.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Lazy load confusion matrix
    try:
        with st.spinner("Loading confusion matrix..."):
            cm = np.load(cm_file)
            
        # Get class names
        class_names = state.config.class_names if state.config else [f"Class_{i}" for i in range(cm.shape[0])]
        
        if len(class_names) != cm.shape[0]:
            class_names = [f"Class_{i}" for i in range(cm.shape[0])]
        
        st.success(f"✅ Loaded confusion matrix for Cycle {selected_cycle_num}")
        
        # Display matrix info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Classes", cm.shape[0])
        
        with col2:
            st.metric("Total Predictions", cm.sum())
        
        with col3:
            correct_predictions = np.trace(cm)
            st.metric("Correct", f"{correct_predictions:,}")
        
        with col4:
            accuracy = correct_predictions / cm.sum() if cm.sum() > 0 else 0
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        # Handle high cardinality datasets
        num_classes = len(class_names)
        
        if num_classes > 50:
            st.info(f"📊 Dataset has {num_classes} classes. Showing filtered view for readability.")
            
            # Filtering options
            filter_option = st.radio(
                "Choose visualization:",
                options=[
                    "Top Confused Pairs",
                    "Class Subset Selection",
                    "Diagonal Analysis"
                ],
                help="Select how to display the large confusion matrix"
            )
            
            if filter_option == "Top Confused Pairs":
                display_top_confused_pairs(cm, class_names)
            
            elif filter_option == "Class Subset Selection":
                display_class_subset_matrix(cm, class_names)
            
            elif filter_option == "Diagonal Analysis":
                display_diagonal_analysis(cm, class_names)
        
        else:
            # Standard heatmap for smaller datasets
            display_full_confusion_matrix(cm, class_names, selected_cycle_num)
    
    except Exception as e:
        st.error(f"❌ Error loading confusion matrix: {str(e)}")
        logger.error(f"Confusion matrix loading failed: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_top_confused_pairs(cm: np.ndarray, class_names: List[str], top_k: int = 20):
    """Display top k most confused class pairs."""
    st.markdown("#### 🔍 Most Confused Class Pairs")
    
    # Find all off-diagonal pairs
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    'True Class': class_names[i],
                    'Predicted As': class_names[j],
                    'Count': int(cm[i, j]),
                    'True Class Support': int(cm[i, :].sum()),
                    'Error Rate': cm[i, j] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                })
    
    if not confused_pairs:
        st.info("No confusion found - perfect classification!")
        return
    
    # Sort by count (descending)
    confused_pairs.sort(key=lambda x: x['Count'], reverse=True)
    
    # Display top k pairs
    top_pairs = confused_pairs[:top_k]
    
    # Create DataFrame
    pairs_df = pd.DataFrame(top_pairs)
    
    # Format for display
    display_pairs_df = pairs_df.copy()
    display_pairs_df['Count'] = display_pairs_df['Count'].apply(lambda x: f"{x:,}")
    display_pairs_df['True Class Support'] = display_pairs_df['True Class Support'].apply(lambda x: f"{x:,}")
    display_pairs_df['Error Rate'] = display_pairs_df['Error Rate'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(display_pairs_df, use_container_width=True, height=400)
    
    # Visualization of top confusions
    if len(top_pairs) > 0:
        fig = go.Figure()
        
        # Create labels for pairs
        pair_labels = [f"{pair['True Class']} → {pair['Predicted As']}" for pair in top_pairs[:10]]
        pair_counts = [pair['Count'] for pair in top_pairs[:10]]
        
        fig.add_trace(go.Bar(
            x=pair_counts,
            y=pair_labels,
            orientation='h',
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="Top 10 Most Confused Class Pairs",
            xaxis_title="Number of Misclassifications",
            yaxis_title="Class Pairs (True → Predicted)",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_class_subset_matrix(cm: np.ndarray, class_names: List[str]):
    """Display confusion matrix for a user-selected subset of classes."""
    st.markdown("#### 🎯 Class Subset Confusion Matrix")
    
    # Class selection
    selected_classes = st.multiselect(
        "Select classes to display:",
        options=class_names,
        default=class_names[:min(10, len(class_names))],
        help="Choose up to 20 classes for detailed confusion matrix"
    )
    
    if not selected_classes:
        st.warning("Please select at least one class.")
        return
    
    if len(selected_classes) > 20:
        st.warning("Too many classes selected. Showing first 20.")
        selected_classes = selected_classes[:20]
    
    # Filter confusion matrix
    indices = [class_names.index(c) for c in selected_classes]
    cm_filtered = cm[np.ix_(indices, indices)]
    
    # Display filtered matrix
    display_full_confusion_matrix(cm_filtered, selected_classes, "Subset")


def display_diagonal_analysis(cm: np.ndarray, class_names: List[str]):
    """Display diagonal analysis for large confusion matrices."""
    st.markdown("#### 📊 Per-Class Performance Analysis")
    
    # Calculate per-class metrics
    class_metrics = []
    
    for i, class_name in enumerate(class_names):
        true_positives = cm[i, i]
        false_positives = cm[:, i].sum() - true_positives
        false_negatives = cm[i, :].sum() - true_positives
        
        support = cm[i, :].sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics.append({
            'Class': class_name,
            'True Positives': int(true_positives),
            'False Positives': int(false_positives),
            'False Negatives': int(false_negatives),
            'Support': int(support),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
    
    # Create DataFrame
    metrics_df = pd.DataFrame(class_metrics)
    
    # Sort by F1-Score
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_precision = metrics_df['Precision'].mean()
        st.metric("Avg Precision", f"{avg_precision:.3f}")
    
    with col2:
        avg_recall = metrics_df['Recall'].mean()
        st.metric("Avg Recall", f"{avg_recall:.3f}")
    
    with col3:
        avg_f1 = metrics_df['F1-Score'].mean()
        st.metric("Avg F1-Score", f"{avg_f1:.3f}")
    
    # Format for display
    display_metrics_df = metrics_df.copy()
    display_metrics_df['Precision'] = display_metrics_df['Precision'].apply(lambda x: f"{x:.3f}")
    display_metrics_df['Recall'] = display_metrics_df['Recall'].apply(lambda x: f"{x:.3f}")
    display_metrics_df['F1-Score'] = display_metrics_df['F1-Score'].apply(lambda x: f"{x:.3f}")
    
    # Add performance indicators
    def get_performance_color(f1_str):
        f1 = float(f1_str)
        if f1 >= 0.8:
            return "🟢"
        elif f1 >= 0.6:
            return "🟡"
        else:
            return "🔴"
    
    display_metrics_df['Status'] = display_metrics_df['F1-Score'].apply(get_performance_color)
    
    # Reorder columns
    display_metrics_df = display_metrics_df[['Status', 'Class', 'Precision', 'Recall', 'F1-Score', 'Support', 'True Positives', 'False Positives', 'False Negatives']]
    
    st.dataframe(display_metrics_df, use_container_width=True, height=400)


def display_full_confusion_matrix(cm: np.ndarray, class_names: List[str], cycle_info: str):
    """Display full confusion matrix heatmap."""
    st.markdown(f"#### 🔥 Confusion Matrix - {cycle_info}")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Count")
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {cycle_info}",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        template="plotly_dark",
        height=max(400, len(class_names) * 25),
        width=max(400, len(class_names) * 25)
    )
    
    # Rotate x-axis labels if many classes
    if len(class_names) > 10:
        fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Matrix statistics
    st.markdown("#### 📈 Matrix Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_samples = cm.sum()
        st.metric("Total Samples", f"{total_samples:,}")
    
    with col2:
        correct_predictions = np.trace(cm)
        st.metric("Correct Predictions", f"{correct_predictions:,}")
    
    with col3:
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        st.metric("Overall Accuracy", f"{accuracy:.3f}")
    
    with col4:
        off_diagonal = cm.sum() - np.trace(cm)
        st.metric("Misclassifications", f"{off_diagonal:,}")
    
    # Class-wise accuracy
    if len(class_names) <= 20:  # Only show for reasonable number of classes
        st.markdown("#### 🎯 Per-Class Accuracy")
        
        class_accuracies = []
        for i, class_name in enumerate(class_names):
            class_total = cm[i, :].sum()
            class_correct = cm[i, i]
            class_acc = class_correct / class_total if class_total > 0 else 0
            class_accuracies.append(class_acc)
        
        # Create bar chart
        fig_acc = go.Figure()
        
        fig_acc.add_trace(go.Bar(
            x=class_names,
            y=class_accuracies,
            marker_color='#2ca02c',
            text=[f"{acc:.2f}" for acc in class_accuracies],
            textposition='auto'
        ))
        
        fig_acc.update_layout(
            title="Per-Class Accuracy",
            xaxis_title="Class",
            yaxis_title="Accuracy",
            template="plotly_dark",
            height=400
        )
        
        if len(class_names) > 5:
            fig_acc.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_acc, use_container_width=True)