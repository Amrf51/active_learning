"""
Interactive Active Learning Dashboard - Main Entry Point

This is the main Streamlit application that provides an interactive interface
for controlling and monitoring Active Learning experiments. It replaces the
static results-only app.py with a full multi-page dashboard.

Usage:
    streamlit run dashboard.py

Pages:
    1. Configuration - Set up new AL experiments
    2. Active Learning - Control and monitor training
    3. Results - Analyze and compare results
"""

import streamlit as st
from pathlib import Path
import sys
import logging

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from state import ExperimentManager, StateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Active Learning Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* 🎨 MAIN BACKGROUND COLOR - Change this to modify the background */
    .main .block-container {
        background-color: #f0f8ff;  /* 👈 CHANGE THIS COLOR */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stApp {
        background-color: #f0f8ff;  /* 👈 CHANGE THIS COLOR TOO */
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-idle {
        color: #6c757d;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .experiment-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .nav-info {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #d1ecf1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "experiment_manager" not in st.session_state:
        experiments_dir = Path("experiments")
        st.session_state.experiment_manager = ExperimentManager(experiments_dir)
    
    if "selected_experiment" not in st.session_state:
        st.session_state.selected_experiment = None
    
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = None
    



def display_experiment_selector():
    """Display experiment selection interface."""
    st.sidebar.header("🔬 Experiment Selection")
    
    # Get all experiments
    experiments = st.session_state.experiment_manager.list_experiments()
    
    if not experiments:
        st.sidebar.info("No experiments found. Create one in the Configuration page.")
        return None
    
    # Display active experiment
    active = st.session_state.experiment_manager.get_active()
    if active:
        st.sidebar.success(f"🟢 Active: {active.experiment_id}")
    
    # Experiment selection
    exp_options = ["None"] + [exp["experiment_id"] for exp in experiments]
    
    current_selection = st.session_state.selected_experiment
    if current_selection not in exp_options:
        current_selection = "None"
    
    selected = st.sidebar.selectbox(
        "Select Experiment",
        exp_options,
        index=exp_options.index(current_selection) if current_selection in exp_options else 0,
        help="Choose an experiment to view or control"
    )
    
    if selected != "None" and selected != st.session_state.selected_experiment:
        st.session_state.selected_experiment = selected
        
        # Initialize state manager for selected experiment
        exp_dir = Path("experiments") / selected
        st.session_state.state_manager = StateManager(exp_dir)
        st.rerun()
    
    return selected if selected != "None" else None


def display_experiment_status():
    """Display current experiment status in sidebar."""
    if not st.session_state.selected_experiment or not st.session_state.state_manager:
        return
    
    try:
        state = st.session_state.state_manager.read_state()
        
        st.sidebar.subheader("📊 Experiment Status")
        
        # Phase status
        phase_colors = {
            "IDLE": "🔵",
            "TRAINING": "🟢", 
            "EVALUATING": "🟡",
            "QUERYING": "🟠",
            "AWAITING_ANNOTATION": "🟣",
            "COMPLETED": "✅",
            "ERROR": "🔴"
        }
        
        phase_icon = phase_colors.get(state.phase.value, "⚪")
        st.sidebar.write(f"**Phase:** {phase_icon} {state.phase.value}")
        
        # Cycle progress
        if state.total_cycles > 0:
            progress = state.current_cycle / state.total_cycles
            st.sidebar.progress(progress)
            st.sidebar.write(f"**Cycle:** {state.current_cycle}/{state.total_cycles}")
        
        # Pool sizes
        if state.labeled_count > 0 or state.unlabeled_count > 0:
            st.sidebar.write(f"**Labeled:** {state.labeled_count:,}")
            st.sidebar.write(f"**Unlabeled:** {state.unlabeled_count:,}")
        
        # Worker status
        if st.session_state.state_manager.is_worker_alive():
            st.sidebar.success("🟢 Worker Active")
        else:
            st.sidebar.warning("🟡 Worker Inactive")
        
        # Error message
        if state.error_message:
            st.sidebar.error(f"❌ {state.error_message}")
    
    except Exception as e:
        st.sidebar.error(f"❌ Error reading state: {str(e)}")


def display_navigation_info():
    """Display navigation information and tips."""
    st.markdown("""
    <div class="nav-info">
    <h4>🧭 Navigation Guide</h4>
    <ul>
        <li><strong>⚙️ Configuration:</strong> Set up new experiments with model and strategy selection</li>
        <li><strong>🎯 Active Learning:</strong> Control training cycles and monitor progress in real-time</li>
        <li><strong>📊 Results:</strong> Analyze performance, view confusion matrices, and export data</li>
    </ul>
    <p><em>💡 Tip: Select an experiment in the sidebar to view its status across all pages.</em></p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Active Learning Dashboard</h1>', unsafe_allow_html=True)
    
    # Display navigation info
    display_navigation_info()
    
    # Sidebar experiment selection
    selected_exp = display_experiment_selector()
    display_experiment_status()
    
    # Main content area
    if selected_exp:
        st.success(f"✅ Experiment **{selected_exp}** selected. Use the pages in the sidebar to configure, control, or analyze your experiment.")
    else:
        st.info("👈 Select an experiment from the sidebar or create a new one in the **Configuration** page.")
    
    # Quick stats if experiment is selected
    if selected_exp and st.session_state.state_manager:
        try:
            state = st.session_state.state_manager.read_state()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Cycle", f"{state.current_cycle}/{state.total_cycles}")
            
            with col2:
                st.metric("Labeled Samples", f"{state.labeled_count:,}")
            
            with col3:
                st.metric("Unlabeled Samples", f"{state.unlabeled_count:,}")
            
            with col4:
                if state.cycle_results:
                    latest_acc = state.cycle_results[-1].test_accuracy
                    st.metric("Latest Test Acc", f"{latest_acc:.3f}")
                else:
                    st.metric("Latest Test Acc", "N/A")
        
        except Exception as e:
            st.warning(f"Could not load experiment stats: {str(e)}")
    
    # Recent experiments overview
    st.subheader("📋 Recent Experiments")
    
    experiments = st.session_state.experiment_manager.list_experiments()
    
    if experiments:
        # Sort by creation date (most recent first) - handle None values
        experiments.sort(key=lambda x: x.get("created") or "1900-01-01", reverse=True)
        
        for exp in experiments[:5]:  # Show last 5 experiments
            with st.expander(f"🔬 {exp['experiment_id']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Status:** {exp.get('phase', 'Unknown')}")
                    if exp.get('created'):
                        st.write(f"**Created:** {exp['created'][:19].replace('T', ' ')}")
                
                with col2:
                    if exp.get('experiment_name'):
                        st.write(f"**Name:** {exp['experiment_name']}")
                    
                    # Quick action buttons
                    if st.button(f"Select {exp['experiment_id']}", key=f"select_{exp['experiment_id']}"):
                        st.session_state.selected_experiment = exp['experiment_id']
                        exp_dir = Path("experiments") / exp['experiment_id']
                        st.session_state.state_manager = StateManager(exp_dir)
                        st.rerun()
    else:
        st.info("No experiments found. Create your first experiment in the **⚙️ Configuration** page!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        🎯 Active Learning Dashboard | Built with Streamlit | 
        <a href="https://github.com/your-repo" target="_blank">Documentation</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()