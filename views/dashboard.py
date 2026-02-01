"""
Interactive Active Learning Dashboard - Main Entry Point (MVC Architecture)

This is the main Streamlit application that provides an interactive interface
for controlling and monitoring Active Learning experiments using the new
MVC (Model-View-Controller) architecture.

Key Changes from Old Version:
- Uses EventDispatcher (Controller) instead of StateManager
- Auto-initializes controller session and multi-tab detection
- Reads from in-memory WorldState instead of JSON files
- Heartbeat mechanism to keep session alive

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

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# NEW: Import MVC controller factory
from views.controller_factory import (
    initialize_controller_session,
    get_controller,
    update_session_heartbeat
)
from model.schemas import ExperimentPhase

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

# Initialize controller session (FIRST THING - before any other code)
# This ensures only one browser tab can run the dashboard at a time
if not initialize_controller_session():
    st.stop()

# Update session heartbeat to prevent timeout
update_session_heartbeat()

# Custom CSS for better styling
st.markdown("""
    <style>
    /* 1. MAIN HEADER */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e6f1ff; /* Light Text (was blue) */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 2. METRIC CARD (Matches .config-section) */
    .metric-card {
        background-color: #112240; /* Dark Navy */
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4; /* Blue Accent */
        color: #e6f1ff; /* Light Text */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* 3. STATUS COLORS (Brightened for Dark Mode) */
    .status-running {
        color: #4ade80; /* Bright Green */
        font-weight: bold;
    }

    .status-idle {
        color: #8892b0; /* Light Slate/Gray */
        font-weight: bold;
    }

    .status-error {
        color: #ff6b6b; /* Soft Red */
        font-weight: bold;
    }

    /* 4. EXPERIMENT CARD (Matches .config-section but simpler) */
    .experiment-card {
        background-color: #112240; /* Dark Navy */
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #233554; /* Dark Blue Border */
        margin: 0.5rem 0;
        color: #e6f1ff; /* Light Text */
    }

    /* 5. NAV INFO (Matches .dataset-info) */
    .nav-info {
        background-color: #0f3d3e; /* Dark Teal */
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #20c997; /* Bright Teal Border */
        color: #e6f1ff; /* Light Text */
    }
    </style>
""", unsafe_allow_html=True)


def display_experiment_status():
    """Display current experiment status in sidebar using Controller."""
    ctrl = get_controller()
    
    # Get current status from Controller (fast, in-memory read)
    status = ctrl.get_status()
    
    if not status.experiment_id:
        st.sidebar.info("👈 No experiment selected. Create one in Configuration page.")
        return
    
    try:
        st.sidebar.subheader("📊 Experiment Status")
        
        # Phase status with color coding
        phase_colors = {
            ExperimentPhase.IDLE: "🔵",
            ExperimentPhase.TRAINING: "🟢", 
            ExperimentPhase.EVALUATING: "🟡",
            ExperimentPhase.QUERYING: "🟠",
            ExperimentPhase.AWAITING_ANNOTATION: "🟣",
            ExperimentPhase.COMPLETED: "✅",
            ExperimentPhase.ERROR: "🔴"
        }
        
        phase_icon = phase_colors.get(status.phase, "⚪")
        st.sidebar.write(f"**Phase:** {phase_icon} {status.phase.value}")
        
        # Cycle progress
        if status.total_cycles > 0:
            progress = status.current_cycle / status.total_cycles
            st.sidebar.progress(progress)
            st.sidebar.write(f"**Cycle:** {status.current_cycle}/{status.total_cycles}")
        
        # Pool sizes
        if status.labeled_count > 0 or status.unlabeled_count > 0:
            st.sidebar.write(f"**Labeled:** {status.labeled_count:,}")
            st.sidebar.write(f"**Unlabeled:** {status.unlabeled_count:,}")
        
        # Service status
        if ctrl._service_manager and ctrl._service_manager.is_alive():
            st.sidebar.success("🟢 Service Active")
        else:
            st.sidebar.warning("🟡 Service Inactive")
        
        # Error message
        if status.error_message:
            st.sidebar.error(f"❌ {status.error_message}")
    
    except Exception as e:
        st.sidebar.error(f"❌ Error reading status: {str(e)}")
        logger.error(f"Error in display_experiment_status: {e}")


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
    <p><em>💡 Tip: The dashboard auto-manages the training service - no manual worker commands needed!</em></p>
    </div>
    """, unsafe_allow_html=True)


def display_quick_stats():
    """Display quick statistics for current experiment."""
    ctrl = get_controller()
    status = ctrl.get_status()
    
    if not status.experiment_id:
        return
    
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Cycle", f"{status.current_cycle}/{status.total_cycles}")
        
        with col2:
            st.metric("Labeled Samples", f"{status.labeled_count:,}")
        
        with col3:
            st.metric("Unlabeled Samples", f"{status.unlabeled_count:,}")
        
        with col4:
            # Get latest test accuracy from database
            if status.current_cycle > 0:
                cycles = ctrl._model_handler.database.get_cycle_summaries(
                    experiment_id=status.experiment_id,
                    limit=1
                )
                if cycles:
                    st.metric("Latest Test Acc", f"{cycles[0].test_accuracy:.3f}")
                else:
                    st.metric("Latest Test Acc", "N/A")
            else:
                st.metric("Latest Test Acc", "N/A")
    
    except Exception as e:
        st.warning(f"Could not load experiment stats: {str(e)}")
        logger.error(f"Error in display_quick_stats: {e}")


def display_recent_experiments():
    """Display overview of recent experiments."""
    st.subheader("📋 Recent Experiments")
    
    ctrl = get_controller()
    
    try:
        # Get recent experiments from database
        experiments = ctrl._model_handler.database.get_all_experiments(limit=5)
        
        if experiments:
            for exp in experiments:
                with st.expander(f"🔬 {exp['name']}", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**ID:** {exp['id']}")
                        st.write(f"**Phase:** {exp.get('phase', 'Unknown')}")
                        if exp.get('created_at'):
                            st.write(f"**Created:** {exp['created_at'][:19].replace('T', ' ')}")
                    
                    with col2:
                        # Show some config info
                        if exp.get('config'):
                            config = exp['config']
                            if isinstance(config, dict):
                                st.write(f"**Model:** {config.get('model', {}).get('name', 'Unknown')}")
                                st.write(f"**Strategy:** {config.get('active_learning', {}).get('sampling_strategy', 'Unknown')}")
        else:
            st.info("No experiments found. Create your first experiment in the **⚙️ Configuration** page!")
    
    except Exception as e:
        st.warning(f"Could not load recent experiments: {str(e)}")
        logger.error(f"Error in display_recent_experiments: {e}")


def main():
    """Main dashboard application - MVC architecture."""
    
    # Main header
    st.markdown('<h1 class="main-header">🎯 Active Learning Dashboard</h1>', unsafe_allow_html=True)
    
    # Display navigation info
    display_navigation_info()
    
    # Sidebar experiment status
    display_experiment_status()
    
    # Get controller and status
    ctrl = get_controller()
    status = ctrl.get_status()
    
    # Main content area
    if status.experiment_id:
        st.success(f"✅ Experiment **{status.experiment_name}** selected. Use the pages in the sidebar to configure, control, or analyze your experiment.")
        
        # Display quick stats
        st.markdown("---")
        display_quick_stats()
    else:
        st.info("👈 No experiment active. Create a new experiment in the **Configuration** page.")
    
    # Recent experiments overview
    st.markdown("---")
    display_recent_experiments()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
        🎯 Active Learning Dashboard (MVC Architecture) | Built with Streamlit | 
        Service Auto-Managed ✨
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()