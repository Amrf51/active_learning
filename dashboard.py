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

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# NEW: Import MVC controller factory
from controller.controller_factory import (
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

# Initialize controller (FIRST THING - before any other code)
# This creates the controller instance if it doesn't exist
get_controller()

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


def initialize_session_state():
    """Initialize session state variables (MVC version - most state in Controller)."""
    # MVC architecture: Controller manages state, not session_state
    # This function is kept for any view-specific session state only
    pass



def display_experiment_selector():
    """Legacy function - experiment selection now handled by Controller (MVC)."""
    # This function is no longer needed in MVC architecture
    # Experiment selection is handled by the Controller via events
    pass


def display_experiment_status():
    """Display current experiment status in sidebar using Controller."""
    ctrl = get_controller()
    
    # Get current status from Controller (fast, in-memory read)
    status = ctrl.get_status()
    
    if not status.get('experiment_id'):
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
        
        phase = status.get('phase', 'IDLE')
        phase_enum = ExperimentPhase(phase) if phase else ExperimentPhase.IDLE
        phase_icon = phase_colors.get(phase_enum, "⚪")
        st.sidebar.write(f"**Phase:** {phase_icon} {phase}")
        
        # Cycle progress
        total_cycles = status.get('total_cycles', 0)
        current_cycle = status.get('current_cycle', 0)
        if total_cycles > 0:
            progress = current_cycle / total_cycles
            st.sidebar.progress(progress)
            st.sidebar.write(f"**Cycle:** {current_cycle}/{total_cycles}")
        
        # Pool sizes
        labeled_count = status.get('labeled_count', 0)
        unlabeled_count = status.get('unlabeled_count', 0)
        if labeled_count > 0 or unlabeled_count > 0:
            st.sidebar.write(f"**Labeled:** {labeled_count:,}")
            st.sidebar.write(f"**Unlabeled:** {unlabeled_count:,}")
        
        # Service status
        if ctrl.is_service_alive():
            st.sidebar.success("🟢 Service Active")
        else:
            st.sidebar.warning("🟡 Service Inactive")
        
        # Error message
        error_message = status.get('error_message')
        if error_message:
            st.sidebar.error(f"❌ {error_message}")
    
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
    <p><em>💡 Tip: Select an experiment in the sidebar to view its status across all pages.</em></p>
    </div>
    """, unsafe_allow_html=True)


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
    if status.get('experiment_id'):
        experiment_name = status.get('experiment_name', 'Unknown')
        st.success(f"✅ Experiment **{experiment_name}** selected. Use the pages in the sidebar to configure, control, or analyze your experiment.")
        
        # Display quick stats
        st.markdown("---")
        display_quick_stats(status)
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
        Backend Auto-Initialized ✨
    </div>
    """, unsafe_allow_html=True)


def display_quick_stats(status):
    """Display quick statistics for current experiment."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_cycle = status.get('current_cycle', 0)
        total_cycles = status.get('total_cycles', 0)
        st.metric("Current Cycle", f"{current_cycle}/{total_cycles}")
    
    with col2:
        labeled_count = status.get('labeled_count', 0)
        st.metric("Labeled Samples", f"{labeled_count:,}")
    
    with col3:
        unlabeled_count = status.get('unlabeled_count', 0)
        st.metric("Unlabeled Samples", f"{unlabeled_count:,}")
    
    with col4:
        # TODO: Get latest test accuracy from controller
        st.metric("Latest Test Acc", "N/A")


def display_recent_experiments():
    """Display overview of recent experiments."""
    st.subheader("📋 Recent Experiments")
    
    # TODO: Implement recent experiments display using controller
    st.info("Recent experiments display will be implemented with the database integration.")
    
    # Placeholder for now
    st.markdown("""
    **Coming Soon:**
    - View recent experiment history
    - Quick experiment selection
    - Performance comparison
    """)


if __name__ == "__main__":
    main()