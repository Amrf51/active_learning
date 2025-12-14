"""
Configuration Page - Set up new Active Learning experiments

This page allows users to:
- Select model architecture (ResNet-18, ResNet-50, MobileNetV2)
- Choose sampling strategy (Random, Uncertainty, Entropy, Margin)
- Configure training parameters (cycles, pool size, batch size, epochs)
- Set reset mode (pretrained, head_only, none)
- View dataset information
- Initialize new experiments
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
from datetime import datetime
import logging

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from state import ExperimentManager, StateManager, ExperimentConfig, DatasetInfo, ExperimentState, ExperimentPhase
from config import Config

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Configuration - AL Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main content area background */
    .main .block-container {
        background-color: #f8f9fa;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    .config-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .dataset-info {
        background-color: #e8f5e8;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #c3e6cb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    
    /* Form elements styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 6px;
    }
    
    .stSlider > div > div {
        background-color: #ffffff;
        border-radius: 6px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state for configuration."""
    if "experiment_manager" not in st.session_state:
        experiments_dir = Path("experiments")
        st.session_state.experiment_manager = ExperimentManager(experiments_dir)
    
    if "config_form_data" not in st.session_state:
        st.session_state.config_form_data = {
            "experiment_name": "",
            "model_name": "resnet18",
            "sampling_strategy": "uncertainty",
            "num_cycles": 5,
            "initial_pool_size": 50,
            "batch_size_al": 10,
            "epochs_per_cycle": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "reset_mode": "pretrained",
            "seed": 42
        }
    
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None


def load_dataset_info():
    """Load and cache dataset information."""
    if st.session_state.dataset_info is not None:
        return st.session_state.dataset_info
    
    try:
        # Try to load from existing config or create default
        config_path = Path("config/base_config.yaml")
        if config_path.exists():
            config = Config.from_yaml(config_path)
            
            # Mock dataset info for now - in real implementation this would
            # load the actual dataset and compute statistics
            dataset_info = DatasetInfo(
                total_images=1000,  # Placeholder
                num_classes=10,     # Placeholder
                class_names=[f"Class_{i}" for i in range(10)],  # Placeholder
                class_counts={f"Class_{i}": 100 for i in range(10)},  # Placeholder
                train_samples=800,
                val_samples=100,
                test_samples=100
            )
            
            st.session_state.dataset_info = dataset_info
            return dataset_info
        else:
            st.error("Base configuration file not found. Please ensure config/base_config.yaml exists.")
            return None
    
    except Exception as e:
        st.error(f"Error loading dataset info: {str(e)}")
        return None


def display_model_selection():
    """Display model architecture selection."""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.subheader("🏗️ Model Architecture")
    
    model_options = {
        "resnet18": "ResNet-18 (11.7M params) - Fast training, good for small datasets",
        "resnet50": "ResNet-50 (25.6M params) - Better accuracy, slower training", 
        "mobilenetv2": "MobileNetV2 (3.5M params) - Lightweight, efficient"
    }
    
    selected_model = st.selectbox(
        "Select Model Architecture",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.config_form_data["model_name"]),
        help="Choose the neural network architecture for your experiment"
    )
    
    st.session_state.config_form_data["model_name"] = selected_model
    
    # Model details
    if selected_model == "resnet18":
        st.info("✅ **Recommended for beginners** - Fast training with good performance")
    elif selected_model == "resnet50":
        st.info("🎯 **Best accuracy** - Slower training but higher performance potential")
    else:
        st.info("⚡ **Most efficient** - Great for resource-constrained environments")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_sampling_strategy():
    """Display sampling strategy selection."""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.subheader("🎯 Active Learning Strategy")
    
    strategy_options = {
        "uncertainty": "Uncertainty Sampling - Select samples with lowest confidence",
        "entropy": "Entropy Sampling - Select samples with highest prediction entropy",
        "margin": "Margin Sampling - Select samples with smallest margin between top predictions",
        "random": "Random Sampling - Baseline random selection"
    }
    
    selected_strategy = st.selectbox(
        "Select Sampling Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=list(strategy_options.keys()).index(st.session_state.config_form_data["sampling_strategy"]),
        help="Choose how the model selects which samples to label next"
    )
    
    st.session_state.config_form_data["sampling_strategy"] = selected_strategy
    
    # Strategy explanation
    strategy_explanations = {
        "uncertainty": "Selects samples where the model is least confident (lowest max probability)",
        "entropy": "Selects samples with highest uncertainty across all classes",
        "margin": "Selects samples where the top two predictions are closest",
        "random": "Random baseline - useful for comparison with AL strategies"
    }
    
    st.info(f"📝 **How it works:** {strategy_explanations[selected_strategy]}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_training_parameters():
    """Display training parameter configuration."""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.subheader("🔧 Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.config_form_data["num_cycles"] = st.slider(
            "Number of AL Cycles",
            min_value=1,
            max_value=20,
            value=st.session_state.config_form_data["num_cycles"],
            help="How many active learning cycles to run"
        )
        
        st.session_state.config_form_data["initial_pool_size"] = st.number_input(
            "Initial Labeled Pool Size",
            min_value=10,
            max_value=500,
            value=st.session_state.config_form_data["initial_pool_size"],
            help="Number of samples to start with in the labeled pool"
        )
        
        st.session_state.config_form_data["batch_size_al"] = st.number_input(
            "Query Batch Size",
            min_value=1,
            max_value=100,
            value=st.session_state.config_form_data["batch_size_al"],
            help="Number of samples to query in each AL cycle"
        )
    
    with col2:
        st.session_state.config_form_data["epochs_per_cycle"] = st.slider(
            "Epochs per Cycle",
            min_value=1,
            max_value=50,
            value=st.session_state.config_form_data["epochs_per_cycle"],
            help="Number of training epochs in each AL cycle"
        )
        
        st.session_state.config_form_data["batch_size"] = st.selectbox(
            "Training Batch Size",
            options=[16, 32, 64, 128],
            index=[16, 32, 64, 128].index(st.session_state.config_form_data["batch_size"]),
            help="Batch size for training (larger = faster but more memory)"
        )
        
        st.session_state.config_form_data["learning_rate"] = st.selectbox(
            "Learning Rate",
            options=[0.0001, 0.001, 0.01, 0.1],
            index=[0.0001, 0.001, 0.01, 0.1].index(st.session_state.config_form_data["learning_rate"]),
            format_func=lambda x: f"{x:.4f}",
            help="Learning rate for optimizer"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_reset_mode():
    """Display reset mode selection."""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.subheader("🔄 Model Reset Mode")
    
    reset_options = {
        "pretrained": "Pretrained - Reset to ImageNet weights each cycle (recommended)",
        "head_only": "Head Only - Reset only the classifier layer",
        "none": "None - Continue training from previous cycle"
    }
    
    selected_reset = st.selectbox(
        "Select Reset Mode",
        options=list(reset_options.keys()),
        format_func=lambda x: reset_options[x],
        index=list(reset_options.keys()).index(st.session_state.config_form_data["reset_mode"]),
        help="How to reset the model between AL cycles"
    )
    
    st.session_state.config_form_data["reset_mode"] = selected_reset
    
    # Reset mode explanation
    reset_explanations = {
        "pretrained": "Reloads ImageNet weights - ensures fair comparison between cycles",
        "head_only": "Keeps feature extractor, resets classifier - faster convergence",
        "none": "Continues from previous weights - may lead to overfitting"
    }
    
    st.info(f"📝 **Effect:** {reset_explanations[selected_reset]}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_advanced_settings():
    """Display advanced configuration options."""
    with st.expander("🔬 Advanced Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.config_form_data["seed"] = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=9999,
                value=st.session_state.config_form_data["seed"],
                help="Seed for reproducible results"
            )
        
        with col2:
            st.write("**Additional Options:**")
            st.write("• Early stopping: Enabled (patience=5)")
            st.write("• Data augmentation: Enabled")
            st.write("• Optimizer: Adam")
            st.write("• Weight decay: 1e-4")


def display_dataset_information():
    """Display dataset information and statistics."""
    st.subheader("📊 Dataset Information")
    
    dataset_info = load_dataset_info()
    
    if dataset_info is None:
        st.error("Could not load dataset information")
        return
    
    st.markdown('<div class="dataset-info">', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", f"{dataset_info.total_images:,}")
    
    with col2:
        st.metric("Number of Classes", dataset_info.num_classes)
    
    with col3:
        st.metric("Train Samples", f"{dataset_info.train_samples:,}")
    
    with col4:
        st.metric("Test Samples", f"{dataset_info.test_samples:,}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Class distribution chart
    if len(dataset_info.class_names) <= 20:  # Only show chart for reasonable number of classes
        st.subheader("📈 Class Distribution")
        
        import pandas as pd
        import matplotlib.pyplot as plt
        
        # Create class distribution data
        class_data = []
        for class_name in dataset_info.class_names:
            count = dataset_info.class_counts.get(class_name, 0)
            class_data.append({"Class": class_name, "Count": count})
        
        df = pd.DataFrame(class_data)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(df["Class"], df["Count"], color='steelblue', alpha=0.7)
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Class Distribution")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.info(f"Dataset has {len(dataset_info.class_names)} classes. Class distribution chart hidden for readability.")


def validate_configuration():
    """Validate the current configuration."""
    errors = []
    warnings = []
    
    form_data = st.session_state.config_form_data
    
    # Required field validation
    if not form_data.get("experiment_name", "").strip():
        errors.append("Experiment name is required")
    
    # Numeric range validation
    if form_data["num_cycles"] <= 0:
        errors.append("Number of cycles must be greater than 0")
    
    if form_data["initial_pool_size"] <= 0:
        errors.append("Initial pool size must be greater than 0")
    
    if form_data["batch_size_al"] <= 0:
        errors.append("Query batch size must be greater than 0")
    
    if form_data["epochs_per_cycle"] <= 0:
        errors.append("Epochs per cycle must be greater than 0")
    
    # Logical validation
    dataset_info = st.session_state.dataset_info
    if dataset_info:
        if form_data["initial_pool_size"] > dataset_info.train_samples:
            errors.append(f"Initial pool size ({form_data['initial_pool_size']}) cannot exceed training samples ({dataset_info.train_samples})")
        
        total_queries = form_data["num_cycles"] * form_data["batch_size_al"]
        available_samples = dataset_info.train_samples - form_data["initial_pool_size"]
        
        if total_queries > available_samples:
            warnings.append(f"Total queries ({total_queries}) may exceed available unlabeled samples ({available_samples})")
    
    # Performance warnings
    if form_data["batch_size"] > 64 and form_data["model_name"] == "resnet50":
        warnings.append("Large batch size with ResNet-50 may cause memory issues")
    
    if form_data["learning_rate"] > 0.01:
        warnings.append("High learning rate may cause training instability")
    
    return errors, warnings


def create_experiment_config():
    """Create ExperimentConfig from form data."""
    form_data = st.session_state.config_form_data
    dataset_info = st.session_state.dataset_info
    
    if not dataset_info:
        raise ValueError("Dataset information not available")
    
    return ExperimentConfig(
        model_name=form_data["model_name"],
        pretrained=True,
        num_classes=dataset_info.num_classes,
        class_names=dataset_info.class_names,
        epochs_per_cycle=form_data["epochs_per_cycle"],
        batch_size=form_data["batch_size"],
        learning_rate=form_data["learning_rate"],
        optimizer="adam",
        weight_decay=1e-4,
        early_stopping_patience=5,
        num_cycles=form_data["num_cycles"],
        sampling_strategy=form_data["sampling_strategy"],
        uncertainty_method="entropy",
        initial_pool_size=form_data["initial_pool_size"],
        batch_size_al=form_data["batch_size_al"],
        reset_mode=form_data["reset_mode"],
        seed=form_data["seed"],
        data_dir="data",  # Default data directory
        val_split=0.1,
        test_split=0.1,
        augmentation=True
    )


def initialize_experiment():
    """Initialize a new experiment with the current configuration."""
    try:
        # Validate configuration
        errors, warnings = validate_configuration()
        
        if errors:
            st.error("❌ Configuration errors found:")
            for error in errors:
                st.error(f"• {error}")
            return False
        
        if warnings:
            st.warning("⚠️ Configuration warnings:")
            for warning in warnings:
                st.warning(f"• {warning}")
        
        # Create experiment
        experiment_name = st.session_state.config_form_data["experiment_name"].strip()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        # Create experiment directory
        exp_dir = st.session_state.experiment_manager.create_experiment(
            experiment_name, experiment_id
        )
        
        # Create state manager and initialize state
        state_manager = StateManager(exp_dir)
        state = state_manager.initialize_state(experiment_id, experiment_name)
        
        # Create and save configuration
        config = create_experiment_config()
        state.config = config
        state.total_cycles = config.num_cycles
        state.dataset_info = st.session_state.dataset_info
        
        # Save updated state
        state_manager.write_state(state)
        
        # Set as active experiment
        st.session_state.experiment_manager.set_active(exp_dir)
        
        # Update session state
        st.session_state.selected_experiment = experiment_id
        st.session_state.state_manager = state_manager
        
        st.success(f"✅ Experiment **{experiment_id}** created successfully!")
        st.info("🚀 You can now start the worker process and begin training in the **Active Learning** page.")
        
        # Display worker command
        st.code(f"python run_worker.py --experiment-id {experiment_id}", language="bash")
        
        return True
    
    except Exception as e:
        st.error(f"❌ Failed to create experiment: {str(e)}")
        logger.error(f"Experiment creation failed: {e}")
        return False


def main():
    """Main configuration page."""
    initialize_session_state()
    
    st.title("⚙️ Experiment Configuration")
    st.markdown("Set up a new Active Learning experiment with your preferred settings.")
    
    # Experiment name input
    st.subheader("📝 Experiment Details")
    st.session_state.config_form_data["experiment_name"] = st.text_input(
        "Experiment Name",
        value=st.session_state.config_form_data["experiment_name"],
        placeholder="e.g., uncertainty_resnet18_baseline",
        help="A descriptive name for your experiment"
    )
    
    # Configuration sections
    display_model_selection()
    display_sampling_strategy()
    display_training_parameters()
    display_reset_mode()
    display_advanced_settings()
    
    # Dataset information
    display_dataset_information()
    
    # Configuration summary and validation
    st.subheader("📋 Configuration Summary")
    
    # Validate configuration
    errors, warnings = validate_configuration()
    
    if errors:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.error("❌ Please fix the following errors before proceeding:")
        for error in errors:
            st.write(f"• {error}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if warnings:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.warning("⚠️ Configuration warnings:")
        for warning in warnings:
            st.write(f"• {warning}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display configuration summary
    form_data = st.session_state.config_form_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model & Strategy:**")
        st.write(f"• Model: {form_data['model_name']}")
        st.write(f"• Strategy: {form_data['sampling_strategy']}")
        st.write(f"• Reset mode: {form_data['reset_mode']}")
        
        st.write("**Training:**")
        st.write(f"• Cycles: {form_data['num_cycles']}")
        st.write(f"• Epochs per cycle: {form_data['epochs_per_cycle']}")
        st.write(f"• Batch size: {form_data['batch_size']}")
    
    with col2:
        st.write("**Active Learning:**")
        st.write(f"• Initial pool: {form_data['initial_pool_size']}")
        st.write(f"• Query batch: {form_data['batch_size_al']}")
        
        st.write("**Other:**")
        st.write(f"• Learning rate: {form_data['learning_rate']}")
        st.write(f"• Seed: {form_data['seed']}")
    
    # Initialize experiment button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button(
            "🚀 Initialize Experiment",
            type="primary",
            disabled=bool(errors),
            use_container_width=True
        ):
            if initialize_experiment():
                st.balloons()


if __name__ == "__main__":
    main()