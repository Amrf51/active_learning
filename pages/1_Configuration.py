"""
Configuration Page - Set up new Active Learning experiments

This page allows users to:
- Select dataset from available datasets in ./data/raw/
- View real dataset statistics (classes, images, distribution)
- Configure train/val/test splits with live preview
- Select model architecture (ResNet-18, ResNet-50, MobileNetV2)
- Choose sampling strategy (Random, Uncertainty, Entropy, Margin)
- Configure training parameters (cycles, pool size, batch size, epochs)
- Set reset mode (pretrained, head_only, none)
- Initialize new experiments
"""

import streamlit as st
from pathlib import Path
import sys
import yaml
from datetime import datetime
import logging
import pandas as pd

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from state import ExperimentManager, StateManager, ExperimentConfig, DatasetInfo, ExperimentState, ExperimentPhase
from config import ConfigManager, Config

logger = logging.getLogger(__name__)

# Constants
DEFAULT_DATA_BASE_PATH = "./data/raw"
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# Page configuration
st.set_page_config(
    page_title="Configuration - AL Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* 1. CONFIGURATION CARD (The main container) */
    .config-section {
        background-color: #112240;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* 2. DATASET INFO BOX (Informational) */
    .dataset-info {
        background-color: #0f3d3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #20c997;
        color: #e6f1ff;
    }

    /* 3. WARNING BOX (Yellow/Orange) */
    .warning-box {
        background-color: #332b00;
        border: 1px solid #ffca28;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ffe082;
    }

    /* 4. SUCCESS BOX (Green) */
    .success-box {
        background-color: #052c16;
        border: 1px solid #4ade80;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #bbf7d0;
    }
    
    /* 5. TEXT INPUTS FIX (Make typing visible) */
    .stTextInput > div > div > input {
        color: white;
        background-color: #112240;
    }
    
    /* 6. AL PREVIEW BOX */
    .al-preview {
        background-color: #1a1a2e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #4a4a6a;
    }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# Dataset Scanning Functions
# =============================================================================

def discover_available_datasets(base_path: str = DEFAULT_DATA_BASE_PATH) -> list:
    """
    Discover available datasets in the base data directory.
    
    Args:
        base_path: Base directory containing dataset folders
        
    Returns:
        List of dataset folder names
    """
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        return []
    
    # Find all subdirectories that look like datasets (contain class folders)
    datasets = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and not item.name.startswith('.'):
            # Check if it contains subdirectories (class folders)
            subdirs = [d for d in item.iterdir() if d.is_dir() and not d.name.startswith('.')]
            if subdirs:
                datasets.append(item.name)
    
    return datasets


def scan_dataset(data_dir: str) -> DatasetInfo:
    """
    Scan an ImageFolder directory and return real statistics.
    
    Args:
        data_dir: Path to dataset directory (contains class folders)
        
    Returns:
        DatasetInfo with actual dataset statistics
        
    Raises:
        ValueError: If path doesn't exist or has no valid structure
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Path does not exist: {data_dir}")
    
    # Discover classes (subdirectories)
    class_names = sorted([
        d.name for d in data_path.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    if not class_names:
        raise ValueError(f"No class folders found in: {data_dir}")
    
    # Count images per class
    class_counts = {}
    total_images = 0
    
    for class_name in class_names:
        class_dir = data_path / class_name
        # Count image files
        count = sum(
            1 for f in class_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        class_counts[class_name] = count
        total_images += count
    
    if total_images == 0:
        raise ValueError(f"No images found in dataset: {data_dir}")
    
    return DatasetInfo(
        total_images=total_images,
        num_classes=len(class_names),
        class_names=class_names,
        class_counts=class_counts,
        train_samples=0,  # Will be computed after split config
        val_samples=0,
        test_samples=0
    )


def compute_split_sizes(total_images: int, val_split: float, test_split: float) -> dict:
    """
    Compute the actual number of samples in each split.
    
    Args:
        total_images: Total number of images in dataset
        val_split: Fraction for validation (0-1)
        test_split: Fraction for test (0-1)
        
    Returns:
        Dict with train_samples, val_samples, test_samples
    """
    val_samples = int(total_images * val_split)
    test_samples = int(total_images * test_split)
    train_samples = total_images - val_samples - test_samples
    
    return {
        "train_samples": train_samples,
        "val_samples": val_samples,
        "test_samples": test_samples
    }


# =============================================================================
# Session State Initialization
# =============================================================================

def initialize_session_state():
    """Initialize session state for configuration."""
    if "experiment_manager" not in st.session_state:
        experiments_dir = Path("experiments")
        st.session_state.experiment_manager = ExperimentManager(experiments_dir)
    
    if "config_manager" not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
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
            "seed": 42,
            # New dataset-related fields
            "data_base_path": DEFAULT_DATA_BASE_PATH,
            "selected_dataset": None,
            "val_split": 0.15,
            "test_split": 0.15,
        }
    
    if "dataset_info" not in st.session_state:
        st.session_state.dataset_info = None
    
    if "dataset_scanned" not in st.session_state:
        st.session_state.dataset_scanned = False
    
    if "available_datasets" not in st.session_state:
        st.session_state.available_datasets = []


# =============================================================================
# Dataset Selection UI
# =============================================================================

def display_dataset_selection():
    """Display dataset selection and scanning interface."""
    st.subheader("📁 Dataset Configuration")
    
    form_data = st.session_state.config_form_data
    
    # Base path display (fixed for simplicity)
    st.markdown(f"**Base Path:** `{form_data['data_base_path']}`")
    
    # Discover available datasets
    available_datasets = discover_available_datasets(form_data['data_base_path'])
    st.session_state.available_datasets = available_datasets
    
    if not available_datasets:
        st.error(f"❌ No datasets found in `{form_data['data_base_path']}`")
        st.info("Please ensure your datasets are organized in ImageFolder structure:")
        st.code("""
./data/raw/
├── kaggle-vehicle/
│   ├── bus/
│   ├── car/
│   └── ...
├── stanford/
│   ├── class1/
│   └── ...
        """)
        return
    
    # Dataset dropdown
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Determine current index
        current_selection = form_data.get("selected_dataset")
        if current_selection and current_selection in available_datasets:
            default_index = available_datasets.index(current_selection)
        else:
            default_index = 0
        
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=available_datasets,
            index=default_index,
            help="Choose a dataset from the available options"
        )
        
        # Update form data
        form_data["selected_dataset"] = selected_dataset
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_clicked = st.button("🔍 Scan Dataset", type="primary", use_container_width=True)
    
    # Handle scan button
    if scan_clicked:
        scan_selected_dataset()
    
    # Show dataset info if scanned
    if st.session_state.dataset_scanned and st.session_state.dataset_info is not None:
        display_dataset_overview()
        display_split_configuration()
        display_al_pool_preview()


def scan_selected_dataset():
    """Scan the currently selected dataset and update session state."""
    form_data = st.session_state.config_form_data
    
    if not form_data.get("selected_dataset"):
        st.error("Please select a dataset first")
        return
    
    dataset_path = Path(form_data["data_base_path"]) / form_data["selected_dataset"]
    
    try:
        with st.spinner(f"Scanning {form_data['selected_dataset']}..."):
            dataset_info = scan_dataset(str(dataset_path))
            
            # Compute initial splits
            splits = compute_split_sizes(
                dataset_info.total_images,
                form_data["val_split"],
                form_data["test_split"]
            )
            
            # Update dataset info with split sizes
            dataset_info.train_samples = splits["train_samples"]
            dataset_info.val_samples = splits["val_samples"]
            dataset_info.test_samples = splits["test_samples"]
            
            # Store in session state
            st.session_state.dataset_info = dataset_info
            st.session_state.dataset_scanned = True
            
            st.success(f"✅ Dataset scanned successfully!")
            st.rerun()
            
    except ValueError as e:
        st.error(f"❌ {str(e)}")
        st.session_state.dataset_scanned = False
    except Exception as e:
        st.error(f"❌ Error scanning dataset: {str(e)}")
        st.session_state.dataset_scanned = False


def display_dataset_overview():
    """Display overview of the scanned dataset."""
    dataset_info = st.session_state.dataset_info
    
    if dataset_info is None:
        return
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", f"{dataset_info.total_images:,}")
    
    with col2:
        st.metric("Classes", dataset_info.num_classes)
    
    with col3:
        # Check for class imbalance
        counts = list(dataset_info.class_counts.values())
        if counts:
            imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if imbalance_ratio > 2:
                st.metric("Balance", f"⚠️ {imbalance_ratio:.1f}x")
            else:
                st.metric("Balance", f"✅ {imbalance_ratio:.1f}x")
    
    with col4:
        avg_per_class = dataset_info.total_images / dataset_info.num_classes if dataset_info.num_classes > 0 else 0
        st.metric("Avg/Class", f"{avg_per_class:.0f}")
    
    # Class list
    with st.expander("📋 Classes Found", expanded=False):
        class_list = ", ".join(dataset_info.class_names)
        st.write(class_list)
    
    # Class distribution chart
    if dataset_info.num_classes <= 30:
        st.markdown("#### Class Distribution")
        
        # Create DataFrame for chart
        chart_data = pd.DataFrame({
            "Class": list(dataset_info.class_counts.keys()),
            "Images": list(dataset_info.class_counts.values())
        })
        chart_data = chart_data.set_index("Class")
        
        st.bar_chart(chart_data)
    else:
        st.info(f"Dataset has {dataset_info.num_classes} classes. Distribution chart hidden for readability.")
        
        # Show top/bottom classes instead
        with st.expander("📊 Top/Bottom Classes by Count"):
            sorted_classes = sorted(dataset_info.class_counts.items(), key=lambda x: x[1], reverse=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Top 5 Classes:**")
                for name, count in sorted_classes[:5]:
                    st.write(f"• {name}: {count}")
            
            with col2:
                st.write("**Bottom 5 Classes:**")
                for name, count in sorted_classes[-5:]:
                    st.write(f"• {name}: {count}")


def display_split_configuration():
    """Display and configure train/val/test splits."""
    dataset_info = st.session_state.dataset_info
    form_data = st.session_state.config_form_data
    
    if dataset_info is None:
        return
    
    st.markdown("---")
    st.markdown("### ⚙️ Data Splits")
    st.markdown("Configure how the dataset is split into training, validation, and test sets.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        val_split = st.slider(
            "Validation Split",
            min_value=0.05,
            max_value=0.30,
            value=form_data["val_split"],
            step=0.05,
            format="%.0f%%",
            help="Percentage of data for validation"
        )
        form_data["val_split"] = val_split
    
    with col2:
        test_split = st.slider(
            "Test Split",
            min_value=0.05,
            max_value=0.30,
            value=form_data["test_split"],
            step=0.05,
            format="%.0f%%",
            help="Percentage of data for testing"
        )
        form_data["test_split"] = test_split
    
    with col3:
        train_split = 1.0 - val_split - test_split
        st.metric("Training Split", f"{train_split:.0%}")
    
    # Compute actual numbers
    splits = compute_split_sizes(dataset_info.total_images, val_split, test_split)
    
    # Update dataset_info
    dataset_info.train_samples = splits["train_samples"]
    dataset_info.val_samples = splits["val_samples"]
    dataset_info.test_samples = splits["test_samples"]
    
    # Display split preview
    st.markdown("#### Split Preview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"🎯 **Training Pool**\n\n{splits['train_samples']:,} images\n\n*Used for Active Learning*")
    
    with col2:
        st.info(f"📊 **Validation**\n\n{splits['val_samples']:,} images\n\n*Fixed throughout AL*")
    
    with col3:
        st.info(f"🧪 **Test**\n\n{splits['test_samples']:,} images\n\n*Final evaluation*")
    
    # Warning if splits are too small
    if splits["val_samples"] < 20:
        st.warning("⚠️ Validation set is very small. Consider reducing the validation split.")
    
    if splits["test_samples"] < 20:
        st.warning("⚠️ Test set is very small. Consider reducing the test split.")


def display_al_pool_preview():
    """Display Active Learning pool configuration preview."""
    dataset_info = st.session_state.dataset_info
    form_data = st.session_state.config_form_data
    
    if dataset_info is None:
        return
    
    st.markdown("---")
    st.markdown("### 🎯 Active Learning Pool Preview")
    
    train_samples = dataset_info.train_samples
    initial_pool = form_data["initial_pool_size"]
    query_batch = form_data["batch_size_al"]
    num_cycles = form_data["num_cycles"]
    
    # Calculate projections
    unlabeled_pool = train_samples - initial_pool
    total_queries = num_cycles * query_batch
    final_labeled = initial_pool + total_queries
    final_unlabeled = train_samples - final_labeled
    
    # Display current AL settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Starting State:**")
        st.write(f"• Training pool: {train_samples:,} images")
        st.write(f"• Initial labeled: {initial_pool:,} images")
        st.write(f"• Unlabeled pool: {unlabeled_pool:,} images")
    
    with col2:
        st.markdown("**After {0} Cycles:**".format(num_cycles))
        st.write(f"• Total labeled: {final_labeled:,} images ({100*final_labeled/train_samples:.1f}%)")
        st.write(f"• Remaining unlabeled: {max(0, final_unlabeled):,} images")
        st.write(f"• Total annotations: {total_queries:,} images")
    
    # Projection visualization
    st.markdown("#### 📈 Labeling Progression")
    
    progression_data = []
    current_labeled = initial_pool
    for cycle in range(num_cycles + 1):
        progression_data.append({
            "Cycle": cycle,
            "Labeled": current_labeled,
            "Unlabeled": train_samples - current_labeled
        })
        if cycle < num_cycles:
            current_labeled = min(current_labeled + query_batch, train_samples)
    
    prog_df = pd.DataFrame(progression_data)
    prog_df = prog_df.set_index("Cycle")
    
    st.line_chart(prog_df)
    
    # Warnings
    if initial_pool > train_samples:
        st.error(f"❌ Initial pool size ({initial_pool}) exceeds training samples ({train_samples})")
    elif initial_pool > train_samples * 0.5:
        st.warning(f"⚠️ Initial pool is {100*initial_pool/train_samples:.0f}% of training data. Consider a smaller initial pool for AL to be effective.")
    
    if total_queries > unlabeled_pool:
        st.warning(f"⚠️ Total queries ({total_queries}) exceed available unlabeled samples ({unlabeled_pool}). Some cycles may query fewer samples.")
    
    if final_labeled >= train_samples:
        st.info(f"ℹ️ All training samples will be labeled by cycle {num_cycles}.")


# =============================================================================
# Template Management
# =============================================================================

def display_template_management():
    """Display template management section."""
    with st.expander("📋 Configuration Templates", expanded=False):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("Load from existing templates or save current configuration as a template.")
        
        with col2:
            if st.button("🔄 Refresh Templates"):
                st.rerun()
        
        # Get available templates
        templates = st.session_state.config_manager.list_templates()
        
        if templates:
            template_options = {}
            for template in templates:
                status = "🔄 AL" if template["has_active_learning"] else "📊 Standard"
                legacy = " (Legacy)" if template["is_legacy"] else ""
                label = f"{status} {template['name']}{legacy}"
                template_options[label] = template
            
            selected_template_label = st.selectbox(
                "Load Configuration Template",
                options=["-- Select Template --"] + list(template_options.keys()),
                help="Choose a template to load its configuration"
            )
            
            if selected_template_label != "-- Select Template --":
                template = template_options[selected_template_label]
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**Description:** {template['description']}")
                
                with col2:
                    if st.button("📥 Load Template", key="load_template"):
                        load_template_config(template['name'])
                
                with col3:
                    if st.button("👁️ Preview", key="preview_template"):
                        preview_template_config(template['name'])
        else:
            st.info("No templates found. Create your first template by configuring an experiment and saving it.")
        
        # Save current config as template
        st.markdown("---")
        st.markdown("**💾 Save Current Configuration**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            template_name = st.text_input(
                "Template Name",
                placeholder="e.g., resnet50_uncertainty_baseline",
                help="Name for the new template"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 Save Template", disabled=not template_name.strip()):
                save_current_config_as_template(template_name.strip(), "")


def load_template_config(template_name: str):
    """Load configuration from a template."""
    try:
        config = Config.from_template(template_name)
        
        st.session_state.config_form_data.update({
            "model_name": config.model.name,
            "num_cycles": config.active_learning.num_cycles,
            "sampling_strategy": config.active_learning.sampling_strategy,
            "initial_pool_size": config.active_learning.initial_pool_size,
            "batch_size_al": config.active_learning.batch_size_al,
            "epochs_per_cycle": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "reset_mode": config.active_learning.reset_mode,
            "seed": config.training.seed
        })
        
        st.success(f"✅ Loaded template: {template_name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to load template: {e}")


def preview_template_config(template_name: str):
    """Preview a template configuration."""
    try:
        config = Config.from_template(template_name)
        config_dict = config.to_dict()
        
        st.markdown(f"### 👁️ Template Preview: {template_name}")
        st.json(config_dict)
        
    except Exception as e:
        st.error(f"❌ Failed to preview template: {e}")


def save_current_config_as_template(template_name: str, description: str):
    """Save current form configuration as a template."""
    try:
        config = create_config_from_form_data()
        template_path = config.save_as_template(template_name, description)
        
        st.success(f"✅ Saved template: {template_name}")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to save template: {e}")


# =============================================================================
# Model & Training Configuration
# =============================================================================

def display_model_selection():
    """Display model architecture selection."""
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.subheader("🏗️ Model Architecture")
    
    model_options = {
        "resnet18": "ResNet-18 (11.7M params) - Fast training, good for small datasets",
        "resnet50": "ResNet-50 (25.6M params) - Better accuracy, slower training", 
        "mobilenetv2_100": "MobileNetV2 (3.5M params) - Lightweight, efficient"
    }
    
    selected_model = st.selectbox(
        "Select Model Architecture",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.config_form_data["model_name"]) 
            if st.session_state.config_form_data["model_name"] in model_options else 0,
        help="Choose the neural network architecture for your experiment"
    )
    
    st.session_state.config_form_data["model_name"] = selected_model
    
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


# =============================================================================
# Configuration Validation & Experiment Creation
# =============================================================================

def validate_configuration():
    """Validate the current configuration."""
    errors = []
    warnings = []
    
    form_data = st.session_state.config_form_data
    dataset_info = st.session_state.dataset_info
    
    # Required field validation
    if not form_data.get("experiment_name", "").strip():
        errors.append("Experiment name is required")
    
    # Dataset validation
    if not st.session_state.dataset_scanned or dataset_info is None:
        errors.append("Please scan a dataset first")
    
    # Numeric range validation
    if form_data["num_cycles"] <= 0:
        errors.append("Number of cycles must be greater than 0")
    
    if form_data["initial_pool_size"] <= 0:
        errors.append("Initial pool size must be greater than 0")
    
    if form_data["batch_size_al"] <= 0:
        errors.append("Query batch size must be greater than 0")
    
    if form_data["epochs_per_cycle"] <= 0:
        errors.append("Epochs per cycle must be greater than 0")
    
    # Dataset-specific validation
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


def create_config_from_form_data() -> Config:
    """Create a Config object from current form data."""
    form_data = st.session_state.config_form_data
    dataset_info = st.session_state.dataset_info
    
    config = Config()
    
    # Model config
    config.model.name = form_data["model_name"]
    config.model.pretrained = True
    config.model.num_classes = dataset_info.num_classes if dataset_info else 4
    
    # Training config
    config.training.epochs = form_data["epochs_per_cycle"]
    config.training.batch_size = form_data["batch_size"]
    config.training.learning_rate = form_data["learning_rate"]
    config.training.seed = form_data["seed"]
    
    # Data config
    if form_data.get("selected_dataset"):
        config.data.data_dir = str(Path(form_data["data_base_path"]) / form_data["selected_dataset"])
    config.data.val_split = form_data["val_split"]
    config.data.test_split = form_data["test_split"]
    
    # Active Learning config
    config.active_learning.enabled = True
    config.active_learning.num_cycles = form_data["num_cycles"]
    config.active_learning.sampling_strategy = form_data["sampling_strategy"]
    config.active_learning.initial_pool_size = form_data["initial_pool_size"]
    config.active_learning.batch_size_al = form_data["batch_size_al"]
    config.active_learning.reset_mode = form_data["reset_mode"]
    
    return config


def create_experiment_config() -> ExperimentConfig:
    """Create ExperimentConfig from form data."""
    form_data = st.session_state.config_form_data
    dataset_info = st.session_state.dataset_info
    
    if not dataset_info:
        raise ValueError("Dataset information not available")
    
    data_dir = str(Path(form_data["data_base_path"]) / form_data["selected_dataset"]) if form_data.get("selected_dataset") else "data"
    
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
        data_dir=data_dir,
        val_split=form_data["val_split"],
        test_split=form_data["test_split"],
        augmentation=True
    )


def initialize_experiment():
    """Initialize a new experiment with the current configuration."""
    try:
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
        
        # Also save the full Config object to the experiment directory
        full_config = create_config_from_form_data()
        full_config.save_to(str(exp_dir / "config.yaml"), include_metadata=True)
        
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


# =============================================================================
# Main Page
# =============================================================================

def main():
    """Main configuration page."""
    initialize_session_state()
    
    st.title("⚙️ Experiment Configuration")
    st.markdown("Set up a new Active Learning experiment with your preferred settings.")
    
    # Step 1: Dataset Selection (FIRST - this is critical)
    display_dataset_selection()
    
    # Only show remaining configuration if dataset is scanned
    if st.session_state.dataset_scanned and st.session_state.dataset_info is not None:
        st.markdown("---")
        
        # Template management (collapsed by default)
        display_template_management()
        
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
        
        # Configuration summary and validation
        st.markdown("---")
        st.subheader("📋 Configuration Summary")
        
        errors, warnings = validate_configuration()
        
        if errors:
            for error in errors:
                st.error(f"❌ {error}")
        
        if warnings:
            for warning in warnings:
                st.warning(f"⚠️ {warning}")
        
        # Display configuration summary
        form_data = st.session_state.config_form_data
        dataset_info = st.session_state.dataset_info
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Dataset & Splits:**")
            st.write(f"• Dataset: {form_data['selected_dataset']}")
            st.write(f"• Classes: {dataset_info.num_classes}")
            st.write(f"• Training pool: {dataset_info.train_samples}")
        
        with col2:
            st.write("**Model & Strategy:**")
            st.write(f"• Model: {form_data['model_name']}")
            st.write(f"• Strategy: {form_data['sampling_strategy']}")
            st.write(f"• Reset mode: {form_data['reset_mode']}")
        
        with col3:
            st.write("**Active Learning:**")
            st.write(f"• Cycles: {form_data['num_cycles']}")
            st.write(f"• Initial pool: {form_data['initial_pool_size']}")
            st.write(f"• Query batch: {form_data['batch_size_al']}")
        
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
    
    else:
        # Prompt user to scan dataset
        st.info("👆 Please select and scan a dataset to continue with experiment configuration.")


if __name__ == "__main__":
    main()