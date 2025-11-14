"""Streamlit dashboard for experiment results visualization and comparison."""

import streamlit as st
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Active Learning Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_experiments():
    """Load all completed experiments."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return {}
    
    experiments = {}
    for exp_dir in sorted(experiments_dir.glob("*/")):
        metrics_file = exp_dir / "metrics.json"
        config_file = exp_dir / "config.yaml"
        
        if metrics_file.exists() and config_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            
            experiments[exp_dir.name] = {
                "path": exp_dir,
                "metrics": metrics,
                "config_file": config_file,
            }
    
    return experiments


def display_experiment_summary(exp_name, exp_data):
    """Display summary for a single experiment."""
    metrics = exp_data["metrics"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    with col2:
        st.metric("Test F1 Score", f"{metrics.get('test_f1', 0):.4f}")
    with col3:
        st.metric("Best Epoch", f"{metrics.get('best_epoch', 0)}")
    with col4:
        st.metric("Best Val Acc", f"{metrics.get('best_val_accuracy', 0):.4f}")
    
    # Per-class metrics if available
    if "per_class" in metrics:
        st.subheader("Per-Class Metrics")
        per_class_data = []
        for class_name, class_metrics in metrics["per_class"].items():
            per_class_data.append({
                "Class": class_name,
                "Precision": class_metrics["precision"],
                "Recall": class_metrics["recall"],
                "F1": class_metrics["f1"],
            })
        
        df_per_class = pd.DataFrame(per_class_data)
        st.dataframe(df_per_class, use_container_width=True)


def load_training_history(exp_dir):
    """Load training history for an experiment."""
    history_file = exp_dir / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)
    return None


def plot_training_curves(exp_dir):
    """Plot training and validation curves."""
    history = load_training_history(exp_dir)
    if history is None:
        st.warning("No training history found")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train Acc", marker="o")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Val Acc", marker="s")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)


def main():
    """Main Streamlit app."""
    st.title("🧪 Active Learning Experiment Dashboard")
    st.markdown("Compare and analyze your model training experiments")
    
    # Load experiments
    experiments = load_experiments()
    
    if not experiments:
        st.warning("⚠️ No experiments found. Run `python train.py --config <config>` to create experiments.")
        st.info("Experiments should be saved in the `experiments/` folder.")
        return
    
    st.success(f"✅ Found {len(experiments)} experiments")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["📊 Compare All", "🔍 Single Experiment"])
    
    if mode == "📊 Compare All":
        display_comparison_view(experiments)
    else:
        display_single_view(experiments)


def display_comparison_view(experiments):
    """Display comparison of all experiments."""
    st.header("Compare All Experiments")
    
    # Prepare comparison data
    comparison_data = []
    for exp_name, exp_data in experiments.items():
        metrics = exp_data["metrics"]
        comparison_data.append({
            "Experiment": exp_name,
            "Test Accuracy": metrics.get("test_accuracy", 0),
            "Test Precision": metrics.get("test_precision", 0),
            "Test Recall": metrics.get("test_recall", 0),
            "Test F1": metrics.get("test_f1", 0),
            "Best Epoch": metrics.get("best_epoch", 0),
            "Best Val Acc": metrics.get("best_val_accuracy", 0),
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display table
    st.subheader("Results Table")
    st.dataframe(df_comparison, use_container_width=True)
    
    # Display plots
    st.subheader("Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Test Accuracy Comparison**")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df_comparison.sort_values("Test Accuracy", ascending=False)
        ax.barh(df_sorted["Experiment"], df_sorted["Test Accuracy"], color="steelblue")
        ax.set_xlabel("Test Accuracy")
        ax.set_title("Test Accuracy by Experiment")
        st.pyplot(fig)
    
    with col2:
        st.write("**Test F1 Score Comparison**")
        fig, ax = plt.subplots(figsize=(10, 5))
        df_sorted = df_comparison.sort_values("Test F1", ascending=False)
        ax.barh(df_sorted["Experiment"], df_sorted["Test F1"], color="seagreen")
        ax.set_xlabel("Test F1 Score")
        ax.set_title("Test F1 Score by Experiment")
        st.pyplot(fig)
    
    # Export comparison
    st.subheader("Export")
    csv_data = df_comparison.to_csv(index=False)
    st.download_button(
        label="Download Comparison as CSV",
        data=csv_data,
        file_name=f"experiment_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def display_single_view(experiments):
    """Display single experiment details."""
    st.header("Single Experiment Analysis")
    
    # Select experiment
    exp_names = sorted(experiments.keys())
    selected_exp = st.selectbox("Select Experiment", exp_names)
    
    if selected_exp:
        exp_data = experiments[selected_exp]
        
        st.subheader(f"📋 {selected_exp}")
        
        # Summary metrics
        display_experiment_summary(selected_exp, exp_data)
        
        # Training curves
        st.subheader("Training Progress")
        plot_training_curves(exp_data["path"])
        
        # Config details
        st.subheader("Configuration")
        with open(exp_data["config_file"]) as f:
            config_content = f.read()
        st.code(config_content, language="yaml")
        
        # Files info
        st.subheader("Saved Files")
        exp_dir = exp_data["path"]
        files_info = {
            "metrics.json": "Test metrics and performance",
            "training_history.json": "Epoch-by-epoch training data",
            "training_log.txt": "Text log of training progress",
            "results_summary.json": "Complete experiment summary",
            "config.yaml": "Experiment configuration",
            "checkpoints/": "Saved model checkpoints",
        }
        
        for file_name, description in files_info.items():
            file_path = exp_dir / file_name
            if file_path.exists():
                st.success(f"✅ {file_name}: {description}")
            else:
                st.info(f"ℹ️ {file_name}: {description}")


if __name__ == "__main__":
    main()