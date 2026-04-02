"""
Training visualization view for threaded active learning.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from core.controller import Controller


def render_epoch_progress(snap: Dict[str, Any], total_epochs: int) -> None:
    current_epoch = int(snap.get("current_epoch", 0))
    epoch_progress = min(1.0, current_epoch / total_epochs) if total_epochs > 0 else 0.0
    st.progress(epoch_progress, text=f"Epoch {current_epoch} / {total_epochs}")


def render_cycle_progress(snap: Dict[str, Any]) -> None:
    current_cycle = snap.get("current_cycle", 0)
    total_cycles = snap.get("total_cycles", 0)
    cycles_completed = len(snap.get("metrics_history", []))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Current Cycle", value=f"{current_cycle} / {total_cycles}")
    with col2:
        st.metric(label="Completed Cycles", value=cycles_completed)
    with col3:
        completion = (cycles_completed / total_cycles * 100.0) if total_cycles else 0.0
        st.metric(label="Overall Progress", value=f"{completion:.1f}%")


def render_training_charts(epoch_metrics: List[Dict[str, Any]]) -> None:
    if not epoch_metrics:
        st.info("Training charts will appear once training starts...")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Training Loss")
        loss_data = {
            "Epoch": [m["epoch"] for m in epoch_metrics if "epoch" in m],
            "Train Loss": [m["train_loss"] for m in epoch_metrics if "train_loss" in m],
        }
        val_losses = [m.get("val_loss") for m in epoch_metrics if "train_loss" in m]
        if any(v is not None for v in val_losses):
            loss_data["Val Loss"] = [v if v is not None else float("nan") for v in val_losses]
        if loss_data["Epoch"]:
            st.line_chart(
                loss_data,
                x="Epoch",
                y=["Train Loss"] + (["Val Loss"] if "Val Loss" in loss_data else []),
                height=300,
            )

    with col2:
        st.markdown("#### Training Accuracy")
        acc_data = {
            "Epoch": [m["epoch"] for m in epoch_metrics if "epoch" in m],
            "Train Acc": [m["train_accuracy"] * 100 for m in epoch_metrics if "train_accuracy" in m],
        }
        val_accs = [m.get("val_accuracy") for m in epoch_metrics if "train_accuracy" in m]
        if any(v is not None for v in val_accs):
            acc_data["Val Acc"] = [v * 100 if v is not None else float("nan") for v in val_accs]
        if acc_data["Epoch"]:
            st.line_chart(
                acc_data,
                x="Epoch",
                y=["Train Acc"] + (["Val Acc"] if "Val Acc" in acc_data else []),
                height=300,
            )

    lr_epochs = [m["epoch"] for m in epoch_metrics if "epoch" in m and "learning_rate" in m]
    lr_values = [m["learning_rate"] for m in epoch_metrics if "epoch" in m and "learning_rate" in m]
    if lr_epochs and any(v is not None for v in lr_values):
        st.markdown("#### Learning Rate Schedule")
        st.line_chart(
            {"Epoch": lr_epochs, "Learning Rate": lr_values},
            x="Epoch",
            y="Learning Rate",
            height=200,
        )


def render_pool_statistics(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Data Pool Status")
    if not metrics_history:
        st.info("Pool statistics will appear after the first cycle completes")
        return

    latest = metrics_history[-1]
    labeled_size = latest.get("labeled_pool_size", 0)
    unlabeled_size = latest.get("unlabeled_pool_size", 0)
    total_size = labeled_size + unlabeled_size

    col1, col2, col3 = st.columns(3)
    with col1:
        pct = f"{(labeled_size / total_size * 100):.1f}%" if total_size else "0%"
        st.metric(label="Labeled Pool", value=labeled_size, delta=pct)
    with col2:
        pct = f"{(unlabeled_size / total_size * 100):.1f}%" if total_size else "0%"
        st.metric(label="Unlabeled Pool", value=unlabeled_size, delta=pct)
    with col3:
        st.metric(label="Total Samples", value=total_size)
    if total_size:
        st.progress(labeled_size / total_size, text=f"Labeled: {labeled_size / total_size * 100:.1f}%")


def render_current_metrics(epoch_metrics: List[Dict[str, Any]]) -> None:
    st.markdown("### Current Metrics")
    if not epoch_metrics:
        st.info("Metrics will appear once training starts...")
        return

    latest = epoch_metrics[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Train Loss", value=f"{latest.get('train_loss', 0.0):.4f}")
    with col2:
        st.metric(label="Train Accuracy", value=f"{latest.get('train_accuracy', 0.0) * 100:.2f}%")
    with col3:
        val_loss = latest.get("val_loss")
        st.metric(label="Val Loss", value=f"{val_loss:.4f}" if val_loss is not None else "N/A")
    with col4:
        val_acc = latest.get("val_accuracy")
        st.metric(label="Val Accuracy", value=f"{val_acc * 100:.2f}%" if val_acc is not None else "N/A")
    if latest.get("learning_rate") is not None:
        st.caption(f"Learning Rate: {latest['learning_rate']:.2e}")


def render_training_view(controller: Controller, snap: Dict[str, Any]) -> None:
    st.title("Training in Progress")
    st.markdown("---")

    render_cycle_progress(snap)
    st.markdown("---")

    st.markdown("### Epoch Progress")
    render_epoch_progress(snap, int(controller.config.training.epochs))
    st.markdown("---")

    st.markdown("### Training Metrics")
    render_training_charts(snap.get("epoch_metrics", []))
    st.markdown("---")

    render_current_metrics(snap.get("epoch_metrics", []))
    st.markdown("---")

    render_pool_statistics(snap.get("metrics_history", []))
    st.caption(snap.get("progress_detail", ""))
