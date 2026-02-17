"""
Results dashboard for completed/ongoing active learning runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from PIL import Image

from controller import Controller


def _load_display_image(image_path: str) -> Image.Image:
    """Open image for Streamlit display and normalize problematic palette mode."""
    img = Image.open(image_path)
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


def render_metrics_table(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Cycle Summary")
    if not metrics_history:
        st.info("Metrics table will appear after cycles complete")
        return

    rows = []
    for metrics in metrics_history:
        rows.append(
            {
                "Cycle": metrics.get("cycle", "N/A"),
                "Labeled": metrics.get("labeled_pool_size", 0),
                "Val Acc": f"{metrics.get('best_val_accuracy', 0) * 100:.2f}%",
                "Test Acc": f"{metrics.get('test_accuracy', 0) * 100:.2f}%",
                "F1": f"{metrics.get('test_f1', 0):.3f}",
                "Precision": f"{metrics.get('test_precision', 0):.3f}",
                "Recall": f"{metrics.get('test_recall', 0):.3f}",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch", hide_index=True)

    latest = metrics_history[-1]
    labeled = latest.get("labeled_pool_size", 0)
    unlabeled = latest.get("unlabeled_pool_size", 0)
    total = labeled + unlabeled
    if total > 0:
        st.caption(
            f"Total samples: {total} | Labeled: {labeled} ({labeled / total * 100:.1f}%) | "
            f"Unlabeled: {unlabeled} ({unlabeled / total * 100:.1f}%)"
        )


def render_accuracy_progression_chart(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Test Accuracy Progression")
    if not metrics_history:
        st.info("Accuracy chart will appear after cycles complete")
        return

    df = pd.DataFrame(
        {
            "Cycle": [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)],
            "Test Accuracy": [m.get("test_accuracy", 0) * 100 for m in metrics_history],
            "Val Accuracy": [m.get("best_val_accuracy", 0) * 100 for m in metrics_history],
        }
    )
    st.line_chart(df, x="Cycle", y=["Test Accuracy", "Val Accuracy"], height=400)

    if len(metrics_history) >= 2:
        first_acc = metrics_history[0].get("test_accuracy", 0) * 100
        last_acc = metrics_history[-1].get("test_accuracy", 0) * 100
        improvement = last_acc - first_acc
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Initial Test Acc", value=f"{first_acc:.2f}%")
        with col2:
            st.metric(label="Final Test Acc", value=f"{last_acc:.2f}%")
        with col3:
            st.metric(label="Improvement", value=f"+{improvement:.2f}%", delta=f"{improvement:.2f}%")


def render_best_cycle_summary(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Best Cycle Summary")
    if not metrics_history:
        st.info("Best cycle summary will appear after cycles complete")
        return

    best_idx = max(range(len(metrics_history)), key=lambda i: metrics_history[i].get("test_accuracy", 0))
    best = metrics_history[best_idx]
    st.success(f"Best Performance: Cycle {best.get('cycle', best_idx + 1)}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Test Accuracy", value=f"{best.get('test_accuracy', 0) * 100:.2f}%")
    with col2:
        st.metric(label="F1 Score", value=f"{best.get('test_f1', 0):.3f}")
    with col3:
        st.metric(label="Precision", value=f"{best.get('test_precision', 0):.3f}")
    with col4:
        st.metric(label="Recall", value=f"{best.get('test_recall', 0):.3f}")

    d1, d2, d3 = st.columns(3)
    with d1:
        st.write(f"Labeled Pool Size: {best.get('labeled_pool_size', 0)}")
    with d2:
        st.write(f"Epochs Trained: {best.get('epochs_trained', 0)}")
    with d3:
        st.write(f"Best Epoch: {best.get('best_epoch', 0)}")


def _prediction_for_cycle(predictions_by_cycle: Dict[str, Any], cycle: int) -> Dict[str, Any] | None:
    if cycle in predictions_by_cycle:
        value = predictions_by_cycle.get(cycle)
        return dict(value) if isinstance(value, dict) else None
    value = predictions_by_cycle.get(str(cycle))
    return dict(value) if isinstance(value, dict) else None


def render_probe_predictions(metrics_history: List[Dict[str, Any]], snap: Dict[str, Any]) -> None:
    st.markdown("### Probe Predictions")
    probe_images = list(snap.get("probe_images", []))
    if not probe_images:
        st.info("Probe predictions will appear after cycle evaluation completes.")
        return

    cycle_options = [int(m.get("cycle", i + 1)) for i, m in enumerate(metrics_history)]
    selected_cycle = st.selectbox(
        "Probe Cycle",
        options=cycle_options,
        index=len(cycle_options) - 1,
        key="results_probe_cycle",
    )

    table_rows: List[Dict[str, Any]] = []
    rendered_probes: List[Dict[str, Any]] = []
    correct_count = 0
    for probe in probe_images:
        predictions = probe.get("predictions_by_cycle", {})
        if not isinstance(predictions, dict):
            continue
        cycle_prediction = _prediction_for_cycle(predictions, selected_cycle)
        if cycle_prediction is None:
            continue

        true_class = str(probe.get("true_class", probe.get("true_class_idx", "N/A")))
        predicted_class = str(cycle_prediction.get("predicted_class", "N/A"))
        confidence = float(cycle_prediction.get("confidence", 0.0))
        is_correct = predicted_class == true_class
        if is_correct:
            correct_count += 1

        table_rows.append(
            {
                "Probe ID": probe.get("image_id", "N/A"),
                "True Class": true_class,
                "Predicted Class": predicted_class,
                "Confidence": f"{confidence * 100:.1f}%",
                "Correct": "Yes" if is_correct else "No",
            }
        )
        rendered_probes.append(
            {
                "probe": probe,
                "prediction": cycle_prediction,
                "correct": is_correct,
            }
        )

    if not table_rows:
        st.info("No probe predictions recorded for this cycle yet.")
        return

    total = len(table_rows)
    accuracy = (correct_count / total * 100.0) if total else 0.0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probes", total)
    with col2:
        st.metric("Correct", correct_count)
    with col3:
        st.metric("Probe Accuracy", f"{accuracy:.1f}%")

    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    with st.expander("Probe Image Cards"):
        num_cols = 4
        for i, entry in enumerate(rendered_probes):
            if i % num_cols == 0:
                cols = st.columns(num_cols)
            with cols[i % num_cols]:
                probe = entry["probe"]
                prediction = entry["prediction"]
                is_correct = bool(entry["correct"])

                image_path = str(probe.get("display_path") or probe.get("image_path") or "")
                if image_path and Path(image_path).exists():
                    st.image(_load_display_image(image_path), width="stretch")

                true_class = str(probe.get("true_class", probe.get("true_class_idx", "N/A")))
                predicted_class = str(prediction.get("predicted_class", "N/A"))
                confidence = float(prediction.get("confidence", 0.0))
                verdict = "Correct" if is_correct else "Wrong"
                st.caption(f"ID: {probe.get('image_id', 'N/A')}")
                st.caption(f"True: {true_class}")
                st.caption(f"Pred: {predicted_class} ({confidence * 100:.1f}%)")
                st.caption(verdict)


def _resolve_confusion_matrix_path(metric: Dict[str, Any], run_dir: str) -> Path | None:
    raw_path = metric.get("confusion_matrix_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path

    cycle = metric.get("cycle")
    if run_dir and cycle is not None:
        fallback = Path(run_dir) / "confusion_matrices" / f"cycle_{cycle}.npy"
        if fallback.exists():
            return fallback

    return None


def render_confusion_matrix(metrics_history: List[Dict[str, Any]], snap: Dict[str, Any]) -> None:
    st.markdown("### Confusion Matrix")
    if not metrics_history:
        st.info("Confusion matrix will appear after at least one cycle completes")
        return

    cycle_options = [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)]
    selected_cycle = st.selectbox(
        "Cycle",
        options=cycle_options,
        index=len(cycle_options) - 1,
        key="results_confusion_matrix_cycle",
    )
    selected_metric = next(
        (m for m in metrics_history if m.get("cycle", None) == selected_cycle),
        metrics_history[-1],
    )

    cm_path = _resolve_confusion_matrix_path(selected_metric, str(snap.get("run_dir", "")))
    if cm_path is None:
        st.info("Confusion matrix file not found for selected cycle.")
        return

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        cm = np.load(cm_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Failed to load confusion matrix: {exc}")
        return

    if cm.ndim != 2:
        st.error(f"Invalid confusion matrix shape: {cm.shape}")
        return

    num_classes = int(cm.shape[0])
    class_names = list(snap.get("class_names", []))
    if len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    fig_size = min(14, max(6, num_classes * 0.7))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Cycle {selected_cycle} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    if num_classes <= 25:
        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(num_classes):
            for j in range(num_classes):
                value = int(cm[i, j])
                color = "white" if value > threshold else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    st.caption(f"Source: {cm_path}")


def render_results_view(controller: Controller, snap: Dict[str, Any]) -> None:
    st.title("Results Dashboard")
    st.markdown("---")

    metrics_history = snap.get("metrics_history", [])
    if not metrics_history:
        st.info("Results will appear here after completing active learning cycles.")
        return

    render_accuracy_progression_chart(metrics_history)
    st.markdown("---")
    render_metrics_table(metrics_history)
    st.markdown("---")
    render_best_cycle_summary(metrics_history)
    st.markdown("---")
    render_probe_predictions(metrics_history, snap)
    st.markdown("---")
    render_confusion_matrix(metrics_history, snap)
    st.markdown("---")

    if st.button("Export Results (JSON)"):
        controller.save_state()
        st.success("Saved snapshot to state.json")
