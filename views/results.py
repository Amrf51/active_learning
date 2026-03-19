"""
Results dashboard for completed/ongoing active learning runs.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from controller import Controller


def _load_display_image(image_path: str) -> Image.Image:
    """Open image for Streamlit display and normalize problematic palette mode."""
    img = Image.open(image_path)
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def _safe_read_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
            return payload if isinstance(payload, dict) else {}
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


@st.cache_data(show_spinner=False, ttl=3)
def _discover_persisted_runs(exp_dir: str) -> List[Dict[str, Any]]:
    """
    Discover runs from experiment folders on disk.

    A run folder is considered valid when ``config.yaml`` exists.
    """
    root = Path(exp_dir)
    if not root.exists():
        return []

    def _class_names_from_metrics(metrics: List[Dict[str, Any]]) -> List[str]:
        if not metrics:
            return []
        latest = metrics[-1]
        if not isinstance(latest, dict):
            return []
        per_class = latest.get("per_class")
        if isinstance(per_class, dict):
            return [str(name) for name in per_class.keys()]
        return []

    runs: List[Dict[str, Any]] = []
    for run_dir in root.glob("*/*"):
        if not run_dir.is_dir():
            continue
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            continue

        results_payload = _safe_read_json(run_dir / "al_cycle_results.json")
        config_payload = _safe_read_yaml(config_file)

        metrics_history = results_payload.get("cycles", [])
        if not isinstance(metrics_history, list):
            metrics_history = []

        exp_cfg = config_payload.get("experiment", {})
        model_cfg = config_payload.get("model", {})
        al_cfg = config_payload.get("active_learning", {})

        experiment_name = str(exp_cfg.get("name", run_dir.parent.name))
        model_name = str(model_cfg.get("name", "unknown"))
        strategy = str(results_payload.get("strategy", al_cfg.get("sampling_strategy", "unknown")))

        configured_cycles = results_payload.get("num_cycles", al_cfg.get("num_cycles", 0))
        try:
            configured_cycles = int(configured_cycles)
        except (TypeError, ValueError):
            configured_cycles = 0

        completed_cycles = len(metrics_history)
        last_test_acc = None
        if metrics_history and isinstance(metrics_history[-1], dict):
            last_test_acc = metrics_history[-1].get("test_accuracy")

        mtime = run_dir.stat().st_mtime
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

        status = "empty"
        if completed_cycles > 0:
            status = "partial"
        if configured_cycles > 0 and completed_cycles >= configured_cycles:
            status = "finished"

        run_key = str(run_dir.resolve())
        cycle_part = f"{completed_cycles}/{configured_cycles}" if configured_cycles > 0 else str(completed_cycles)
        label = (
            f"{timestamp} | {experiment_name}/{run_dir.name} | "
            f"{model_name} | {strategy} | cycles {cycle_part} | {status}"
        )
        if isinstance(last_test_acc, (int, float)):
            label = f"{label} | last acc {(float(last_test_acc) * 100):.1f}%"

        runs.append(
            {
                "key": run_key,
                "label": label,
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "experiment_name": experiment_name,
                "model_name": model_name,
                "strategy": strategy,
                "status": status,
                "configured_cycles": configured_cycles,
                "completed_cycles": completed_cycles,
                "metrics_history": metrics_history,
                "class_names": _class_names_from_metrics(metrics_history),
                "last_test_acc": last_test_acc,
                "modified_at": timestamp,
                "mtime": mtime,
            }
        )

    runs.sort(key=lambda item: float(item.get("mtime", 0.0)), reverse=True)
    return runs


def _widget_key_prefix(raw_value: str) -> str:
    """Convert arbitrary identifier into a Streamlit-safe widget key prefix."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in raw_value)
    cleaned = cleaned.strip("_")
    return cleaned or "results_source"


def render_metrics_table(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Cycle Summary")
    if not metrics_history:
        st.info("Metrics table will appear after cycles complete")
        return

    rows = []
    for metrics in metrics_history:
        ece = metrics.get("ece")
        rows.append(
            {
                "Cycle": metrics.get("cycle", "N/A"),
                "Labeled": metrics.get("labeled_pool_size", 0),
                "Val Acc": f"{metrics.get('best_val_accuracy', 0) * 100:.2f}%",
                "Test Acc": f"{metrics.get('test_accuracy', 0) * 100:.2f}%",
                "F1": f"{metrics.get('test_f1', 0):.3f}",
                "Precision": f"{metrics.get('test_precision', 0):.3f}",
                "Recall": f"{metrics.get('test_recall', 0):.3f}",
                "ECE": f"{ece:.4f}" if ece is not None else "N/A",
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


def render_ece_chart(metrics_history: List[Dict[str, Any]]) -> None:
    st.markdown("### Calibration (ECE) Across Cycles")
    ece_rows = [
        {"Cycle": m.get("cycle", i + 1), "ECE": m["ece"]}
        for i, m in enumerate(metrics_history)
        if m.get("ece") is not None
    ]
    if not ece_rows:
        st.info("ECE values will appear after cycles complete (requires calibration to be enabled)")
        return
    df = pd.DataFrame(ece_rows)
    st.line_chart(df, x="Cycle", y="ECE", height=250)


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


def render_probe_predictions(
    metrics_history: List[Dict[str, Any]],
    snap: Dict[str, Any],
    widget_prefix: str = "live",
) -> None:
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
        key=f"{widget_prefix}_results_probe_cycle",
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


def render_confusion_matrix(
    metrics_history: List[Dict[str, Any]],
    snap: Dict[str, Any],
    widget_prefix: str = "live",
) -> None:
    st.markdown("### Confusion Matrix")
    if not metrics_history:
        st.info("Confusion matrix will appear after at least one cycle completes")
        return

    cycle_options = [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)]
    selected_cycle = st.selectbox(
        "Cycle",
        options=cycle_options,
        index=len(cycle_options) - 1,
        key=f"{widget_prefix}_results_confusion_matrix_cycle",
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


def _resolve_embeddings_path(metric: Dict[str, Any], run_dir: str) -> Path | None:
    raw_path = metric.get("embeddings_path")
    if raw_path:
        path = Path(str(raw_path))
        if path.exists():
            return path

    cycle = metric.get("cycle")
    if run_dir and cycle is not None:
        fallback = Path(run_dir) / "embeddings" / f"cycle_{cycle}.npz"
        if fallback.exists():
            return fallback

    return None


def render_embedding_plot(
    metrics_history: List[Dict[str, Any]],
    snap: Dict[str, Any],
    widget_prefix: str = "live",
) -> None:
    st.markdown("### UMAP Embedding Visualization")
    if not metrics_history:
        st.info("Embedding plot will appear after at least one cycle completes")
        return

    cycle_options = [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)]
    selected_cycle = st.selectbox(
        "Cycle",
        options=cycle_options,
        index=len(cycle_options) - 1,
        key=f"{widget_prefix}_embedding_cycle",
    )
    selected_metric = next(
        (m for m in metrics_history if m.get("cycle", None) == selected_cycle),
        metrics_history[-1],
    )

    emb_path = _resolve_embeddings_path(selected_metric, str(snap.get("run_dir", "")))
    if emb_path is None:
        st.info("Embedding file not found for the selected cycle (umap-learn may not have been installed during the run).")
        return

    try:
        import numpy as np
        import plotly.graph_objects as go

        data = np.load(emb_path)
        coords = data["coords"]   # [N, 2]
        labels = data["labels"]   # [N] int
        pool = data["pool"]       # [N] int8: 0=labeled, 1=unlabeled
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Failed to load embedding file: {exc}")
        return

    color_mode = st.radio(
        "Color by",
        options=["Class Label", "Pool Membership"],
        horizontal=True,
        key=f"{widget_prefix}_embedding_color_mode",
    )

    fig = go.Figure()

    if color_mode == "Pool Membership":
        pool_labels = {0: "Labeled", 1: "Unlabeled", 2: "Queried"}
        pool_colors = {0: "#2196F3", 1: "#BDBDBD", 2: "#FF5722"}
        for pool_val, pool_name in pool_labels.items():
            mask = pool == pool_val
            if not mask.any():
                continue
            fig.add_trace(
                go.Scattergl(
                    x=coords[mask, 0].tolist(),
                    y=coords[mask, 1].tolist(),
                    mode="markers",
                    name=pool_name,
                    legendgroup=pool_name,
                    marker=dict(size=4, opacity=0.7, color=pool_colors[pool_val]),
                )
            )
    else:
        class_names = list(snap.get("class_names", []))
        unique_labels = sorted(set(int(l) for l in labels))
        for label_idx in unique_labels:
            mask = labels == label_idx
            name = class_names[label_idx] if label_idx < len(class_names) else str(label_idx)
            fig.add_trace(
                go.Scattergl(
                    x=coords[mask, 0].tolist(),
                    y=coords[mask, 1].tolist(),
                    mode="markers",
                    name=name,
                    legendgroup=name,
                    marker=dict(size=4, opacity=0.7),
                )
            )

    fig.update_layout(
        title=f"UMAP — Cycle {selected_cycle} ({len(coords)} points)",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend=dict(itemsizing="constant"),
        height=550,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Source: {emb_path}")


def render_results_view(controller: Controller, snap: Dict[str, Any]) -> None:
    st.title("Results Dashboard")
    st.markdown("---")

    exp_dir = str(getattr(controller.config.experiment, "exp_dir", "experiments"))
    if st.button("Refresh Runs", key="results_refresh_runs"):
        _discover_persisted_runs.clear()
        st.rerun()

    persisted_runs = _discover_persisted_runs(exp_dir)

    source_by_key: Dict[str, Dict[str, Any]] = {}
    source_keys: List[str] = []

    for run in persisted_runs:
        run_key = str(run["key"])
        source_by_key[run_key] = {
            "label": str(run["label"]),
            "metrics_history": list(run.get("metrics_history", [])),
            "snap": {
                "run_dir": str(run.get("run_dir", "")),
                "class_names": list(run.get("class_names", [])),
                "probe_images": [],
            },
            "metadata": run,
        }
        source_keys.append(run_key)

    if not source_keys:
        st.info("No run folders found yet. Start an experiment to create artifacts.")
        return

    default_index = 0
    current_run_dir = str(snap.get("run_dir", "")).strip()
    if current_run_dir:
        for idx, key in enumerate(source_keys):
            run_meta = source_by_key[key]["metadata"]
            if str(run_meta.get("run_dir", "")) == current_run_dir:
                default_index = idx
                break

    if len(source_keys) == 1:
        selected_source_key = source_keys[0]
    else:
        selected_source_key = st.selectbox(
            "Experiment Run",
            options=source_keys,
            index=default_index,
            format_func=lambda key: str(source_by_key[key]["label"]),
            key="results_source_selector",
        )

    selected_source = source_by_key[selected_source_key]
    selected_snap = dict(selected_source["snap"])
    metrics_history = list(selected_source["metrics_history"])
    widget_prefix = _widget_key_prefix(selected_source_key)

    metadata = dict(selected_source.get("metadata", {}))
    st.caption(
        f"Run: {metadata.get('experiment_name', 'unknown')}/{metadata.get('run_name', 'unknown')}"
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model", metadata.get("model_name", "N/A"))
    with c2:
        st.metric("Strategy", metadata.get("strategy", "N/A"))
    with c3:
        st.metric(
            "Cycles",
            f"{metadata.get('completed_cycles', 0)}/{metadata.get('configured_cycles', 0)}",
        )
    with c4:
        st.metric("Status", str(metadata.get("status", "unknown")).upper())
    if metadata.get("run_dir"):
        st.caption(f"Run directory: {metadata['run_dir']}")
    st.markdown("---")

    if not metrics_history:
        st.info("No cycle metrics available yet for the selected run.")
        return

    render_accuracy_progression_chart(metrics_history)
    st.markdown("---")
    render_ece_chart(metrics_history)
    st.markdown("---")
    render_metrics_table(metrics_history)
    st.markdown("---")
    render_best_cycle_summary(metrics_history)
    st.markdown("---")
    render_probe_predictions(metrics_history, selected_snap, widget_prefix=widget_prefix)
    st.markdown("---")
    render_confusion_matrix(metrics_history, selected_snap, widget_prefix=widget_prefix)
    st.markdown("---")
    render_embedding_plot(metrics_history, selected_snap, widget_prefix=widget_prefix)
    st.markdown("---")
