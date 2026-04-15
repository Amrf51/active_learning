"""
Results dashboard for completed/ongoing active learning runs.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import yaml
from PIL import Image

from core.controller import Controller


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


@st.cache_data(show_spinner=False, ttl=30)
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
        per_class = latest.get("per_class") or latest.get("per_class_metrics")
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
        ece_cal = metrics.get("ece_calibrated")
        temp = metrics.get("temperature")
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
                "ECE (cal.)": f"{ece_cal:.4f}" if ece_cal is not None else "N/A",
                "Temp.": f"{temp:.2f}" if temp is not None else "N/A",
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
    ece_rows = []
    for i, m in enumerate(metrics_history):
        if m.get("ece") is None:
            continue
        row: Dict[str, Any] = {"Cycle": m.get("cycle", i + 1), "ECE (raw)": m["ece"]}
        if m.get("ece_calibrated") is not None:
            row["ECE (calibrated)"] = m["ece_calibrated"]
        ece_rows.append(row)
    if not ece_rows:
        st.info("ECE values will appear after cycles complete (requires calibration to be enabled)")
        return
    df = pd.DataFrame(ece_rows)
    y_cols = ["ECE (raw)"]
    if "ECE (calibrated)" in df.columns:
        y_cols.append("ECE (calibrated)")
    st.line_chart(df, x="Cycle", y=y_cols, height=250)

    # Show temperature evolution if available
    temp_rows = [
        {"Cycle": m.get("cycle", i + 1), "Temperature": m["temperature"]}
        for i, m in enumerate(metrics_history)
        if m.get("temperature") is not None
    ]
    if temp_rows:
        with st.expander("Learned Temperature per Cycle"):
            temp_df = pd.DataFrame(temp_rows)
            st.line_chart(temp_df, x="Cycle", y="Temperature", height=200)
            latest_t = temp_rows[-1]["Temperature"]
            st.caption(
                f"T > 1.0 = model was overconfident (softened), "
                f"T < 1.0 = model was underconfident (sharpened). "
                f"Latest T = {latest_t:.4f}"
            )


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


_BODY_TYPE_KEYWORDS = [
    ("Convertible", ["convertible"]),
    ("Coupe", ["coupe"]),
    ("Wagon", ["wagon", "estate"]),
    ("Van", ["cargo van", "minivan", " van"]),
    ("Cab", [
        "regular cab", "extended cab", "supercrew cab", "crew cab",
        "club cab", "quad cab", "access cab", "king cab", "double cab",
    ]),
    ("Truck", ["truck", "pickup"]),
    ("Hatchback", ["hatchback"]),
    ("SUV", ["suv", "4wd", "awd"]),
    ("Sedan", ["sedan"]),
]
_BODY_TYPE_FALLBACK = "Other"


def _class_to_body_type(class_name: str) -> str:
    lower = class_name.lower()
    for body_type, keywords in _BODY_TYPE_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return body_type
    return _BODY_TYPE_FALLBACK


def _build_body_type_matrix(cm, class_names: List[str]):
    import numpy as np

    labels = [_class_to_body_type(n) for n in class_names]
    # Preserve encounter order so the matrix rows/cols match natural grouping
    seen: dict = {}
    for lbl in labels:
        if lbl not in seen:
            seen[lbl] = len(seen)
    body_types = list(seen.keys())
    n = len(body_types)
    agg = np.zeros((n, n), dtype=np.int64)
    for r, lbl_r in enumerate(labels):
        for c, lbl_c in enumerate(labels):
            agg[seen[lbl_r], seen[lbl_c]] += cm[r, c]
    return agg, body_types


def _render_top_confused_pairs(cm, class_names: List[str], top_k: int) -> None:
    import numpy as np

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    flat_idx = np.argsort(cm_no_diag.ravel())[::-1][:top_k]
    rows, cols = np.unravel_index(flat_idx, cm.shape)

    records = []
    for rank, (r, c) in enumerate(zip(rows, cols), 1):
        count = int(cm_no_diag[r, c])
        if count == 0:
            break
        records.append({
            "Rank": rank,
            "True class": class_names[r],
            "Predicted as": class_names[c],
            "Count": count,
        })

    if not records:
        st.info("No off-diagonal confusions found.")
        return

    st.dataframe(pd.DataFrame(records), width='stretch', hide_index=True)


def _render_body_type_matrix(cm, class_names: List[str], cycle: int) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    agg, body_types = _build_body_type_matrix(cm, class_names)
    row_sums = agg.sum(axis=1, keepdims=True).clip(1)
    norm = agg / row_sums

    n = len(body_types)
    fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    threshold = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if norm[i, j] > threshold else "black"
            ax.text(j, i, str(int(agg[i, j])), ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_xticks(range(n))
    ax.set_xticklabels(body_types, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(body_types)
    ax.set_xlabel("Predicted body type")
    ax.set_ylabel("True body type")
    ax.set_title(f"Cycle {cycle} — confusion by body type (row-normalised)")
    fig.colorbar(im, ax=ax, label="Recall within true body type", fraction=0.046, pad=0.04)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Show how many classes landed in each bucket
    from collections import Counter as _Counter
    bucket_counts = _Counter(_class_to_body_type(cls) for cls in class_names)
    st.caption(
        "Classes per bucket: "
        + "  |  ".join(f"{bt}: {bucket_counts[bt]}" for bt in body_types)
    )


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

    with st.expander("Most confused pairs", expanded=True):
        top_k = st.slider(
            "Show top N pairs",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key=f"{widget_prefix}_cm_top_k",
        )
        _render_top_confused_pairs(cm, class_names, top_k)

    with st.expander("Confusion by body type", expanded=True):
        _render_body_type_matrix(cm, class_names, selected_cycle)

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


def _build_umap_figure(
    coords,
    labels,
    pool,
    class_names: List[str],
    color_mode: str,
    title: str,
    height: int = 550,
    show_legend: bool = True,
    uncertainty=None,
):
    """Build a Plotly Figure for a single UMAP embedding snapshot."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if color_mode == "Pool Membership":
        pool_labels = {0: "Labeled", 1: "Unlabeled", 2: "Queried"}
        pool_colors = {0: "#2196F3", 1: "#BDBDBD", 2: "#FF5722"}
        pool_sizes = {0: 4, 1: 4, 2: 6}
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
                    marker=dict(
                        size=pool_sizes[pool_val],
                        opacity=0.7,
                        color=pool_colors[pool_val],
                    ),
                )
            )
    elif color_mode == "Uncertainty" and uncertainty is not None:
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0].tolist(),
                y=coords[:, 1].tolist(),
                mode="markers",
                name="Uncertainty",
                marker=dict(
                    size=4,
                    opacity=0.7,
                    color=uncertainty.tolist(),
                    colorscale="Plasma",
                    colorbar=dict(title="Entropy", thickness=14),
                    showscale=True,
                ),
            )
        )
    else:
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
        title=title,
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend=dict(itemsizing="constant"),
        showlegend=show_legend,
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


@st.cache_data(show_spinner=False, ttl=60)
def _load_all_embeddings(run_dir: str, cycle_numbers: tuple) -> Dict[int, Dict]:
    """Load embedding .npz files for multiple cycles (cached)."""
    import numpy as np

    result: Dict[int, Dict] = {}
    for cycle in cycle_numbers:
        path = Path(run_dir) / "embeddings" / f"cycle_{cycle}.npz"
        if not path.exists():
            continue
        try:
            data = np.load(path)
            result[cycle] = {
                "coords": data["coords"],
                "labels": data["labels"],
                "pool": data["pool"],
                "uncertainty": data["uncertainty"] if "uncertainty" in data else None,
            }
        except Exception:
            continue
    return result


def render_query_summary(
    metrics_history: List[Dict[str, Any]],
    snap: Dict[str, Any],
    widget_prefix: str = "live",
) -> None:
    """Display per-cycle query summary: strategy, distributions, top uncertain."""
    import numpy as np
    import plotly.graph_objects as go

    st.markdown("### Query Summary")

    if not metrics_history:
        st.info("Query summaries will appear after the first query cycle completes.")
        return

    run_dir = str(snap.get("run_dir", ""))
    cycle_options = [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)]
    selected_cycle = st.selectbox(
        "Cycle",
        options=cycle_options,
        index=len(cycle_options) - 1,
        key=f"{widget_prefix}_query_summary_cycle",
    )

    summary_path = Path(run_dir) / f"cycle_{selected_cycle}_query_summary.json"
    if not summary_path.exists():
        st.info(f"No query summary found for cycle {selected_cycle}.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Strategy info
    st.info(
        f"**Strategy: {summary.get('strategy_name', 'N/A')}** — "
        f"{summary.get('strategy_description', '')}"
    )

    # Key metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Candidates Ranked", summary.get("pool_size_before_query", "N/A"))
    with c2:
        st.metric("Images Queried", summary.get("n_queried", "N/A"))
    with c3:
        stats = summary.get("uncertainty_stats", {})
        st.metric("Mean Uncertainty", f"{stats.get('mean', 0):.4f}")

    # Uncertainty stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Min Uncertainty", f"{stats.get('min', 0):.4f}")
    with c2:
        st.metric("Max Uncertainty", f"{stats.get('max', 0):.4f}")
    with c3:
        st.metric("Mean", f"{stats.get('mean', 0):.4f}")
    with c4:
        st.metric("Std Dev", f"{stats.get('std', 0):.4f}")

    # Grouped bar chart: queried vs labeled class distribution
    queried_dist = summary.get("queried_class_distribution", {})
    labeled_dist = summary.get("labeled_class_distribution", {})
    all_classes = sorted(set(list(queried_dist.keys()) + list(labeled_dist.keys())))

    if all_classes:
        # Normalize labeled distribution to percentages for fair comparison
        labeled_total = sum(labeled_dist.values()) or 1
        queried_total = sum(queried_dist.values()) or 1

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Queried Batch (%)",
            x=all_classes,
            y=[100.0 * queried_dist.get(c, 0) / queried_total for c in all_classes],
            marker_color="#FF5722",
        ))
        fig.add_trace(go.Bar(
            name="Labeled Pool (%)",
            x=all_classes,
            y=[100.0 * labeled_dist.get(c, 0) / labeled_total for c in all_classes],
            marker_color="#2196F3",
        ))
        fig.update_layout(
            barmode="group",
            title="Class Distribution: Queried Batch vs. Labeled Pool",
            xaxis_title="Class",
            yaxis_title="Percentage (%)",
            height=400,
            margin=dict(l=40, r=20, t=50, b=100),
            xaxis_tickangle=-45 if len(all_classes) > 10 else 0,
        )
        st.plotly_chart(fig, width='stretch')

    # Top uncertain images table
    top_uncertain = summary.get("top_uncertain", [])
    if top_uncertain:
        with st.expander(f"Top {len(top_uncertain)} Most Uncertain Images"):
            rows = []
            for rank, item in enumerate(top_uncertain, 1):
                rows.append({
                    "Rank": rank,
                    "Image ID": item.get("image_id"),
                    "Predicted Class": item.get("predicted_class"),
                    "Confidence": f"{item.get('predicted_confidence', 0):.4f}",
                    "Uncertainty": f"{item.get('uncertainty_score', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


def render_umap_evolution(
    metrics_history: List[Dict[str, Any]],
    snap: Dict[str, Any],
    widget_prefix: str = "live",
) -> None:
    """Show UMAP embeddings across cycles with queried-point highlighting."""
    import numpy as np

    st.markdown("### UMAP Evolution Across Cycles")

    run_dir = str(snap.get("run_dir", ""))
    if not run_dir or not metrics_history:
        st.info("UMAP evolution will appear after at least two cycles complete.")
        return

    # Determine which cycles have embedding files
    all_cycles = [m.get("cycle", i + 1) for i, m in enumerate(metrics_history)]
    available_cycles = [
        c for c in all_cycles
        if (Path(run_dir) / "embeddings" / f"cycle_{c}.npz").exists()
    ]

    if len(available_cycles) < 2:
        st.info("At least 2 cycles with embeddings are needed for the evolution view.")
        return

    class_names = list(snap.get("class_names", []))

    # Peek at available cycles to know whether any have uncertainty data.
    # Load all available embeddings once so we can check the uncertainty field.
    preview_data = _load_all_embeddings(run_dir, tuple(available_cycles))
    has_uncertainty = any(
        preview_data.get(c, {}).get("uncertainty") is not None
        for c in available_cycles
    )

    # Color mode — "Uncertainty" only shown when data is present in the .npz files
    color_options = ["Pool Membership"]
    if has_uncertainty:
        color_options.append("Uncertainty")
    color_options.append("Class Label")

    color_mode = st.radio(
        "Color by",
        options=color_options,
        horizontal=True,
        key=f"{widget_prefix}_umap_evo_color_mode",
    )

    # Display mode
    display_mode = st.radio(
        "Display mode",
        options=["Slider", "Side-by-side"],
        horizontal=True,
        key=f"{widget_prefix}_umap_evo_display_mode",
    )

    if display_mode == "Slider":
        selected_cycle = st.select_slider(
            "Cycle",
            options=available_cycles,
            value=available_cycles[-1],
            key=f"{widget_prefix}_umap_evo_slider",
        )
        emb_data = _load_all_embeddings(run_dir, tuple([selected_cycle]))
        if selected_cycle not in emb_data:
            st.warning(f"Failed to load embeddings for cycle {selected_cycle}.")
            return
        d = emb_data[selected_cycle]
        title = f"UMAP — Cycle {selected_cycle} ({len(d['coords'])} points)"
        fig = _build_umap_figure(
            d["coords"], d["labels"], d["pool"],
            class_names, color_mode, title,
            uncertainty=d.get("uncertainty"),
        )
        st.plotly_chart(fig, width='stretch')

        n_queried = int(np.sum(d["pool"] == 2))
        if n_queried > 0:
            st.caption(f"{n_queried} queried points highlighted (from previous cycle's query).")

    else:  # Side-by-side
        max_cols = 4
        selected_cycles = st.multiselect(
            "Select cycles to compare",
            options=available_cycles,
            default=available_cycles[:min(max_cols, len(available_cycles))],
            key=f"{widget_prefix}_umap_evo_multi",
        )
        if not selected_cycles:
            st.info("Select at least one cycle.")
            return
        selected_cycles = selected_cycles[:max_cols]

        emb_data = _load_all_embeddings(run_dir, tuple(selected_cycles))
        cols = st.columns(len(selected_cycles))
        for i, (col, cyc) in enumerate(zip(cols, selected_cycles)):
            with col:
                if cyc not in emb_data:
                    st.warning(f"Cycle {cyc}: no data")
                    continue
                d = emb_data[cyc]
                title = f"Cycle {cyc}"
                show_legend = (i == 0) and color_mode == "Pool Membership"
                fig = _build_umap_figure(
                    d["coords"], d["labels"], d["pool"],
                    class_names, color_mode, title,
                    height=400, show_legend=show_legend,
                    uncertainty=d.get("uncertainty"),
                )
                st.plotly_chart(fig, width='stretch')
                n_queried = int(np.sum(d["pool"] == 2))
                if n_queried > 0:
                    st.caption(f"{n_queried} queried pts")


def render_comparison_view(controller: Controller, snap: Dict[str, Any]) -> None:
    """Compare multiple runs side-by-side: accuracy, F1, ECE curves."""
    st.title("Run Comparison")
    st.markdown("---")

    exp_dir = str(getattr(controller.config.experiment, "exp_dir", "experiments"))
    persisted_runs = _discover_persisted_runs(exp_dir)

    runs_with_data = [r for r in persisted_runs if r.get("completed_cycles", 0) > 0]
    if len(runs_with_data) < 2:
        st.info("At least 2 completed runs are needed for comparison. Run experiments with different strategies or models first.")
        return

    run_options = {r["key"]: r for r in runs_with_data}
    selected_keys = st.multiselect(
        "Select runs to compare",
        options=list(run_options.keys()),
        default=list(run_options.keys())[:min(4, len(run_options))],
        format_func=lambda k: run_options[k]["label"],
        key="comparison_run_selector",
    )

    if len(selected_keys) < 2:
        st.info("Select at least 2 runs to compare.")
        return

    selected_runs = [run_options[k] for k in selected_keys]

    # Build short labels for the legend, disambiguating when model/strategy collide
    def _base_label(run: Dict[str, Any]) -> str:
        model = run.get("model_name", "?")
        strategy = run.get("strategy", "?")
        return f"{model} / {strategy}"

    raw_labels = [_base_label(r) for r in selected_runs]
    dupes = {lbl for lbl, cnt in Counter(raw_labels).items() if cnt > 1}
    run_labels: Dict[str, str] = {}
    for run, raw in zip(selected_runs, raw_labels):
        if raw in dupes:
            run_labels[run["key"]] = f"{raw} ({run.get('modified_at', '?')})"
        else:
            run_labels[run["key"]] = raw

    # --- Test Accuracy Comparison ---
    st.markdown("### Test Accuracy")
    acc_rows = []
    for run in selected_runs:
        label = run_labels[run["key"]]
        for m in run.get("metrics_history", []):
            acc_rows.append({
                "Labeled Samples": m.get("labeled_pool_size", 0),
                "Test Accuracy": m.get("test_accuracy", 0) * 100,
                "Run": label,
            })
    if acc_rows:
        acc_df = pd.DataFrame(acc_rows).sort_values("Labeled Samples")
        st.line_chart(acc_df, x="Labeled Samples", y="Test Accuracy", color="Run", height=400)
    st.markdown("---")

    # --- F1 Score Comparison ---
    st.markdown("### F1 Score")
    f1_rows = []
    for run in selected_runs:
        label = run_labels[run["key"]]
        for m in run.get("metrics_history", []):
            f1_rows.append({
                "Labeled Samples": m.get("labeled_pool_size", 0),
                "F1 Score": m.get("test_f1", 0),
                "Run": label,
            })
    if f1_rows:
        f1_df = pd.DataFrame(f1_rows).sort_values("Labeled Samples")
        st.line_chart(f1_df, x="Labeled Samples", y="F1 Score", color="Run", height=400)
    st.markdown("---")

    # --- ECE Comparison ---
    st.markdown("### Calibration (ECE)")
    ece_rows = []
    for run in selected_runs:
        label = run_labels[run["key"]]
        for m in run.get("metrics_history", []):
            if m.get("ece") is not None:
                ece_rows.append({
                    "Labeled Samples": m.get("labeled_pool_size", 0),
                    "ECE": m.get("ece"),
                    "Run": label,
                })
    if ece_rows:
        ece_df = pd.DataFrame(ece_rows).sort_values("Labeled Samples")
        st.line_chart(ece_df, x="Labeled Samples", y="ECE", color="Run", height=300)
    else:
        st.info("No ECE data available for the selected runs.")
    st.markdown("---")

    # --- Final Results Summary Table ---
    st.markdown("### Final Results Summary")
    summary_rows = []
    for run in selected_runs:
        history = run.get("metrics_history", [])
        if not history:
            continue
        final = history[-1]
        summary_rows.append({
            "Run": run_labels[run["key"]],
            "Cycles": run.get("completed_cycles", 0),
            "Final Labeled": final.get("labeled_pool_size", 0),
            "Test Acc": f"{final.get('test_accuracy', 0) * 100:.2f}%",
            "F1": f"{final.get('test_f1', 0):.3f}",
            "ECE": f"{final.get('ece', 0):.4f}" if final.get("ece") is not None else "N/A",
            "ECE (cal.)": f"{final.get('ece_calibrated', 0):.4f}" if final.get("ece_calibrated") is not None else "N/A",
        })
    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width='stretch')


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
    render_query_summary(metrics_history, selected_snap, widget_prefix=widget_prefix)
    st.markdown("---")
    render_umap_evolution(metrics_history, selected_snap, widget_prefix=widget_prefix)
    st.markdown("---")
