"""
Dataset explorer view for labeled/unlabeled pool inspection.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from core.controller import Controller


def _resolve_pool_sizes(snap: Dict[str, Any]) -> tuple[int, int]:
    labeled = int(snap.get("labeled_pool_size", 0))
    unlabeled = int(snap.get("unlabeled_pool_size", 0))
    if (labeled + unlabeled) == 0 and snap.get("metrics_history"):
        latest = snap["metrics_history"][-1]
        labeled = int(latest.get("labeled_pool_size", labeled))
        unlabeled = int(latest.get("unlabeled_pool_size", unlabeled))
    return labeled, unlabeled


def _distribution_dataframe(
    class_names: List[str],
    labeled_distribution: Dict[str, int],
    unlabeled_distribution: Dict[str, int],
) -> pd.DataFrame:
    classes = class_names or sorted(set(labeled_distribution) | set(unlabeled_distribution))
    rows = []
    for class_name in classes:
        labeled_count = int(labeled_distribution.get(class_name, 0))
        unlabeled_count = int(unlabeled_distribution.get(class_name, 0))
        rows.append(
            {
                "Class": class_name,
                "Labeled": labeled_count,
                "Unlabeled": unlabeled_count,
                "Total": labeled_count + unlabeled_count,
            }
        )
    return pd.DataFrame(rows)


def render_explorer_view(controller: Controller, snap: Dict[str, Any]) -> None:
    """Render pool sizes and class distribution for dataset exploration."""
    _ = controller  # Reserved for future explorer actions

    st.subheader("Dataset Explorer")

    labeled, unlabeled = _resolve_pool_sizes(snap)
    total = labeled + unlabeled

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Labeled Pool", labeled)
    with c2:
        st.metric("Unlabeled Pool", unlabeled)
    with c3:
        st.metric("Total Samples", total)

    if total > 0:
        ratio = labeled / total
        st.progress(ratio, text=f"Labeled share: {ratio * 100:.1f}%")
    else:
        st.info("Pool statistics will appear once the experiment initializes.")

    class_names = list(snap.get("class_names", []))
    labeled_distribution = dict(snap.get("labeled_class_distribution", {}))
    unlabeled_distribution = dict(snap.get("unlabeled_class_distribution", {}))
    if not labeled_distribution and not unlabeled_distribution:
        st.info("Class distribution is not available yet.")
        return

    st.markdown("### Class Distribution")
    dist_df = _distribution_dataframe(class_names, labeled_distribution, unlabeled_distribution)
    if dist_df.empty:
        st.info("No class distribution data to show.")
        return

    st.bar_chart(dist_df.set_index("Class")[["Labeled", "Unlabeled"]], height=350)
    st.dataframe(dist_df, width="stretch", hide_index=True)
