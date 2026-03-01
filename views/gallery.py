"""
Gallery view for manual annotation during ANNOTATING state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from controller import Controller
from events import Event, EventType

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _load_display_image(image_path: str) -> Image.Image:
    """Open image for Streamlit display and normalize problematic palette mode."""
    img = Image.open(image_path)
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA")
    return img


def render_image_grid(
    queried_images: List[Dict[str, Any]],
    available_classes: List[str],
    num_columns: int = 4,
) -> None:
    if not queried_images:
        st.warning("No images to display")
        return

    num_images = len(queried_images)
    num_rows = (num_images + num_columns - 1) // num_columns

    for row_idx in range(num_rows):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            img_idx = row_idx * num_columns + col_idx
            if img_idx < num_images:
                with cols[col_idx]:
                    render_image_card(queried_images[img_idx], img_idx, available_classes)


def render_image_card(image_data: Dict[str, Any], img_idx: int, available_classes: List[str]) -> None:
    image_id = image_data.get("image_id", img_idx)
    image_path = image_data.get("image_path", "")
    display_path = image_data.get("display_path", image_path)
    predicted_class = image_data.get("predicted_class", "Unknown")
    predicted_confidence = image_data.get("predicted_confidence", 0.0)
    uncertainty_score = image_data.get("uncertainty_score", 0.0)
    ground_truth_name = image_data.get("ground_truth_name", "Unknown")

    with st.container():
        try:
            if Path(display_path).exists():
                img = _load_display_image(display_path)
                st.image(img, width="stretch")
            else:
                st.warning(f"Image not found: {display_path}")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Error loading image: {exc}")
            logger.error("Failed to load image %s: %s", display_path, exc)

        uncertainty_color = get_uncertainty_color(uncertainty_score)
        st.markdown(
            f"<div style='background-color: {uncertainty_color}; padding: 5px; "
            f"border-radius: 5px; text-align: center; margin-bottom: 5px;'>"
            f"<b>{get_uncertainty_label(uncertainty_score)} Uncertainty</b></div>",
            unsafe_allow_html=True,
        )

        st.caption(f"Prediction: {predicted_class}")
        st.caption(f"Confidence: {predicted_confidence * 100:.1f}%")
        st.caption(f"Uncertainty: {uncertainty_score:.3f}")

        is_auto = (
            "config" in st.session_state
            and st.session_state.config.active_learning.auto_annotate
        )
        if is_auto:
            with st.expander("Ground Truth"):
                st.write(f"True Label: {ground_truth_name}")

        annotation_key = f"annotation_{img_idx}"
        default_idx = available_classes.index(ground_truth_name) if ground_truth_name in available_classes else 0
        selected_label = st.selectbox(
            "Select Label",
            options=available_classes,
            index=default_idx if is_auto else 0,
            key=annotation_key,
            label_visibility="collapsed",
        )

        use_ground_truth = False
        if is_auto:
            use_gt_key = f"use_gt_{img_idx}"
            use_ground_truth = st.checkbox(
                "Use GT",
                value=True,
                key=use_gt_key,
                help="Use ground truth label for this image",
            )

        if "annotations" not in st.session_state:
            st.session_state.annotations = {}

        ground_truth_val = image_data.get("ground_truth", 0)
        if use_ground_truth:
            st.session_state.annotations[image_id] = {
                "image_id": image_id,
                "user_label": ground_truth_val,
                "ground_truth": ground_truth_val,
                "label_name": ground_truth_name,
            }
        else:
            label_idx = available_classes.index(selected_label) if selected_label in available_classes else 0
            st.session_state.annotations[image_id] = {
                "image_id": image_id,
                "user_label": label_idx,
                "ground_truth": ground_truth_val,
                "label_name": selected_label,
            }


def get_uncertainty_color(uncertainty_score: float) -> str:
    if uncertainty_score > 0.7:
        return "#ff4444"
    if uncertainty_score > 0.4:
        return "#ff9944"
    return "#ffdd44"


def get_uncertainty_label(uncertainty_score: float) -> str:
    if uncertainty_score > 0.7:
        return "High"
    if uncertainty_score > 0.4:
        return "Medium"
    return "Low"


def get_available_classes(snap: Dict[str, Any]) -> List[str]:
    if snap.get("class_names"):
        return list(snap["class_names"])
    if snap.get("queried_images"):
        names = sorted(
            {
                img.get("ground_truth_name", "")
                for img in snap["queried_images"]
                if img.get("ground_truth_name")
            }
        )
        if names:
            return names
    return [f"Class_{i}" for i in range(10)]


def render_auto_label_button(queried_images: List[Dict[str, Any]]) -> bool:
    if st.button(
        "Auto-Label All (Ground Truth)",
        type="secondary",
        width="stretch",
        help="Automatically label all images with their ground truth labels",
    ):
        if "annotations" not in st.session_state:
            st.session_state.annotations = {}
        for img_data in queried_images:
            image_id = img_data.get("image_id")
            ground_truth = img_data.get("ground_truth", 0)
            ground_truth_name = img_data.get("ground_truth_name", "Unknown")
            st.session_state.annotations[image_id] = {
                "image_id": image_id,
                "user_label": ground_truth,
                "ground_truth": ground_truth,
                "label_name": ground_truth_name,
            }
        st.success(f"Auto-labeled {len(queried_images)} images with ground truth")
        logger.info("Auto-labeled %d images", len(queried_images))
        return True
    return False


def render_submit_button(
    controller: Controller,
    queried_images: List[Dict[str, Any]],
    snap: Dict[str, Any],
) -> bool:
    annotations = st.session_state.get("annotations", {})
    current_image_ids = {img.get("image_id") for img in queried_images}
    relevant = {k: v for k, v in annotations.items() if k in current_image_ids}
    st.session_state.annotations = relevant

    num_annotated = len(relevant)
    num_required = len(queried_images)
    progress = min(1.0, num_annotated / num_required) if num_required > 0 else 0.0

    st.progress(
        progress,
        text=f"Annotated: {num_annotated} / {num_required}",
    )

    submit_disabled = num_annotated < num_required
    if st.button(
        "Submit Annotations",
        type="primary",
        width="stretch",
        disabled=submit_disabled,
        help="Submit annotations and continue"
        if not submit_disabled
        else f"Please annotate all {num_required} images before submitting",
    ):
        try:
            annotations_list = [
                {"image_id": ann["image_id"], "user_label": ann["user_label"]}
                for ann in relevant.values()
            ]
            accepted = controller.dispatch(
                Event(
                    type=EventType.SUBMIT_ANNOTATIONS,
                    run_id=snap["run_id"],
                    cycle=snap["current_cycle"],
                    data={
                        "annotations": annotations_list,
                        "query_token": snap.get("query_token", ""),
                    },
                )
            )
            if not accepted:
                st.warning("Annotations rejected: run or cycle is stale. Refreshing view.")
                return False

            st.session_state.last_annotation_feedback = {
                "num_submitted": len(annotations_list),
                "annotations": relevant,
            }
            st.session_state.annotations = {}
            logger.info("Submitted %d annotations", len(annotations_list))
            return True
        except Exception as exc:  # pylint: disable=broad-exception-caught
            st.error(f"Failed to submit annotations: {exc}")
            logger.error("Failed to submit annotations: %s", exc)
            return False
    return False


def render_annotation_feedback() -> None:
    if "last_annotation_feedback" not in st.session_state:
        return

    feedback = st.session_state.last_annotation_feedback
    annotations = feedback.get("annotations", {})
    if not annotations:
        return

    num_total = len(annotations)
    num_correct = sum(
        1 for ann in annotations.values() if ann["user_label"] == ann.get("ground_truth", -1)
    )
    accuracy = (num_correct / num_total * 100.0) if num_total > 0 else 0.0
    st.success(
        f"Annotations Submitted: {num_total} images\n\n"
        f"Accuracy: {num_correct}/{num_total} correct ({accuracy:.1f}%)"
    )

    with st.expander("Annotation Details"):
        for ann in annotations.values():
            label_name = ann.get("label_name", "Unknown")
            is_correct = ann["user_label"] == ann.get("ground_truth", -1)
            icon = "OK" if is_correct else "X"
            st.write(f"{icon} Image {ann['image_id']}: {label_name}")


def render_gallery_view(controller: Controller, snap: Dict[str, Any]) -> None:
    st.title("Gallery of Uncertainty")
    st.markdown(f"Cycle {snap['current_cycle']} - Please label the selected samples")
    st.markdown("---")

    cycle_id = (snap["run_id"], snap["current_cycle"])
    if st.session_state.get("gallery_cycle_id") != cycle_id:
        st.session_state["gallery_cycle_id"] = cycle_id
        st.session_state.annotations = {}
        st.session_state.pop("last_annotation_feedback", None)

    queried_images = snap["queried_images"]
    if not queried_images:
        st.warning("No images available for annotation")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Samples to Annotate", len(queried_images))
    with col2:
        config_overrides = st.session_state.get("config_overrides", {})
        strategy = config_overrides.get("active_learning.sampling_strategy", "Unknown")
        st.metric("Strategy", str(strategy).capitalize())
    with col3:
        avg_uncertainty = sum(img.get("uncertainty_score", 0) for img in queried_images) / len(queried_images)
        st.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}")

    st.markdown("---")
    render_annotation_feedback()

    st.markdown("### Selected Images")
    available_classes = get_available_classes(snap)
    render_image_grid(queried_images, available_classes, num_columns=4)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if render_auto_label_button(queried_images):
            st.rerun()
    with col2:
        if render_submit_button(controller, queried_images, snap):
            st.success("Annotations submitted. Training will continue automatically.")
            st.rerun()

    st.markdown("---")
    st.info(
        "Instructions:\n"
        "1. Review each image and predicted label.\n"
        "2. Select corrected labels where needed.\n"
        "3. Optional: Auto-label all with ground truth.\n"
        "4. Submit annotations to continue."
    )
