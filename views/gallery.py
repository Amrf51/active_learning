"""
Gallery of Uncertainty view for Active Learning Framework.

This module provides the annotation interface during the ANNOTATING state:
- Grid display of queried images (3-4 columns)
- Per-image card with thumbnail, prediction, confidence, uncertainty score
- Auto-label button for batch simulation
- Submit annotations button
- Annotation feedback after submission

The gallery displays images selected by the sampling strategy and allows
users to provide labels (or auto-label with ground truth).
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from PIL import Image
from controller import Controller

logger = logging.getLogger(__name__)


# ============================================================================
# SUBTASK 13.1: Grid display of queried images (3-4 columns)
# ============================================================================

def render_image_grid(queried_images: List[Dict[str, Any]], num_columns: int = 4) -> None:
    """
    Render grid display of queried images.
    
    Args:
        queried_images: List of queried image dictionaries from controller
        num_columns: Number of columns in the grid (default: 4)
    """
    if not queried_images:
        st.warning("⚠️ No images to display")
        return
    
    # Calculate number of rows needed
    num_images = len(queried_images)
    num_rows = (num_images + num_columns - 1) // num_columns
    
    # Render grid row by row
    for row_idx in range(num_rows):
        cols = st.columns(num_columns)
        
        for col_idx in range(num_columns):
            img_idx = row_idx * num_columns + col_idx
            
            # Check if we have an image for this position
            if img_idx < num_images:
                with cols[col_idx]:
                    render_image_card(queried_images[img_idx], img_idx)


# ============================================================================
# SUBTASK 13.2: Per-image card: image thumbnail, prediction, confidence %, 
#                uncertainty score
# ============================================================================

def render_image_card(image_data: Dict[str, Any], img_idx: int) -> None:
    """
    Render individual image card with all information.
    
    Args:
        image_data: Dictionary with image information from QueriedImage
        img_idx: Index of the image in the queried list
    """
    # Extract image information
    image_id = image_data.get('image_id', img_idx)
    image_path = image_data.get('image_path', '')
    display_path = image_data.get('display_path', image_path)
    predicted_class = image_data.get('predicted_class', 'Unknown')
    predicted_confidence = image_data.get('predicted_confidence', 0.0)
    uncertainty_score = image_data.get('uncertainty_score', 0.0)
    ground_truth_name = image_data.get('ground_truth_name', 'Unknown')
    
    # Create a container for the card
    with st.container():
        # Display image thumbnail
        try:
            if Path(display_path).exists():
                img = Image.open(display_path)
                st.image(img, use_container_width=True)
            else:
                st.warning(f"Image not found: {display_path}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
            logger.error(f"Failed to load image {display_path}: {e}")
        
        # Uncertainty indicator with color coding
        uncertainty_color = get_uncertainty_color(uncertainty_score)
        st.markdown(
            f"<div style='background-color: {uncertainty_color}; padding: 5px; "
            f"border-radius: 5px; text-align: center; margin-bottom: 5px;'>"
            f"<b>{get_uncertainty_label(uncertainty_score)} Uncertainty</b></div>",
            unsafe_allow_html=True
        )
        
        # Prediction information
        st.caption(f"**Prediction:** {predicted_class}")
        st.caption(f"**Confidence:** {predicted_confidence * 100:.1f}%")
        st.caption(f"**Uncertainty:** {uncertainty_score:.3f}")
        
        # Determine if we're in auto-annotate mode
        is_auto = (
            'config' in st.session_state
            and st.session_state.config.active_learning.auto_annotate
        )

        # Ground truth (only visible in auto-annotate / simulation mode)
        if is_auto:
            with st.expander("Ground Truth"):
                st.write(f"**True Label:** {ground_truth_name}")

        # Store annotation selection in session state
        annotation_key = f"annotation_{img_idx}"

        # Get available classes
        available_classes = get_available_classes()

        # Annotation dropdown
        default_idx = (
            available_classes.index(ground_truth_name)
            if ground_truth_name in available_classes else 0
        )
        selected_label = st.selectbox(
            "Select Label",
            options=available_classes,
            index=default_idx if is_auto else 0,
            key=annotation_key,
            label_visibility="collapsed"
        )

        # "Use GT" checkbox only in auto-annotate mode
        use_ground_truth = False
        if is_auto:
            use_gt_key = f"use_gt_{img_idx}"
            use_ground_truth = st.checkbox(
                "Use GT",
                value=True,
                key=use_gt_key,
                help="Use ground truth label for this image"
            )

        # Store the annotation in session state
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}

        ground_truth_val = image_data.get('ground_truth', 0)

        if use_ground_truth:
            st.session_state.annotations[image_id] = {
                'image_id': image_id,
                'user_label': ground_truth_val,
                'ground_truth': ground_truth_val,
                'label_name': ground_truth_name,
            }
        else:
            label_idx = available_classes.index(selected_label) if selected_label in available_classes else 0
            st.session_state.annotations[image_id] = {
                'image_id': image_id,
                'user_label': label_idx,
                'ground_truth': ground_truth_val,
                'label_name': selected_label,
            }


def get_uncertainty_color(uncertainty_score: float) -> str:
    """
    Get color code based on uncertainty score.
    
    Args:
        uncertainty_score: Uncertainty score (0.0 to 1.0+)
        
    Returns:
        Hex color code
    """
    # High uncertainty (> 0.7) -> Red
    if uncertainty_score > 0.7:
        return "#ff4444"
    # Medium uncertainty (0.4 - 0.7) -> Orange
    elif uncertainty_score > 0.4:
        return "#ff9944"
    # Low uncertainty (< 0.4) -> Yellow
    else:
        return "#ffdd44"


def get_uncertainty_label(uncertainty_score: float) -> str:
    """
    Get text label based on uncertainty score.
    
    Args:
        uncertainty_score: Uncertainty score (0.0 to 1.0+)
        
    Returns:
        Text label (High, Medium, Low)
    """
    if uncertainty_score > 0.7:
        return "🔴 High"
    elif uncertainty_score > 0.4:
        return "🟠 Medium"
    else:
        return "🟡 Low"


def get_available_classes() -> List[str]:
    """
    Get list of available class names from controller or queried images.

    Returns:
        List of class names
    """
    # Try to get from controller (populated by worker's cycle_prepared message)
    controller = st.session_state.get('controller')
    if controller and getattr(controller, 'class_names', None):
        return controller.class_names

    # Fallback: extract unique names from queried images
    if controller and controller.queried_images:
        names = sorted(set(
            img.get('ground_truth_name', '')
            for img in controller.queried_images
            if img.get('ground_truth_name')
        ))
        if names:
            return names

    # Last resort fallback
    return [f"Class_{i}" for i in range(10)]


# ============================================================================
# SUBTASK 13.3: "Auto-label All (Ground Truth)" button for batch simulation
# ============================================================================

def render_auto_label_button(queried_images: List[Dict[str, Any]]) -> bool:
    """
    Render auto-label button for batch simulation.
    
    This button automatically fills all annotations with ground truth labels,
    useful for simulating the annotation process without manual labeling.
    
    Args:
        queried_images: List of queried image dictionaries
        
    Returns:
        True if auto-label was clicked, False otherwise
    """
    if st.button(
        "🏷️ Auto-Label All (Ground Truth)",
        type="secondary",
        use_container_width=True,
        help="Automatically label all images with their ground truth labels"
    ):
        # Initialize annotations dict if not exists
        if 'annotations' not in st.session_state:
            st.session_state.annotations = {}
        
        # Fill all annotations with ground truth
        for img_data in queried_images:
            image_id = img_data.get('image_id')
            ground_truth = img_data.get('ground_truth', 0)
            ground_truth_name = img_data.get('ground_truth_name', 'Unknown')

            st.session_state.annotations[image_id] = {
                'image_id': image_id,
                'user_label': ground_truth,
                'ground_truth': ground_truth,
                'label_name': ground_truth_name,
            }
        
        st.success(f"✅ Auto-labeled {len(queried_images)} images with ground truth")
        logger.info(f"Auto-labeled {len(queried_images)} images")
        return True
    
    return False


# ============================================================================
# SUBTASK 13.4: Submit annotations button (dispatches ANNOTATE, triggers 
#                next cycle)
# ============================================================================

def render_submit_button(controller: Controller, queried_images: List[Dict[str, Any]]) -> bool:
    """
    Render submit annotations button.
    
    This button collects all annotations and dispatches them to the worker,
    which will trigger the next training cycle.
    
    Args:
        controller: Controller instance
        queried_images: List of queried image dictionaries
        
    Returns:
        True if submit was successful, False otherwise
    """
    # Check if we have annotations for all images
    annotations = st.session_state.get('annotations', {})
    num_annotated = len(annotations)
    num_required = len(queried_images)
    
    # Show progress
    st.progress(
        num_annotated / num_required if num_required > 0 else 0,
        text=f"Annotated: {num_annotated} / {num_required}"
    )
    
    # Submit button (disabled if not all images are annotated)
    submit_disabled = num_annotated < num_required
    
    if st.button(
        "✅ Submit Annotations",
        type="primary",
        use_container_width=True,
        disabled=submit_disabled,
        help="Submit annotations and start next training cycle" if not submit_disabled 
             else f"Please annotate all {num_required} images before submitting"
    ):
        try:
            # Prepare annotations list for worker
            annotations_list = [
                {'image_id': ann['image_id'], 'user_label': ann['user_label']}
                for ann in annotations.values()
            ]
            
            # Dispatch annotations to worker
            controller.dispatch_annotate(annotations_list)
            
            # Store annotation feedback for display
            st.session_state.last_annotation_feedback = {
                'num_submitted': len(annotations_list),
                'annotations': annotations
            }
            
            # Clear annotations for next cycle
            st.session_state.annotations = {}
            
            logger.info(f"Submitted {len(annotations_list)} annotations")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to submit annotations: {e}")
            logger.error(f"Failed to submit annotations: {e}")
            return False
    
    return False


# ============================================================================
# SUBTASK 13.5: Annotation feedback after submission (X/Y correct)
# ============================================================================

def render_annotation_feedback() -> None:
    """
    Render annotation feedback after submission.
    
    Shows how many annotations were correct compared to ground truth.
    This provides immediate feedback on annotation quality.
    """
    if 'last_annotation_feedback' not in st.session_state:
        return
    
    feedback = st.session_state.last_annotation_feedback
    annotations = feedback.get('annotations', {})
    
    if not annotations:
        return
    
    # Calculate accuracy
    num_total = len(annotations)
    num_correct = sum(
        1 for ann in annotations.values()
        if ann['user_label'] == ann.get('ground_truth', -1)
    )
    
    accuracy = (num_correct / num_total * 100) if num_total > 0 else 0
    
    # Display feedback
    st.success(
        f"✅ **Annotations Submitted:** {num_total} images\n\n"
        f"**Accuracy:** {num_correct}/{num_total} correct ({accuracy:.1f}%)"
    )
    
    # Show details in expander
    with st.expander("📊 Annotation Details"):
        for ann in annotations.values():
            label_name = ann.get('label_name', 'Unknown')
            is_correct = ann['user_label'] == ann.get('ground_truth', -1)
            icon = "✅" if is_correct else "❌"
            st.write(f"{icon} Image {ann['image_id']}: {label_name}")


# ============================================================================
# MAIN GALLERY VIEW RENDER FUNCTION
# ============================================================================

def render_gallery_view(controller: Controller) -> None:
    """
    Main render function for the Gallery of Uncertainty.
    
    This view is shown during the ANNOTATING state when images have been
    queried and are ready for annotation.
    
    Args:
        controller: Controller instance
    """
    # Header
    st.title("🔍 Gallery of Uncertainty")
    st.markdown(f"**Cycle {controller.current_cycle}** — Please label the selected samples")
    st.markdown("---")
    
    # Get queried images from controller
    queried_images = controller.queried_images
    
    if not queried_images:
        st.warning("⚠️ No images available for annotation")
        return
    
    # Show strategy and query info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Samples to Annotate", len(queried_images))
    
    with col2:
        # Get strategy from config
        config_overrides = st.session_state.get('config_overrides', {})
        strategy = config_overrides.get('active_learning.sampling_strategy', 'Unknown')
        st.metric("Strategy", strategy.capitalize())
    
    with col3:
        # Show average uncertainty
        avg_uncertainty = sum(
            img.get('uncertainty_score', 0) for img in queried_images
        ) / len(queried_images) if queried_images else 0
        st.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}")
    
    st.markdown("---")
    
    # Show annotation feedback if available (Subtask 13.5)
    render_annotation_feedback()
    
    # Render image grid (Subtask 13.1 & 13.2)
    st.markdown("### 📸 Selected Images")
    render_image_grid(queried_images, num_columns=4)
    
    st.markdown("---")
    
    # Render control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Auto-label button (Subtask 13.3)
        auto_labeled = render_auto_label_button(queried_images)
        if auto_labeled:
            st.rerun()
    
    with col2:
        # Submit button (Subtask 13.4)
        submitted = render_submit_button(controller, queried_images)
        if submitted:
            # Controller will automatically dispatch next cycle after annotation completes
            st.success("✓ Annotations submitted. Controller will proceed to next cycle automatically.")
            st.rerun()
    
    # Instructions
    st.markdown("---")
    st.info(
        "💡 **Instructions:**\n"
        "1. Review each image and its predicted label\n"
        "2. Use the dropdown to correct labels if needed, or check '✓ Use GT' to use ground truth\n"
        "3. Click 'Auto-Label All' to quickly label all images with ground truth\n"
        "4. Click 'Submit Annotations' when all images are labeled to continue to the next cycle"
    )
