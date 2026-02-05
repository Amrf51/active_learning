"""
Dataset Explorer Page - Inspect labeled and unlabeled data pools

This page allows users to:
- View images from the labeled pool with their assigned labels
- View images from the unlabeled pool
- See pool statistics (counts, class distribution)
- Filter by class name or filename
- Navigate large pools with pagination

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import logging
from typing import Optional, Dict, List, Any
from emoji_sanitizer import EmojiSanitizer

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# MVC imports
from controller.controller_factory import get_controller, update_session_heartbeat

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Dataset Explorer - AL Dashboard",
    layout="wide"
)
EmojiSanitizer(st).patch()

# Custom CSS
st.markdown("""
    <style>
    .explorer-section {
        background-color: #112240;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .pool-stats {
        background-color: #0f3d3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #20c997;
        color: #e6f1ff;
    }
    .image-card {
        background-color: #1a1a2e;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #4a4a6a;
    }
    .filter-section {
        background-color: #112240;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #233554;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
IMAGES_PER_PAGE = 12
COLS_PER_ROW = 4


def initialize_session_state():
    """Initialize session state for Dataset Explorer page."""
    update_session_heartbeat()
    
    if "explorer_pool" not in st.session_state:
        st.session_state.explorer_pool = "labeled"
    
    if "explorer_page" not in st.session_state:
        st.session_state.explorer_page = 0
    
    if "explorer_class_filter" not in st.session_state:
        st.session_state.explorer_class_filter = "All"
    
    if "explorer_filename_filter" not in st.session_state:
        st.session_state.explorer_filename_filter = ""


def check_experiment_active() -> bool:
    """Check if an experiment is active with initialized data manager."""
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        return status.get('experiment_id') is not None
    except Exception:
        return False


def get_pool_info_from_state() -> Optional[Dict[str, Any]]:
    """Get pool information from the current WorldState.
    
    Returns:
        Dictionary with pool info or None if not available
    """
    try:
        ctrl = get_controller()
        ctrl.poll_updates()  # Get latest state
        status = ctrl.get_status()
        
        if status.get('experiment_id') is None:
            return None
        
        return {
            'labeled': status.get('labeled_count', 0),
            'unlabeled': status.get('unlabeled_count', 0),
            'total': status.get('labeled_count', 0) + status.get('unlabeled_count', 0),
            'labeled_percentage': 0,
            'unlabeled_percentage': 0,
            'num_queries': status.get('current_cycle', 0),
            'experiment_id': status.get('experiment_id'),
            'experiment_name': status.get('experiment_name')
        }
    except Exception as e:
        logger.warning(f"Could not get pool info: {e}")
    return None


def get_experiment_config() -> Optional[Dict[str, Any]]:
    """Get experiment configuration from the database.
    
    Returns:
        Configuration dictionary or None if not available
    """
    try:
        ctrl = get_controller()
        status = ctrl.get_status()
        exp_id = status.get('experiment_id')
        
        if not exp_id:
            return None
        
        # Access experiment manager to get config
        from model.experiment_manager import ExperimentManager
        exp_manager = ExperimentManager(Path("./experiments"))
        exp_data = exp_manager.load_experiment(exp_id)
        
        return exp_data.get('config', {})
    except Exception as e:
        logger.warning(f"Could not get experiment config: {e}")
    return None


def get_dataset_images(data_dir: str, pool: str = "all") -> List[Dict[str, Any]]:
    """Scan dataset directory to get image information.
    
    This is a fallback method that scans the filesystem directly
    when the data manager is not accessible (multiprocessing boundary).
    
    Args:
        data_dir: Path to the dataset directory
        pool: "all" to get all images (we can't distinguish pools without data manager)
        
    Returns:
        List of image info dictionaries
    """
    images = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        return images
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # Scan class folders
    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        
        class_name = class_dir.name
        
        for img_file in sorted(class_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    'path': str(img_file),
                    'filename': img_file.name,
                    'class_name': class_name,
                    'class_dir': str(class_dir)
                })
    
    return images


def display_pool_info_banner(pool_info: Dict[str, Any]):
    """Display a banner showing current pool status.
    
    Requirements: 11.1, 11.2
    """
    labeled = pool_info.get('labeled', 0)
    unlabeled = pool_info.get('unlabeled', 0)
    total = labeled + unlabeled
    
    if total > 0:
        st.info(
            f"📊 **Pool Status:** {labeled:,} labeled ({labeled/total*100:.1f}%) | "
            f"{unlabeled:,} unlabeled ({unlabeled/total*100:.1f}%)"
        )



def display_pool_statistics(pool_info: Dict[str, Any], class_names: List[str], images: List[Dict[str, Any]]):
    """Display pool statistics (counts, class distribution).
    
    Requirements: 11.3
    """
    st.markdown('<div class="pool-stats">', unsafe_allow_html=True)
    st.subheader("📊 Pool Statistics")
    
    total = pool_info.get('total', len(images))
    labeled = pool_info.get('labeled', 0)
    unlabeled = pool_info.get('unlabeled', 0)
    
    # Calculate percentages
    labeled_pct = (labeled / total * 100) if total > 0 else 0
    unlabeled_pct = (unlabeled / total * 100) if total > 0 else 0
    
    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{total:,}")
    
    with col2:
        st.metric(
            "Labeled Pool",
            f"{labeled:,}",
            delta=f"{labeled_pct:.1f}%"
        )
    
    with col3:
        st.metric(
            "Unlabeled Pool",
            f"{unlabeled:,}",
            delta=f"{unlabeled_pct:.1f}%"
        )
    
    with col4:
        st.metric("Cycles Completed", f"{pool_info.get('num_queries', 0)}")
    
    # Class distribution from scanned images
    st.markdown("---")
    st.markdown("#### 📈 Dataset Class Distribution")
    
    if images:
        # Count images per class
        class_counts = {}
        for img in images:
            class_name = img.get('class_name', 'Unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Create DataFrame for chart
        dist_data = [{"Class": k, "Count": v} for k, v in sorted(class_counts.items())]
        dist_df = pd.DataFrame(dist_data)
        
        if len(dist_df) <= 20:
            st.bar_chart(dist_df.set_index("Class"))
        else:
            # For many classes, show top/bottom
            with st.expander("View Class Distribution"):
                sorted_df = dist_df.sort_values("Count", ascending=False)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top 10 Classes:**")
                    st.dataframe(sorted_df.head(10), hide_index=True)
                with col2:
                    st.markdown("**Bottom 10 Classes:**")
                    st.dataframe(sorted_df.tail(10), hide_index=True)
    else:
        st.info("No images found in dataset.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def display_filter_controls(class_names: List[str]):
    """Display filtering controls.
    
    Requirements: 11.5
    """
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        # Class filter dropdown
        class_options = ["All"] + sorted(class_names)
        current_filter = st.session_state.explorer_class_filter
        if current_filter not in class_options:
            current_filter = "All"
        
        selected_class = st.selectbox(
            "Filter by Class",
            options=class_options,
            index=class_options.index(current_filter) if current_filter in class_options else 0,
            help="Filter images by class name"
        )
        
        if selected_class != st.session_state.explorer_class_filter:
            st.session_state.explorer_class_filter = selected_class
            st.session_state.explorer_page = 0
            st.rerun()
    
    with col2:
        # Filename search
        filename_filter = st.text_input(
            "Search by Filename",
            value=st.session_state.explorer_filename_filter,
            placeholder="Enter filename pattern...",
            help="Filter images by filename (partial match)"
        )
        
        if filename_filter != st.session_state.explorer_filename_filter:
            st.session_state.explorer_filename_filter = filename_filter
            st.session_state.explorer_page = 0
            st.rerun()
    
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Clear Filters", use_container_width=True):
            st.session_state.explorer_class_filter = "All"
            st.session_state.explorer_filename_filter = ""
            st.session_state.explorer_page = 0
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


def get_filtered_images(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get filtered images based on current filters.
    
    Returns:
        List of image dictionaries matching the filters
    """
    class_filter = st.session_state.explorer_class_filter
    filename_filter = st.session_state.explorer_filename_filter.lower()
    
    filtered = images
    
    # Apply class filter
    if class_filter != "All":
        filtered = [img for img in filtered if img.get('class_name') == class_filter]
    
    # Apply filename filter
    if filename_filter:
        filtered = [img for img in filtered if filename_filter in img.get('filename', '').lower()]
    
    return filtered


def display_pagination(total_items: int):
    """Display pagination controls.
    
    Requirements: 11.4
    """
    total_pages = max(1, (total_items + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
    current_page = st.session_state.explorer_page
    
    # Ensure current page is valid
    if current_page >= total_pages:
        st.session_state.explorer_page = max(0, total_pages - 1)
        current_page = st.session_state.explorer_page
    
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⏮️ First", disabled=current_page == 0, use_container_width=True):
            st.session_state.explorer_page = 0
            st.rerun()
    
    with col2:
        if st.button("◀️ Prev", disabled=current_page == 0, use_container_width=True):
            st.session_state.explorer_page = max(0, current_page - 1)
            st.rerun()
    
    with col3:
        st.markdown(
            f"<div style='text-align: center; padding: 0.5rem;'>"
            f"Page {current_page + 1} of {total_pages} ({total_items:,} items)"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col4:
        if st.button("Next ▶️", disabled=current_page >= total_pages - 1, use_container_width=True):
            st.session_state.explorer_page = min(total_pages - 1, current_page + 1)
            st.rerun()
    
    with col5:
        if st.button("Last ⏭️", disabled=current_page >= total_pages - 1, use_container_width=True):
            st.session_state.explorer_page = total_pages - 1
            st.rerun()



def display_image_grid(images: List[Dict[str, Any]]):
    """Display images in a grid with lazy loading.
    
    Requirements: 11.1, 11.2, 11.4
    """
    current_page = st.session_state.explorer_page
    
    # Calculate page slice
    start_idx = current_page * IMAGES_PER_PAGE
    end_idx = min(start_idx + IMAGES_PER_PAGE, len(images))
    page_images = images[start_idx:end_idx]
    
    if not page_images:
        st.info("No images found with current filters.")
        return
    
    st.markdown("### 🖼️ Dataset Images")
    
    # Display images in grid
    for row_start in range(0, len(page_images), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        
        for col_idx in range(COLS_PER_ROW):
            img_idx = row_start + col_idx
            if img_idx >= len(page_images):
                break
            
            img_info = page_images[img_idx]
            
            with cols[col_idx]:
                display_image_card(img_info)


def display_image_card(img_info: Dict[str, Any]):
    """Display a single image card with metadata.
    
    Args:
        img_info: Dictionary with image information
    """
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    
    image_path = img_info.get('path', '')
    class_name = img_info.get('class_name', 'Unknown')
    filename = img_info.get('filename', 'unknown')
    
    # Display image (lazy loading - only load when visible)
    if image_path and Path(image_path).exists():
        try:
            st.image(image_path, use_container_width=True)
        except Exception as e:
            st.warning("Could not load image")
            logger.warning(f"Failed to load image {image_path}: {e}")
    else:
        st.markdown(
            f"<div style='height: 150px; background: #333; display: flex; "
            f"align-items: center; justify-content: center; border-radius: 4px;'>"
            f"<span style='color: #888;'>Image not found</span></div>",
            unsafe_allow_html=True
        )
    
    # Display metadata
    display_name = f"{filename[:20]}..." if len(filename) > 20 else filename
    st.caption(f"📁 {display_name}")
    st.markdown(f"**Class:** {class_name}")
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main Dataset Explorer page."""
    initialize_session_state()
    
    st.title("🔍 Dataset Explorer")
    st.markdown("Explore the dataset used in your active learning experiment.")
    
    # Check if experiment is active
    if not check_experiment_active():
        st.warning("⚠️ No active experiment. Please create or load an experiment first.")
        st.info("Go to the **Configuration** page to create a new experiment.")
        return
    
    # Get pool info from WorldState
    pool_info = get_pool_info_from_state()
    
    if pool_info is None:
        st.warning("⚠️ Could not retrieve experiment information.")
        return
    
    # Get experiment config to find data directory
    config = get_experiment_config()
    
    if config is None:
        st.warning("⚠️ Could not retrieve experiment configuration.")
        st.info("The experiment may not be fully initialized yet.")
        return
    
    data_dir = config.get('data_dir', './data/raw')
    
    # Display experiment info
    st.markdown(f"**Experiment:** {pool_info.get('experiment_name', 'Unknown')}")
    st.markdown(f"**Dataset:** `{data_dir}`")
    
    st.markdown("---")
    
    # Scan dataset to get images
    images = get_dataset_images(data_dir)
    
    if not images:
        st.warning(f"⚠️ No images found in dataset directory: `{data_dir}`")
        st.info("Make sure the dataset path is correct and contains image files.")
        return
    
    # Get class names from scanned images
    class_names = sorted(list(set(img.get('class_name', 'Unknown') for img in images)))
    
    # Display pool statistics
    display_pool_statistics(pool_info, class_names, images)
    
    st.markdown("---")
    
    # Display filter controls
    display_filter_controls(class_names)
    
    st.markdown("---")
    
    # Get filtered images
    filtered_images = get_filtered_images(images)
    
    # Display pagination
    display_pagination(len(filtered_images))
    
    st.markdown("---")
    
    # Display image grid
    display_image_grid(filtered_images)
    
    # Bottom pagination
    if len(filtered_images) > IMAGES_PER_PAGE:
        st.markdown("---")
        display_pagination(len(filtered_images))


if __name__ == "__main__":
    main()
