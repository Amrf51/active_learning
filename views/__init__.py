"""
Views package for the Active Learning Framework Streamlit UI.

This package contains all UI components organized by functionality:
- router.py: Main view dispatcher based on application state
- sidebar.py: Configuration controls and experiment settings
- training.py: Live training visualization with progress and metrics
- gallery.py: Gallery of Uncertainty for annotation
- results.py: Results dashboard with metrics and charts
- comparison.py: Strategy comparison view (deferred to Phase 7)
- explorer.py: Dataset explorer (deferred to Phase 7)

The views follow the MVC pattern where:
- Model: Worker process (AL loop, trainer, data manager)
- View: Streamlit UI components (this package)
- Controller: State machine and command dispatcher
"""

from views.router import render

__all__ = ['render']
