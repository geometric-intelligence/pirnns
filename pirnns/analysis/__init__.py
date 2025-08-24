"""
Analysis module for PIRNNS.

Provides tools for analyzing trained RNN models including:
- Spatial analysis (rate maps, place cells, grid cells)
- Dynamics analysis (PCA)
- Performance analysis (loss curves)
"""

from .spatial import SpatialAnalyzer
from .dynamics import DynamicsAnalyzer
from .performance import plot_loss_curves

__all__ = [
    "SpatialAnalyzer",
    "DynamicsAnalyzer",
    "plot_loss_curves",
]
