"""FDAFT - Fast Double-Channel Aggregated Feature Transform"""

__version__ = "1.0.0"
__author__ = "FDAFT Team"

from .models.fdaft import FDAFT
from .utils.visualization import FDAFTVisualizer

__all__ = ['FDAFT', 'FDAFTVisualizer']