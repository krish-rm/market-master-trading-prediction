"""
Market Master - AI-Powered Trading Assistant

A comprehensive MLOps system that provides real-time trading recommendations
by analyzing 18+ technical indicators simultaneously across all asset classes.
"""

__version__ = "1.0.0"
__author__ = "Market Master Team"
__description__ = "AI-Powered Trading Assistant for Universal Asset Classes"

from .config import settings
from .utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Export main components
__all__ = [
    "settings",
    "logger",
] 