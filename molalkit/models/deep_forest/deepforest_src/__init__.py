"""
Standalone Deep Forest implementation using only sklearn backend.

This is a modified version of deep-forest that doesn't require C extensions.
All estimators use sklearn's RandomForest and ExtraTrees.
"""

from .cascade import CascadeForestClassifier

__all__ = ["CascadeForestClassifier"]
