"""
Interestingness - A toolkit for identifying interesting patterns in grouped data

This package provides functions to calculate various measures of "interestingness"
in pandas GroupBy results, helping identify notable patterns and relationships.
"""

# Import key functions from submodules to expose at package level
from .advanced import (
    anomaly_score,
    concentration_ratio,
    cramers_v,
    gini_impurity,
    information_gain,
    theil_index,
)
from .contribution import feature_importance, mutual_information, variance_explained
from .distribution import distribution_outliers, entropy_score, kl_divergence
from .effect_size import cohens_d, percent_difference, standardized_difference

# Version
__version__ = "0.1.0"
