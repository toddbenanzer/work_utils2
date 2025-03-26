"""
Trend Metrics Package

A package for evaluating how "interesting" or unusual a group's trend is
compared to the overall trend or other groups. This helps identify which groups
show distinct patterns over time that might warrant further investigation.

Example usage:
    from trend_metrics import trend_divergence_score
    from trend_utils import trend_analysis_report

    # Using your pre-aggregated DataFrame with time component
    results = trend_analysis_report(
        df, group_col='segment', time_col='month',
        mean_col='average_value', count_col='sample_count'
    )
"""

# Import core trend metrics
from .trend_metrics import (
    acceleration_detection,
    trend_break_detection,
    trend_consistency,
    trend_direction_difference,
    trend_divergence_score,
    trend_slope_divergence,
    trend_volatility,
)

# Try to import advanced metrics (with statsmodels dependency)
try:
    from .trend_metrics_advanced import (
        advanced_trend_divergence_score,
        autocorrelation_pattern,
        outlier_period_detection,
        periodicity_detection,
        seasonal_pattern_difference,
        trend_stationarity_test,
    )

    __has_advanced_metrics__ = True
except ImportError:
    __has_advanced_metrics__ = False

# Import utility functions
from .trend_utils import (
    compare_trends_across_dimensions,
    detect_trend_anomalies,
    evaluate_all_trend_metrics,
    plot_group_trends,
    rank_groups_by_trend_divergence,
    trend_analysis_report,
)

__all__ = [
    # Core metrics
    "trend_slope_divergence",
    "trend_direction_difference",
    "trend_volatility",
    "trend_consistency",
    "acceleration_detection",
    "trend_break_detection",
    "trend_divergence_score",
    # Utilities
    "rank_groups_by_trend_divergence",
    "plot_group_trends",
    "evaluate_all_trend_metrics",
    "trend_analysis_report",
    "compare_trends_across_dimensions",
    "detect_trend_anomalies",
]

# Add advanced metrics to __all__ if available
if __has_advanced_metrics__:
    __all__.extend(
        [
            "seasonal_pattern_difference",
            "trend_stationarity_test",
            "autocorrelation_pattern",
            "outlier_period_detection",
            "periodicity_detection",
            "advanced_trend_divergence_score",
        ]
    )
