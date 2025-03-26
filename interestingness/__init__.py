"""
Interestingness Metrics

A package for evaluating how "interesting" or meaningful a grouping is
in aggregated data. This helps identify which grouping variables create
the most significant or meaningful distinctions in your metrics of interest.

Example usage:
    from interestingness_metrics import group_variance, coefficient_of_variation
    from interestingness_utils import interestingness_report

    # Using your pre-aggregated DataFrame
    results = interestingness_report(
        df, mean_col='average_value', count_col='sample_count',
        group_col='segment', var_col='variance'
    )
"""

# Import core metrics
from .interestingness_metrics import (
    anova_f_statistic,
    coefficient_of_variation,
    effect_size_f,
    gini_coefficient,
    group_variance,
    max_deviation_ratio,
    range_to_mean_ratio,
)

# Import advanced metrics
from .interestingness_metrics_advanced import (
    discriminative_power,
    entropy_reduction,
    group_separation_index,
    kruskal_wallis_h,
    outlier_group_score,
    permutation_significance,
)

# Import utilities
from .interestingness_utils import (
    evaluate_all_metrics,
    flatten_dict,
    interestingness_report,
    plot_group_means,
    rank_groups_by_metric,
)

__all__ = [
    # Core metrics
    "group_variance",
    "coefficient_of_variation",
    "max_deviation_ratio",
    "range_to_mean_ratio",
    "anova_f_statistic",
    "effect_size_f",
    "gini_coefficient",
    # Advanced metrics
    "entropy_reduction",
    "kruskal_wallis_h",
    "discriminative_power",
    "outlier_group_score",
    "group_separation_index",
    "permutation_significance",
    # Utilities
    "rank_groups_by_metric",
    "evaluate_all_metrics",
    "plot_group_means",
    "interestingness_report",
    "flatten_dict",
]
