"""
Interestingness Metrics for Grouped Data

A collection of functions to evaluate how "interesting" a grouping is based on
various statistical metrics. These functions expect pre-aggregated pandas DataFrames
where rows represent different groups and columns contain metrics like mean, count,
variance, etc.
"""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def group_variance(
    df: pd.DataFrame, mean_col: str, weight_col: Optional[str] = None
) -> float:
    """
    Calculate the variance of group means to measure how spread out
    the groups are from each other.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    weight_col : str, optional
        Column name containing weights (usually counts) for each group.
        If provided, calculates weighted variance.

    Returns:
    --------
    float
        Variance of the group means
    """
    if df.empty or len(df) <= 1:
        return 0.0

    if weight_col is not None:
        # Calculate weighted variance
        weights = df[weight_col]
        means = df[mean_col]
        weighted_mean = np.average(means, weights=weights)
        weighted_variance = np.average((means - weighted_mean) ** 2, weights=weights)
        return weighted_variance
    else:
        # Simple unweighted variance
        return df[mean_col].var()


def coefficient_of_variation(
    df: pd.DataFrame, mean_col: str, weight_col: Optional[str] = None
) -> float:
    """
    Calculate the coefficient of variation (CV) across group means.
    This is the standard deviation divided by the mean, which normalizes
    the variance to handle different scales.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    weight_col : str, optional
        Column name containing weights for weighted calculation

    Returns:
    --------
    float
        Coefficient of variation of the group means
    """
    if df.empty or len(df) <= 1:
        return 0.0

    if weight_col is not None:
        # Weighted calculation
        weights = df[weight_col]
        means = df[mean_col]
        weighted_mean = np.average(means, weights=weights)
        weighted_variance = np.average((means - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_variance)
        return weighted_std / weighted_mean if weighted_mean != 0 else 0.0
    else:
        # Simple unweighted calculation
        mean = df[mean_col].mean()
        std = df[mean_col].std()
        return std / mean if mean != 0 else 0.0


def max_deviation_ratio(df: pd.DataFrame, mean_col: str) -> float:
    """
    Calculate the maximum deviation ratio from the overall mean.
    This identifies how extreme the outlier groups are.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group

    Returns:
    --------
    float
        Ratio of the maximum distance from overall mean to the overall mean
    """
    if df.empty or len(df) <= 1:
        return 0.0

    overall_mean = df[mean_col].mean()

    if overall_mean == 0:
        return 0.0

    max_distance = max(abs(df[mean_col] - overall_mean))
    return max_distance / abs(overall_mean)


def range_to_mean_ratio(df: pd.DataFrame, mean_col: str) -> float:
    """
    Calculate the ratio of the range of group means to the overall mean.
    This measures the span of values relative to the central tendency.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group

    Returns:
    --------
    float
        Ratio of range to mean
    """
    if df.empty or len(df) <= 1:
        return 0.0

    overall_mean = df[mean_col].mean()
    group_range = df[mean_col].max() - df[mean_col].min()

    if overall_mean == 0:
        return 0.0

    return group_range / abs(overall_mean)


def anova_f_statistic(
    df: pd.DataFrame, mean_col: str, count_col: str, var_col: Optional[str] = None
) -> dict:
    """
    Calculate one-way ANOVA F-statistic and p-value to test if there are
    significant differences between group means.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count (sample size) for each group
    var_col : str, optional
        Column name containing the variance for each group. If not provided,
        within-group variance is estimated using a pooled approach.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'f_statistic': The F-statistic
        - 'p_value': The p-value
        - 'significant': Boolean indicating if result is significant (p < 0.05)
    """
    if df.empty or len(df) <= 1:
        return {"f_statistic": 0.0, "p_value": 1.0, "significant": False}

    # Get required data
    group_means = df[mean_col]
    group_counts = df[count_col]

    # Calculate grand mean
    total_count = group_counts.sum()
    grand_mean = np.average(group_means, weights=group_counts)

    # Calculate between-group sum of squares
    between_ss = np.sum(group_counts * (group_means - grand_mean) ** 2)
    between_df = len(df) - 1

    # Calculate within-group sum of squares
    if var_col is not None:
        # If variance is provided
        group_vars = df[var_col]
        within_ss = np.sum(group_counts * group_vars)
    else:
        # Estimate pooled within-group variance
        # Note: This is an approximation without raw data
        # Assuming equal variance across groups
        within_ss = np.sum((group_counts - 1) * group_means.var())

    within_df = total_count - len(df)

    # Handle potential division by zero
    if within_ss == 0 or within_df == 0:
        return {"f_statistic": float("inf"), "p_value": 0.0, "significant": True}

    # Calculate F-statistic
    between_ms = between_ss / between_df
    within_ms = within_ss / within_df
    f_stat = between_ms / within_ms

    # Calculate p-value
    p_value = 1.0
    try:
        p_value = stats.f.sf(f_stat, between_df, within_df)
    except:
        pass

    return {"f_statistic": f_stat, "p_value": p_value, "significant": p_value < 0.05}


def effect_size_f(
    df: pd.DataFrame, mean_col: str, count_col: str, var_col: Optional[str] = None
) -> float:
    """
    Calculate Cohen's f effect size for one-way ANOVA designs.
    This is a standardized measure of the magnitude of the effect.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count (sample size) for each group
    var_col : str, optional
        Column name containing the variance for each group

    Returns:
    --------
    float
        Cohen's f effect size where:
        - Small effect: f = 0.10
        - Medium effect: f = 0.25
        - Large effect: f = 0.40
    """
    if df.empty or len(df) <= 1:
        return 0.0

    # Get required data
    group_means = df[mean_col]
    group_counts = df[count_col]

    # Calculate grand mean
    grand_mean = np.average(group_means, weights=group_counts)

    # Calculate between-group variance
    between_var = (
        np.sum(group_counts * (group_means - grand_mean) ** 2) / group_counts.sum()
    )

    # Calculate within-group variance
    if var_col is not None:
        # If variance is provided
        within_var = np.average(df[var_col], weights=group_counts)
    else:
        # Use an approximation
        within_var = group_means.var()

    # Handle potential division by zero
    if within_var == 0:
        return float("inf")

    # Calculate Cohen's f
    f = np.sqrt(between_var / within_var)

    return f


def gini_coefficient(
    df: pd.DataFrame, mean_col: str, count_col: Optional[str] = None
) -> float:
    """
    Calculate the Gini coefficient to measure inequality among group means.
    Values range from 0 (perfect equality) to 1 (perfect inequality).

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str, optional
        Column name containing counts for weighted calculation

    Returns:
    --------
    float
        Gini coefficient of inequality
    """
    if df.empty or len(df) <= 1:
        return 0.0

    # Get sorted means
    if count_col is not None:
        # Expand means by counts for weighted calculation
        means = []
        for _, row in df.iterrows():
            means.extend([row[mean_col]] * int(row[count_col]))
        means = np.array(means)
    else:
        means = df[mean_col].values

    # Convert to absolute values if there are negatives
    means = np.abs(means)

    # Sort values
    means = np.sort(means)

    # Calculate Gini coefficient
    n = len(means)
    indices = np.arange(1, n + 1)
    return (
        (2 * np.sum(indices * means) / (n * np.sum(means))) - (n + 1) / n
        if np.sum(means) > 0
        else 0.0
    )
