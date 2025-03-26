"""
Distribution-based interestingness metrics.

This module contains functions that analyze differences in distributions
across segments to identify when a segment or combination behaves unusually.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_aggregated, get_group_combinations


def entropy_score(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Calculate the entropy of the target variable within each group.

    Lower entropy indicates more uniformity (less surprise) within the group.
    Higher entropy indicates more variability (more surprise) within the group.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    count_col : str
        Column name with the count per group
    normalize : bool
        Whether to normalize by the maximum possible entropy

    Returns:
    --------
    dict
        Dictionary with group values as keys and entropy scores as values
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # We need to discretize the target for entropy calculation if it's continuous
    n_bins = min(10, len(agg_df) // 5)  # Rule of thumb: at least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(
        agg_df[target_col], q=n_bins, duplicates="drop", labels=False
    )

    # Get all unique combinations of group values
    group_combos = get_group_combinations(agg_df, group_cols)

    # Calculate entropy for each group combination
    result = {}

    for combo in group_combos:
        # Create filter for this combination
        combo_filter = True
        for col, val in zip(group_cols, combo):
            combo_filter = combo_filter & (agg_df[col] == val)

        # Get the bin counts for this combination
        bin_counts = agg_df.loc[combo_filter, count_col].values

        # Calculate entropy
        if bin_counts.sum() > 0:
            probs = bin_counts / bin_counts.sum()
            entropy = stats.entropy(probs)

            if normalize:
                # Maximum entropy is log(n) where n is the number of bins
                max_entropy = np.log(len(bin_counts))
                if max_entropy > 0:
                    entropy = entropy / max_entropy

            combo_name = "__".join([str(val) for val in combo])
            result[combo_name] = entropy

    return result


def kl_divergence(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    reference_group: Optional[Union[str, Tuple]] = None,
    count_col: str = "count",
    symmetric: bool = True,
) -> Dict[str, float]:
    """
    Calculate Kullback-Leibler divergence between the distribution of each group
    and a reference distribution (default: overall distribution).

    Higher KL divergence indicates a more unusual distribution compared to reference.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    reference_group : str or tuple, optional
        Reference group to compare against (if None, uses overall distribution)
    count_col : str
        Column name with the count per group
    symmetric : bool
        Whether to use symmetric KL divergence (Jensen-Shannon divergence)

    Returns:
    --------
    dict
        Dictionary with group values as keys and KL divergence scores as values
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # We need to discretize the target for KL calculation if it's continuous
    n_bins = min(10, len(agg_df) // 5)  # At least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(
        agg_df[target_col], q=n_bins, duplicates="drop", labels=False
    )

    # Get all unique combinations of group values
    group_combos = get_group_combinations(agg_df, group_cols)

    # Calculate reference distribution
    if reference_group is None:
        # Use overall distribution as reference
        ref_dist = agg_df.groupby("target_binned")[count_col].sum()
        ref_dist = ref_dist / ref_dist.sum()
    else:
        # Use specified reference group
        if isinstance(reference_group, str) and len(group_cols) == 1:
            reference_group = (reference_group,)

        # Create filter for reference group
        ref_filter = True
        for col, val in zip(group_cols, reference_group):
            ref_filter = ref_filter & (agg_df[col] == val)

        # Get reference distribution
        ref_dist = agg_df.loc[ref_filter].groupby("target_binned")[count_col].sum()
        ref_dist = ref_dist / ref_dist.sum()

    # Ensure reference distribution includes all bins
    for bin_id in range(n_bins):
        if bin_id not in ref_dist.index:
            ref_dist[bin_id] = 1e-10  # Small non-zero value for bins with no data

    ref_dist = ref_dist.sort_index()

    # Calculate KL divergence for each group combination
    result = {}

    for combo in group_combos:
        # Skip reference group if specified
        if combo == reference_group:
            continue

        # Create filter for this combination
        combo_filter = True
        for col, val in zip(group_cols, combo):
            combo_filter = combo_filter & (agg_df[col] == val)

        # Get the distribution for this combination
        group_dist = agg_df.loc[combo_filter].groupby("target_binned")[count_col].sum()
        group_dist = group_dist / group_dist.sum()

        # Ensure group distribution includes all bins
        for bin_id in range(n_bins):
            if bin_id not in group_dist.index:
                group_dist[bin_id] = 1e-10  # Small non-zero value for bins with no data

        group_dist = group_dist.sort_index()

        # Calculate KL divergence
        if symmetric:
            # Jensen-Shannon divergence (symmetric)
            m_dist = 0.5 * (ref_dist + group_dist)
            js_div = 0.5 * (
                stats.entropy(ref_dist, m_dist) + stats.entropy(group_dist, m_dist)
            )
            divergence = js_div
        else:
            # Regular KL divergence (asymmetric)
            kl_div = stats.entropy(group_dist, ref_dist)
            divergence = kl_div

        combo_name = "__".join([str(val) for val in combo])
        result[combo_name] = divergence

    return result


def distribution_outliers(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    method: str = "z_score",
    threshold: float = 2.0,
    count_col: str = "count",
) -> pd.DataFrame:
    """
    Find groups with outlier distributions based on statistical criteria.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    method : str
        Method to use for detecting outliers: 'z_score', 'iqr', or 'percentile'
    threshold : float
        Threshold for considering a value an outlier (e.g., z-score > 2.0)
    count_col : str
        Column name with the count per group

    Returns:
    --------
    DataFrame
        DataFrame with outlier groups and their scores
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # Get group statistics
    group_stats = []

    # For each group combination
    for _, group_df in agg_df.groupby(group_cols):
        # Get group values
        group_values = {}
        for col in group_cols:
            group_values[col] = group_df[col].iloc[0]

        # Calculate statistics for this group
        stats_dict = {
            "mean": np.average(group_df[target_col], weights=group_df[count_col]),
            "count": group_df[count_col].sum(),
        }

        # Add standard deviation if available
        if "std" in group_df.columns:
            # For weighted pooled std, we need individual stddevs
            # This is approximate when only given aggregated data
            stats_dict["std"] = np.sqrt(
                np.average(group_df["std"] ** 2, weights=group_df[count_col])
            )

        # Combined the group values and statistics
        group_stats.append({**group_values, **stats_dict})

    if not group_stats:
        return pd.DataFrame()

    # Create DataFrame from statistics
    stats_df = pd.DataFrame(group_stats)

    # Calculate outlier scores based on the method
    if method == "z_score":
        # Z-score method
        overall_mean = np.average(stats_df["mean"], weights=stats_df["count"])
        overall_std = np.sqrt(
            np.average(
                (stats_df["mean"] - overall_mean) ** 2, weights=stats_df["count"]
            )
        )

        if overall_std > 0:
            stats_df["outlier_score"] = np.abs(
                (stats_df["mean"] - overall_mean) / overall_std
            )
        else:
            stats_df["outlier_score"] = 0

    elif method == "iqr":
        # Interquartile range method
        q1 = stats_df["mean"].quantile(0.25)
        q3 = stats_df["mean"].quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            stats_df["outlier_score"] = np.maximum(
                np.abs(stats_df["mean"] - q1) / iqr, np.abs(stats_df["mean"] - q3) / iqr
            )
        else:
            stats_df["outlier_score"] = 0

    elif method == "percentile":
        # Percentile method
        p10 = stats_df["mean"].quantile(0.1)
        p90 = stats_df["mean"].quantile(0.9)
        range_90 = p90 - p10

        if range_90 > 0:
            stats_df["outlier_score"] = (
                np.maximum(
                    np.abs(stats_df["mean"] - p10) / range_90,
                    np.abs(stats_df["mean"] - p90) / range_90,
                )
                * 2
            )  # Scale to make comparable with z-score
        else:
            stats_df["outlier_score"] = 0
    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'z_score', 'iqr', or 'percentile'."
        )

    # Filter outliers based on threshold
    outliers = stats_df[stats_df["outlier_score"] >= threshold].copy()

    # Sort by outlier score (most extreme first)
    outliers = outliers.sort_values("outlier_score", ascending=False)

    # Flag outlier direction
    overall_mean = np.average(stats_df["mean"], weights=stats_df["count"])
    outliers["direction"] = np.where(outliers["mean"] > overall_mean, "higher", "lower")

    return outliers
