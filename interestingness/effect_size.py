"""
Effect size metrics for interestingness analysis.

This module contains functions that calculate standardized measures of
difference between segments, providing consistent ways to compare the
magnitude of differences.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .utils import ensure_aggregated


def cohens_d(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    group_a: Union[str, int, float],
    group_b: Optional[Union[str, int, float]] = None,
    count_col: str = "count",
    std_col: Optional[str] = "std",
    pooled: bool = True,
) -> float:
    """
    Calculate Cohen's d effect size between two groups.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_col : str
        Column name for the grouping variable
    group_a : str, int, or float
        Value of group_col identifying the first group
    group_b : str, int, or float, optional
        Value of group_col identifying the second group
        If None, compares group_a to all other groups combined
    count_col : str
        Column name with the count per group
    std_col : str, optional
        Column name with the standard deviation per group
    pooled : bool
        Whether to use pooled standard deviation (True) or just group A's std (False)

    Returns:
    --------
    float
        Cohen's d effect size
    """
    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col, std_col=std_col)

    # Get group A data
    group_a_data = agg_df[agg_df[group_col] == group_a]
    if len(group_a_data) == 0:
        raise ValueError(f"Group '{group_a}' not found in column '{group_col}'")

    # Get group B data
    if group_b is not None:
        group_b_data = agg_df[agg_df[group_col] == group_b]
        if len(group_b_data) == 0:
            raise ValueError(f"Group '{group_b}' not found in column '{group_col}'")
    else:
        # Compare to all other groups combined
        group_b_data = agg_df[agg_df[group_col] != group_a]
        if len(group_b_data) == 0:
            raise ValueError(
                f"No groups other than '{group_a}' found in column '{group_col}'"
            )

    # Calculate weighted means for each group
    mean_a = np.average(group_a_data[target_col], weights=group_a_data[count_col])
    mean_b = np.average(group_b_data[target_col], weights=group_b_data[count_col])

    # Calculate mean difference
    mean_diff = mean_a - mean_b

    # If there's no difference, return 0
    if mean_diff == 0:
        return 0.0

    # Calculate standard deviations
    if std_col in agg_df.columns:
        # Use provided standard deviations
        std_a = np.sqrt(
            np.average(group_a_data[std_col] ** 2, weights=group_a_data[count_col])
        )

        if pooled:
            std_b = np.sqrt(
                np.average(group_b_data[std_col] ** 2, weights=group_b_data[count_col])
            )

            # Calculate pooled standard deviation
            n_a = group_a_data[count_col].sum()
            n_b = group_b_data[count_col].sum()

            pooled_std = np.sqrt(
                ((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2)
            )
            denominator = pooled_std
        else:
            # Use only group A's standard deviation
            denominator = std_a
    else:
        # Estimate standard deviation from the data
        # For aggregated data, this is approximate and assumes similar group sizes
        overall_mean = np.average(agg_df[target_col], weights=agg_df[count_col])
        overall_var = np.average(
            (agg_df[target_col] - overall_mean) ** 2, weights=agg_df[count_col]
        )
        denominator = np.sqrt(overall_var)

    # Avoid division by zero
    if denominator == 0:
        return np.inf if mean_diff > 0 else -np.inf

    # Calculate Cohen's d
    d = mean_diff / denominator

    return d


def standardized_difference(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
    std_col: Optional[str] = "std",
) -> pd.DataFrame:
    """
    Calculate standardized differences for all combinations of grouping variables.

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
    std_col : str, optional
        Column name with the standard deviation per group

    Returns:
    --------
    DataFrame
        DataFrame with pairwise standardized differences for each group
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col, std_col=std_col)

    # Calculate overall standard deviation for normalization
    overall_mean = np.average(agg_df[target_col], weights=agg_df[count_col])
    overall_var = np.average(
        (agg_df[target_col] - overall_mean) ** 2, weights=agg_df[count_col]
    )
    overall_std = np.sqrt(overall_var)

    if overall_std == 0:
        # No variation to compare
        return pd.DataFrame()

    # For each grouping variable, calculate pairwise standardized differences
    results = []

    for col in group_cols:
        # Get unique values for this grouping variable
        unique_values = agg_df[col].unique()

        # Calculate group means
        group_means = agg_df.groupby(col).apply(
            lambda x: np.average(x[target_col], weights=x[count_col])
        )

        # Calculate group standard deviations if available
        if std_col in agg_df.columns:
            group_stds = agg_df.groupby(col).apply(
                lambda x: np.sqrt(np.average(x[std_col] ** 2, weights=x[count_col]))
            )
        else:
            # Use overall standard deviation
            group_stds = pd.Series(overall_std, index=group_means.index)

        # Calculate group sizes
        group_sizes = agg_df.groupby(col)[count_col].sum()

        # Calculate standardized differences for all pairs
        for i, val1 in enumerate(unique_values):
            for val2 in unique_values[i + 1 :]:
                # Calculate Cohen's d
                mean_diff = group_means[val1] - group_means[val2]

                # Pooled standard deviation
                n1 = group_sizes[val1]
                n2 = group_sizes[val2]
                pooled_std = np.sqrt(
                    (
                        (n1 - 1) * group_stds[val1] ** 2
                        + (n2 - 1) * group_stds[val2] ** 2
                    )
                    / (n1 + n2 - 2)
                )

                # Standardized difference
                if pooled_std > 0:
                    std_diff = mean_diff / pooled_std
                else:
                    std_diff = (
                        0.0
                        if mean_diff == 0
                        else (np.inf if mean_diff > 0 else -np.inf)
                    )

                # Store the result
                results.append(
                    {
                        "group_var": col,
                        "value1": val1,
                        "value2": val2,
                        "mean1": group_means[val1],
                        "mean2": group_means[val2],
                        "std1": group_stds[val1],
                        "std2": group_stds[val2],
                        "n1": n1,
                        "n2": n2,
                        "mean_diff": mean_diff,
                        "std_diff": std_diff,
                        "abs_std_diff": abs(std_diff),
                    }
                )

    if not results:
        return pd.DataFrame()

    # Convert to DataFrame and sort by absolute standardized difference
    result_df = pd.DataFrame(results)
    return result_df.sort_values("abs_std_diff", ascending=False)


def percent_difference(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    reference_value: Optional[Union[str, Tuple]] = None,
    count_col: str = "count",
    relative: bool = True,
) -> pd.DataFrame:
    """
    Calculate percent differences between groups and a reference value.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    reference_value : str or tuple, optional
        Reference group value(s) to compare against
        If None, compares to the overall mean
    count_col : str
        Column name with the count per group
    relative : bool
        If True, calculates relative percent difference: (group - ref) / ref * 100
        If False, calculates absolute percent difference: (group - ref) / overall_mean * 100

    Returns:
    --------
    DataFrame
        DataFrame with percent differences for each group
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # Calculate overall weighted mean
    overall_mean = np.average(agg_df[target_col], weights=agg_df[count_col])

    # Calculate reference value
    if reference_value is None:
        # Use overall mean as reference
        ref_value = overall_mean
    else:
        # Use specified reference value
        if isinstance(reference_value, str) and len(group_cols) == 1:
            ref_filter = agg_df[group_cols[0]] == reference_value
        else:
            # Build filter for multiple columns
            ref_filter = True
            for col, val in zip(group_cols, reference_value):
                ref_filter = ref_filter & (agg_df[col] == val)

        ref_data = agg_df[ref_filter]
        if len(ref_data) == 0:
            raise ValueError(f"Reference value not found in data: {reference_value}")

        ref_value = np.average(ref_data[target_col], weights=ref_data[count_col])

    # Avoid division by zero
    if ref_value == 0:
        if relative:
            raise ValueError(
                "Reference value is zero, cannot calculate relative percent difference"
            )
        else:
            # For absolute difference, we can use the overall mean if it's not zero
            if overall_mean == 0:
                raise ValueError(
                    "Overall mean is zero, cannot calculate percent difference"
                )
    elif not relative and overall_mean == 0:
        raise ValueError(
            "Overall mean is zero, cannot calculate absolute percent difference"
        )

    # Calculate group means
    group_results = []

    for _, group in agg_df.groupby(group_cols):
        # Get group values
        group_values = {}
        for col in group_cols:
            group_values[col] = group[col].iloc[0]

        # Calculate group mean
        group_mean = np.average(group[target_col], weights=group[count_col])

        # Calculate percent difference
        diff = group_mean - ref_value
        if relative:
            pct_diff = (diff / ref_value) * 100 if ref_value != 0 else np.inf
        else:
            pct_diff = (diff / overall_mean) * 100 if overall_mean != 0 else np.inf

        # Store the result
        group_results.append(
            {
                **group_values,
                "mean": group_mean,
                "reference": ref_value,
                "diff": diff,
                "percent_diff": pct_diff,
                "abs_percent_diff": abs(pct_diff),
            }
        )

    if not group_results:
        return pd.DataFrame()

    # Convert to DataFrame and sort by absolute percent difference
    result_df = pd.DataFrame(group_results)
    return result_df.sort_values("abs_percent_diff", ascending=False)
