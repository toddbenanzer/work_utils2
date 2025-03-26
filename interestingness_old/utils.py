"""
Utility functions for the interestingness package.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def ensure_aggregated(
    df: pd.DataFrame,
    value_col: str,
    count_col: Optional[str] = "count",
    mean_col: Optional[str] = None,
    std_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ensure the DataFrame has the required aggregation columns.
    If columns are missing, they will be added if possible.

    Parameters:
    -----------
    df : DataFrame
        The input DataFrame, typically a result of pandas groupby.agg
    value_col : str
        Column name that contains the values to measure (if not using mean)
    count_col : str, optional
        Column name with the count per group
    mean_col : str, optional
        Column name with the mean values per group
    std_col : str, optional
        Column name with the standard deviation per group

    Returns:
    --------
    DataFrame
        A DataFrame with additional computed columns if needed
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    # If mean_col is not provided, use value_col as the mean
    if mean_col is None:
        mean_col = value_col

    # Check if we need to compute standard deviation
    if std_col is not None and std_col not in result_df.columns:
        # We can't compute std from aggregated data without additional info
        raise ValueError(
            f"Column '{std_col}' not found in DataFrame. Cannot compute standard deviation from aggregated data."
        )

    # Check if we have a count column
    if count_col not in result_df.columns:
        # We assume equal counts if not provided
        result_df[count_col] = 1

    return result_df


def normalize_weights(df: pd.DataFrame, count_col: str = "count") -> pd.Series:
    """
    Normalize a count column to get weights that sum to 1.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the count column
    count_col : str
        Name of the column containing counts

    Returns:
    --------
    Series
        Normalized weights for each row
    """
    total = df[count_col].sum()
    return (
        df[count_col] / total if total > 0 else pd.Series([0] * len(df), index=df.index)
    )


def get_group_combinations(
    df: pd.DataFrame, group_cols: Union[str, List[str]]
) -> List[Tuple]:
    """
    Get all unique combinations of values in the specified group columns.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame to extract group combinations from
    group_cols : str or list of str
        Column name(s) for the grouping variables

    Returns:
    --------
    list of tuples
        All unique combinations found in the data
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    return list(df.groupby(group_cols).groups.keys())


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate a weighted mean.

    Parameters:
    -----------
    values : array-like
        Values to average
    weights : array-like
        Weights for each value

    Returns:
    --------
    float
        Weighted mean
    """
    return np.average(values, weights=weights) if len(values) > 0 else np.nan


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate a weighted variance.

    Parameters:
    -----------
    values : array-like
        Values to compute variance for
    weights : array-like
        Weights for each value

    Returns:
    --------
    float
        Weighted variance
    """
    if len(values) == 0:
        return np.nan

    # Normalize weights
    weights = weights / weights.sum()

    # Calculate weighted mean
    weighted_avg = weighted_mean(values, weights)

    # Calculate weighted variance
    variance = np.sum(weights * (values - weighted_avg) ** 2)

    # Adjust for weighted sample
    # This is the unbiased weighted sample variance
    return variance * len(values) / (len(values) - 1) if len(values) > 1 else 0


def flatten_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a DataFrame with MultiIndex columns to a flat structure.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with potentially MultiIndex columns

    Returns:
    --------
    DataFrame
        DataFrame with flattened column names
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).rstrip("_") for col in df.columns.values]
    return df
