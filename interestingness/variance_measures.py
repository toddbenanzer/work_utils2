"""
Functions for measuring interestingness of segments based on variance and dispersion.
This module provides variance-based methods to quantify how "interesting" a segmentation
is based on how much a metric varies across segments.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def calculate_coefficient_variation(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: Union[str, List[str]],
    weight_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate coefficient of variation (CV) across segments as an interestingness measure.
    CV = standard deviation / mean, which provides a normalized measure of dispersion.

    Args:
        df: Aggregated DataFrame containing segments and values
        value_col: Column containing the metric to analyze (e.g., 'avg_balance')
        segment_cols: Column(s) defining the segments to analyze
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        Dictionary containing coefficient of variation and supporting statistics
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [col for col in segment_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    if weight_col is not None and weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    # Extract values
    values = df[value_col].values

    # Calculate weighted statistics if weight_col is provided
    if weight_col is not None:
        weights = df[weight_col].values

        # Normalize weights to sum to 1
        weights_normalized = weights / np.sum(weights)

        # Calculate weighted mean
        weighted_mean = np.sum(values * weights_normalized)

        # Calculate weighted variance
        weighted_var = np.sum(weights_normalized * (values - weighted_mean) ** 2)

        # Calculate weighted standard deviation
        weighted_std = np.sqrt(weighted_var)

        # Calculate weighted coefficient of variation
        weighted_cv = weighted_std / weighted_mean if weighted_mean != 0 else 0

        result = {
            "coefficient_variation": weighted_cv,
            "weighted_mean": weighted_mean,
            "weighted_std": weighted_std,
            "weighted_var": weighted_var,
            "method": "weighted",
        }
    else:
        # Unweighted calculation
        mean_value = np.mean(values)
        std_value = np.std(values)
        cv = std_value / mean_value if mean_value != 0 else 0

        result = {
            "coefficient_variation": cv,
            "mean": mean_value,
            "std": std_value,
            "var": std_value**2,
            "method": "unweighted",
        }

    # Add segment information
    result["segment_cols"] = segment_cols
    result["value_col"] = value_col
    result["n_segments"] = len(df)

    return result


def calculate_weighted_variance(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: Union[str, List[str]],
    weight_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate variance of a metric across segments, weighted by segment size.
    This measures how dispersed the values are across segments.

    Args:
        df: Aggregated DataFrame containing segments and values
        value_col: Column containing the metric to analyze (e.g., 'avg_balance')
        segment_cols: Column(s) defining the segments to analyze
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        Dictionary containing weighted variance and supporting statistics
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [col for col in segment_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # If no weight column is provided, use equal weights
    if weight_col is None:
        df["_weight"] = 1
        weight_col = "_weight"
    elif weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    # Extract values and weights
    values = df[value_col].values
    weights = df[weight_col].values

    # Normalize weights to sum to 1
    weights_normalized = weights / np.sum(weights)

    # Calculate overall weighted mean
    weighted_mean = np.sum(values * weights_normalized)

    # Calculate weighted variance
    weighted_var = np.sum(weights_normalized * (values - weighted_mean) ** 2)

    # Calculate weighted standard deviation
    weighted_std = np.sqrt(weighted_var)

    result = {
        "weighted_variance": weighted_var,
        "weighted_std": weighted_std,
        "weighted_mean": weighted_mean,
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": len(df),
    }

    # Calculate additional statistics
    # Weighted range (max - min)
    result["weighted_range"] = np.max(values) - np.min(values)

    # Weighted interquartile range (75th - 25th percentile)
    # Note: This is an approximation as weighted percentiles are complex
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights_normalized[sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    p25_idx = np.searchsorted(cumulative_weights, 0.25)
    p75_idx = np.searchsorted(cumulative_weights, 0.75)

    if p25_idx < len(sorted_values) and p75_idx < len(sorted_values):
        result["weighted_iqr"] = sorted_values[p75_idx] - sorted_values[p25_idx]
    else:
        result["weighted_iqr"] = np.nan

    return result


def calculate_max_deviation(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: Union[str, List[str]],
    weight_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate maximum deviation from the overall average across segments.
    This identifies segments that are most different from the average.

    Args:
        df: Aggregated DataFrame containing segments and values
        value_col: Column containing the metric to analyze (e.g., 'avg_balance')
        segment_cols: Column(s) defining the segments to analyze
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        Dictionary containing maximum deviation and supporting statistics
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [col for col in segment_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Calculate overall mean (weighted if weight_col is provided)
    if weight_col is not None and weight_col in df.columns:
        weights = df[weight_col].values
        weights_normalized = weights / np.sum(weights)
        overall_mean = np.sum(df[value_col].values * weights_normalized)
    else:
        overall_mean = df[value_col].mean()

    # Calculate absolute deviations from overall mean
    deviations = np.abs(df[value_col] - overall_mean)
    max_deviation = np.max(deviations)

    # Find segment with maximum deviation
    max_deviation_idx = np.argmax(deviations)
    max_deviation_segment = df.iloc[max_deviation_idx]

    # Create segment identifier string
    segment_id = ", ".join(
        [f"{col}={max_deviation_segment[col]}" for col in segment_cols]
    )

    # Calculate relative deviation (as percentage of mean)
    relative_deviation = (
        (max_deviation / overall_mean) * 100 if overall_mean != 0 else np.nan
    )

    result = {
        "max_deviation": max_deviation,
        "max_deviation_relative": relative_deviation,
        "max_deviation_segment": segment_id,
        "max_deviation_value": max_deviation_segment[value_col],
        "overall_mean": overall_mean,
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": len(df),
    }

    return result


def calculate_dispersion_ratio(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: Union[str, List[str]],
    weight_col: Optional[str] = None,
) -> Dict[str, float]:
    """
    Calculate dispersion ratio, a combined measure of variance and range.
    Dispersion ratio = weighted_variance / (max_value - min_value)²

    Args:
        df: Aggregated DataFrame containing segments and values
        value_col: Column containing the metric to analyze (e.g., 'avg_balance')
        segment_cols: Column(s) defining the segments to analyze
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        Dictionary containing dispersion ratio and supporting statistics
    """
    # Get weighted variance
    weighted_var_results = calculate_weighted_variance(
        df, value_col, segment_cols, weight_col
    )
    weighted_var = weighted_var_results["weighted_variance"]

    # Calculate range
    value_range = df[value_col].max() - df[value_col].min()

    # Calculate dispersion ratio
    # A higher ratio means the variance is distributed across segments
    # A lower ratio means the variance is concentrated in extreme values
    dispersion_ratio = weighted_var / (value_range**2) if value_range != 0 else 0

    result = {
        "dispersion_ratio": dispersion_ratio,
        "weighted_variance": weighted_var,
        "value_range": value_range,
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": len(df),
    }

    return result


def calculate_all_variance_measures(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: Union[str, List[str]],
    weight_col: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate all variance-based interestingness measures.

    Args:
        df: Aggregated DataFrame containing segments and values
        value_col: Column containing the metric to analyze (e.g., 'avg_balance')
        segment_cols: Column(s) defining the segments to analyze
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        Dictionary containing all interestingness measures
    """
    return {
        "coefficient_variation": calculate_coefficient_variation(
            df, value_col, segment_cols, weight_col
        ),
        "weighted_variance": calculate_weighted_variance(
            df, value_col, segment_cols, weight_col
        ),
        "max_deviation": calculate_max_deviation(
            df, value_col, segment_cols, weight_col
        ),
        "dispersion_ratio": calculate_dispersion_ratio(
            df, value_col, segment_cols, weight_col
        ),
    }
