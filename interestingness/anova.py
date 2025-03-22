"""
Functions for measuring interestingness of segments using ANOVA and related statistical methods.
This module quantifies how effectively a segmentation explains variance in a metric.
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats


def calculate_anova_f_statistic(
    df: pd.DataFrame, value_col: str, segment_cols: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Calculate F-statistic from ANOVA as an interestingness measure.
    A higher F-statistic indicates that segments explain more of the variance.

    Args:
        df: Raw DataFrame containing individual records (not aggregated)
        value_col: Column containing the metric to analyze (e.g., 'balance')
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        Dictionary containing F-statistic and related ANOVA results
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [col for col in segment_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Create a single segment identifier if multiple segment columns
    if len(segment_cols) > 1:
        # Combine segment columns into a single identifier
        df["_segment"] = df[segment_cols].astype(str).agg("_".join, axis=1)
        segment_id = "_segment"
    else:
        segment_id = segment_cols[0]

    # Extract groups for ANOVA
    groups = []
    unique_segments = df[segment_id].unique()

    # Skip if there's only one segment (ANOVA requires at least 2 groups)
    if len(unique_segments) < 2:
        return {
            "f_statistic": 0,
            "p_value": 1.0,
            "degrees_freedom_between": 0,
            "degrees_freedom_within": 0,
            "segment_cols": segment_cols,
            "value_col": value_col,
            "n_segments": len(unique_segments),
            "n_samples": len(df),
            "error": "Not enough segments for ANOVA",
        }

    # Extract values for each segment
    for segment in unique_segments:
        group_values = df[df[segment_id] == segment][value_col].dropna().values
        if len(group_values) > 0:
            groups.append(group_values)

    # Calculate ANOVA
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)

            # Calculate degrees of freedom
            n_segments = len(groups)
            n_total = sum(len(group) for group in groups)
            df_between = n_segments - 1
            df_within = n_total - n_segments

            result = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "degrees_freedom_between": df_between,
                "degrees_freedom_within": df_within,
                "segment_cols": segment_cols,
                "value_col": value_col,
                "n_segments": n_segments,
                "n_samples": n_total,
            }

            # Interpret p-value
            if p_value < 0.001:
                result["significance"] = "highly significant"
            elif p_value < 0.01:
                result["significance"] = "very significant"
            elif p_value < 0.05:
                result["significance"] = "significant"
            elif p_value < 0.1:
                result["significance"] = "marginally significant"
            else:
                result["significance"] = "not significant"

            return result
        except Exception as e:
            return {
                "f_statistic": np.nan,
                "p_value": np.nan,
                "segment_cols": segment_cols,
                "value_col": value_col,
                "error": str(e),
            }
    else:
        return {
            "f_statistic": 0,
            "p_value": 1.0,
            "segment_cols": segment_cols,
            "value_col": value_col,
            "n_segments": len(groups),
            "error": "Not enough valid groups for ANOVA",
        }


def calculate_eta_squared(
    df: pd.DataFrame, value_col: str, segment_cols: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Calculate eta-squared (η²) effect size measure for segmentation.
    η² represents the proportion of variance explained by the segments.

    Args:
        df: Raw DataFrame containing individual records (not aggregated)
        value_col: Column containing the metric to analyze (e.g., 'balance')
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        Dictionary containing eta-squared and supporting statistics
    """
    # First get ANOVA results
    anova_results = calculate_anova_f_statistic(df, value_col, segment_cols)

    # Check if there was an error in ANOVA calculation
    if "error" in anova_results and anova_results.get("f_statistic", 0) == 0:
        return {
            "eta_squared": 0,
            "segment_cols": segment_cols,
            "value_col": value_col,
            "error": anova_results.get("error", "Error in ANOVA calculation"),
        }

    # Calculate eta-squared from F-statistic
    f_stat = anova_results["f_statistic"]
    df_between = anova_results["degrees_freedom_between"]
    df_within = anova_results["degrees_freedom_within"]

    # Formula: η² = (df_between * F) / (df_between * F + df_within)
    eta_squared = (df_between * f_stat) / (df_between * f_stat + df_within)

    result = {
        "eta_squared": eta_squared,
        "f_statistic": f_stat,
        "p_value": anova_results["p_value"],
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": anova_results["n_segments"],
        "n_samples": anova_results["n_samples"],
    }

    # Interpret eta-squared
    if eta_squared < 0.01:
        result["effect_size"] = "negligible"
    elif eta_squared < 0.06:
        result["effect_size"] = "small"
    elif eta_squared < 0.14:
        result["effect_size"] = "medium"
    else:
        result["effect_size"] = "large"

    return result


def calculate_omega_squared(
    df: pd.DataFrame, value_col: str, segment_cols: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Calculate omega-squared (ω²) effect size measure for segmentation.
    ω² is a less biased alternative to eta-squared, especially for small samples.

    Args:
        df: Raw DataFrame containing individual records (not aggregated)
        value_col: Column containing the metric to analyze (e.g., 'balance')
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        Dictionary containing omega-squared and supporting statistics
    """
    # First get ANOVA results
    anova_results = calculate_anova_f_statistic(df, value_col, segment_cols)

    # Check if there was an error in ANOVA calculation
    if "error" in anova_results and anova_results.get("f_statistic", 0) == 0:
        return {
            "omega_squared": 0,
            "segment_cols": segment_cols,
            "value_col": value_col,
            "error": anova_results.get("error", "Error in ANOVA calculation"),
        }

    # Extract values from ANOVA results
    f_stat = anova_results["f_statistic"]
    df_between = anova_results["degrees_freedom_between"]
    df_within = anova_results["degrees_freedom_within"]
    n_total = anova_results["n_samples"]

    # Calculate Sum of Squares
    # Create a single segment identifier if multiple segment columns
    if isinstance(segment_cols, list) and len(segment_cols) > 1:
        # Combine segment columns into a single identifier
        df["_segment"] = df[segment_cols].astype(str).agg("_".join, axis=1)
        segment_id = "_segment"
    else:
        segment_id = segment_cols[0] if isinstance(segment_cols, list) else segment_cols

    # Calculate SST (Total Sum of Squares)
    grand_mean = df[value_col].mean()
    sst = np.sum((df[value_col] - grand_mean) ** 2)

    # Calculate SSB (Between Sum of Squares)
    segments = df.groupby(segment_id)[value_col]
    ssb = np.sum(segments.count() * (segments.mean() - grand_mean) ** 2)

    # Calculate MSE (Mean Square Error) = SSW / df_within
    # where SSW (Within Sum of Squares) = SST - SSB
    ssw = sst - ssb
    mse = ssw / df_within if df_within > 0 else 0

    # Calculate omega-squared
    # Formula: ω² = (SSB - (df_between * MSE)) / (SST + MSE)
    omega_squared = (ssb - (df_between * mse)) / (sst + mse) if (sst + mse) > 0 else 0

    # Ensure omega-squared is not negative (can happen with very small effects)
    omega_squared = max(0, omega_squared)

    result = {
        "omega_squared": omega_squared,
        "f_statistic": f_stat,
        "p_value": anova_results["p_value"],
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": anova_results["n_segments"],
        "n_samples": n_total,
    }

    # Interpret omega-squared (similar to eta-squared interpretation)
    if omega_squared < 0.01:
        result["effect_size"] = "negligible"
    elif omega_squared < 0.06:
        result["effect_size"] = "small"
    elif omega_squared < 0.14:
        result["effect_size"] = "medium"
    else:
        result["effect_size"] = "large"

    return result


def calculate_between_within_variance_ratio(
    df: pd.DataFrame, value_col: str, segment_cols: Union[str, List[str]]
) -> Dict[str, float]:
    """
    Calculate the ratio of between-segment variance to within-segment variance.
    A higher ratio indicates the segmentation effectively separates different values.

    Args:
        df: Raw DataFrame containing individual records (not aggregated)
        value_col: Column containing the metric to analyze (e.g., 'balance')
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        Dictionary containing variance ratio and supporting statistics
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [col for col in segment_cols + [value_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Create a single segment identifier if multiple segment columns
    if len(segment_cols) > 1:
        # Combine segment columns into a single identifier
        df["_segment"] = df[segment_cols].astype(str).agg("_".join, axis=1)
        segment_id = "_segment"
    else:
        segment_id = segment_cols[0]

    # Calculate overall mean and variance
    overall_mean = df[value_col].mean()
    overall_var = df[value_col].var(ddof=1)  # Use sample variance (N-1 denominator)

    # Calculate between-segment variance
    segment_means = df.groupby(segment_id)[value_col].mean()
    segment_counts = df.groupby(segment_id)[value_col].count()

    # Weighted variance of segment means (between-segment variance)
    segment_weights = segment_counts / segment_counts.sum()
    between_var = np.sum(segment_weights * (segment_means - overall_mean) ** 2)

    # Calculate within-segment variance (average of segment variances)
    within_vars = df.groupby(segment_id)[value_col].var(ddof=1)
    within_var = np.sum(segment_weights * within_vars)

    # Calculate variance ratio
    var_ratio = between_var / within_var if within_var > 0 else np.nan

    result = {
        "variance_ratio": var_ratio,
        "between_variance": between_var,
        "within_variance": within_var,
        "segment_cols": segment_cols,
        "value_col": value_col,
        "n_segments": len(segment_means),
        "n_samples": len(df),
    }

    return result


def calculate_all_anova_measures(
    df: pd.DataFrame, value_col: str, segment_cols: Union[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate all ANOVA-based interestingness measures.

    Args:
        df: Raw DataFrame containing individual records (not aggregated)
        value_col: Column containing the metric to analyze (e.g., 'balance')
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        Dictionary containing all interestingness measures
    """
    return {
        "f_statistic": calculate_anova_f_statistic(df, value_col, segment_cols),
        "eta_squared": calculate_eta_squared(df, value_col, segment_cols),
        "omega_squared": calculate_omega_squared(df, value_col, segment_cols),
        "variance_ratio": calculate_between_within_variance_ratio(
            df, value_col, segment_cols
        ),
    }
