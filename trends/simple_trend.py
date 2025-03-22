"""
Functions for simple trend analysis on time series data.
This module provides tools for detecting and comparing linear trends across segments.
"""

from typing import List, Union

import numpy as np
import pandas as pd
from scipy import stats


def calculate_segment_slopes(
    df: pd.DataFrame, time_col: str, value_col: str, segment_cols: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate linear regression slopes for each segment's time series.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values (will be converted to numeric for regression)
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        DataFrame with slope, intercept, and related statistics for each segment
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist in DataFrame
    missing_cols = [
        col for col in segment_cols + [time_col, value_col] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Convert time column to numeric if it's not already
    # This allows regression on dates by converting to ordinal values
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        # Convert dates to ordinal values (days since 1970-01-01)
        df = df.copy()
        df["_time_numeric"] = (df[time_col] - pd.Timestamp("1970-01-01")).dt.days
        time_numeric = "_time_numeric"
    else:
        # Try to convert to numeric directly
        try:
            df["_time_numeric"] = pd.to_numeric(df[time_col])
            time_numeric = "_time_numeric"
        except:
            # If conversion fails, use as-is and assume it's already numeric
            time_numeric = time_col

    # Group by segment(s)
    if segment_cols:
        grouped = df.groupby(segment_cols)
    else:
        # If no segment columns, treat entire dataset as one segment
        # Add a dummy column for consistency in result structure
        df["_all"] = "All"
        grouped = df.groupby("_all")

    # Calculate slopes for each segment
    results = []

    for segment_name, segment_df in grouped:
        # Skip segments with too few points for regression
        if len(segment_df) < 2:
            continue

        # Prepare segment name as tuple for consistency
        if not isinstance(segment_name, tuple):
            segment_name = (segment_name,)

        # Get x and y values for regression
        x = segment_df[time_numeric].values
        y = segment_df[value_col].values

        # Skip if all values are the same (to avoid division by zero)
        if np.all(y == y[0]) or np.all(x == x[0]):
            # Add row with zero slope
            row = {col: val for col, val in zip(segment_cols, segment_name)}
            row.update(
                {
                    "slope": 0,
                    "intercept": y[0] if y.size > 0 else np.nan,
                    "r_value": 0,
                    "p_value": 1.0,
                    "std_err": np.nan,
                    "n_points": len(x),
                }
            )
            results.append(row)
            continue

        # Calculate linear regression
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Create result row with segment identifiers
            row = {col: val for col, val in zip(segment_cols, segment_name)}
            row.update(
                {
                    "slope": slope,
                    "intercept": intercept,
                    "r_value": r_value,
                    "p_value": p_value,
                    "std_err": std_err,
                    "n_points": len(x),
                }
            )
            results.append(row)
        except Exception as e:
            # Handle regression errors
            row = {col: val for col, val in zip(segment_cols, segment_name)}
            row.update(
                {
                    "slope": np.nan,
                    "intercept": np.nan,
                    "r_value": np.nan,
                    "p_value": np.nan,
                    "std_err": np.nan,
                    "n_points": len(x),
                    "error": str(e),
                }
            )
            results.append(row)

    # Convert results to DataFrame
    result_df = pd.DataFrame(results)

    # Add trend direction
    if "slope" in result_df.columns:
        result_df["trend_direction"] = result_df["slope"].apply(
            lambda s: "increasing" if s > 0 else ("decreasing" if s < 0 else "flat")
        )

    # Add significance indicator
    if "p_value" in result_df.columns:
        result_df["significant"] = result_df["p_value"] < 0.05

    return result_df


def compare_to_overall_trend(
    df: pd.DataFrame, time_col: str, value_col: str, segment_cols: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Compare segment trends to the overall trend in the data.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        DataFrame with comparison of segment slopes to overall slope
    """
    # Calculate slopes for each segment
    segment_slopes = calculate_segment_slopes(df, time_col, value_col, segment_cols)

    # Calculate overall slope (ignoring segments)
    overall_slope_df = calculate_segment_slopes(df, time_col, value_col, None)
    overall_slope = (
        overall_slope_df["slope"].iloc[0] if len(overall_slope_df) > 0 else 0
    )
    overall_p_value = (
        overall_slope_df["p_value"].iloc[0] if len(overall_slope_df) > 0 else 1.0
    )

    # Compare each segment's slope to the overall slope
    segment_slopes["overall_slope"] = overall_slope
    segment_slopes["slope_difference"] = segment_slopes["slope"] - overall_slope
    segment_slopes["slope_diff_ratio"] = (
        segment_slopes["slope"] / overall_slope if overall_slope != 0 else np.nan
    )

    # Flag segments with trends different from overall
    segment_slopes["differs_from_overall"] = (
        (segment_slopes["slope"] > 0 and overall_slope < 0)
        or (segment_slopes["slope"] < 0 and overall_slope > 0)
        or (
            abs(segment_slopes["slope_diff_ratio"]) > 2
        )  # Trend at least twice as strong
    )

    # Add information about overall trend
    segment_slopes["overall_trend_direction"] = (
        "increasing"
        if overall_slope > 0
        else ("decreasing" if overall_slope < 0 else "flat")
    )
    segment_slopes["overall_trend_significant"] = overall_p_value < 0.05

    return segment_slopes


def identify_diverging_trends(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    segment_cols: Union[str, List[str]],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Identify segments with trends that diverge significantly from the overall trend.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze
        threshold: Threshold for considering a trend as diverging (as a fraction of overall trend)

    Returns:
        DataFrame with diverging segments and related statistics
    """
    # Get segment trends compared to overall
    comparison = compare_to_overall_trend(df, time_col, value_col, segment_cols)

    # Identify diverging segments
    # Criteria:
    # 1. Opposite direction from overall, or
    # 2. Significantly stronger/weaker than overall (by threshold)
    diverging = comparison[
        (comparison["differs_from_overall"])
        | (abs(comparison["slope_diff_ratio"] - 1) > threshold)
    ]

    # Sort by degree of divergence
    if len(diverging) > 0:
        diverging = diverging.sort_values(
            by="slope_difference", key=abs, ascending=False
        )

    # Add a divergence metric (how much the segment diverges from overall)
    # Normalized to make it interpretable
    if "slope_difference" in diverging.columns:
        std_slope_diff = diverging["slope_difference"].std()
        mean_value = df[value_col].mean()

        if std_slope_diff > 0 and mean_value > 0:
            diverging["divergence_metric"] = (
                abs(diverging["slope_difference"]) / mean_value
            )
        else:
            diverging["divergence_metric"] = abs(diverging["slope_difference"])

    return diverging


def calculate_moving_average(
    df: pd.DataFrame, time_col: str, value_col: str, window: int = 3
) -> pd.DataFrame:
    """
    Calculate moving average of values to smooth out noise in time series.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        window: Window size for moving average

    Returns:
        DataFrame with original data and moving average column added
    """
    # Check if columns exist
    if time_col not in df.columns or value_col not in df.columns:
        raise ValueError("Column not found in DataFrame")

    # Ensure DataFrame is sorted by time
    df_sorted = df.sort_values(by=time_col).copy()

    # Calculate moving average
    df_sorted[f"{value_col}_ma{window}"] = (
        df_sorted[value_col].rolling(window=window, min_periods=1).mean()
    )

    return df_sorted


def detect_trend_changes(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    segment_cols: Union[str, List[str]],
    window_size: int = 3,
) -> pd.DataFrame:
    """
    Detect changes in trend direction for each segment over time.
    Uses a moving window approach to calculate local slopes.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze
        window_size: Size of the rolling window for local trend calculation

    Returns:
        DataFrame with trend change points for each segment
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist
    missing_cols = [
        col for col in segment_cols + [time_col, value_col] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Convert time column to numeric if it's not already
    if pd.api.types.is_datetime64_any_dtype(df[time_col]):
        # Convert dates to ordinal values (days since 1970-01-01)
        df = df.copy()
        df["_time_numeric"] = (df[time_col] - pd.Timestamp("1970-01-01")).dt.days
        time_numeric = "_time_numeric"
    else:
        # Try to convert to numeric directly
        try:
            df["_time_numeric"] = pd.to_numeric(df[time_col])
            time_numeric = "_time_numeric"
        except:
            # If conversion fails, use as-is and assume it's already numeric
            time_numeric = time_col

    # Group by segment(s)
    grouped = df.groupby(segment_cols)

    # Store results for each segment
    all_changes = []

    for segment_name, segment_df in grouped:
        # Skip segments with too few points
        if len(segment_df) < window_size + 1:
            continue

        # Ensure segment_name is a tuple
        if not isinstance(segment_name, tuple):
            segment_name = (segment_name,)

        # Sort by time
        segment_df = segment_df.sort_values(by=time_numeric)

        # Calculate rolling slopes
        slopes = []
        dates = []

        # Use a rolling window to calculate local slopes
        for i in range(len(segment_df) - window_size + 1):
            window = segment_df.iloc[i : i + window_size]
            x = window[time_numeric].values
            y = window[value_col].values

            # Calculate slope for this window
            if len(x) >= 2 and not (np.all(y == y[0]) or np.all(x == x[0])):
                try:
                    slope, _, _, p_value, _ = stats.linregress(x, y)
                    slopes.append(slope)
                    dates.append(window[time_col].iloc[-1])  # Use last date in window
                except:
                    slopes.append(np.nan)
                    dates.append(window[time_col].iloc[-1])

        # Detect changes in slope direction
        changes = []
        if len(slopes) > 1:
            for i in range(1, len(slopes)):
                # Detect trend direction change (positive to negative or vice versa)
                if (slopes[i - 1] > 0 and slopes[i] < 0) or (
                    slopes[i - 1] < 0 and slopes[i] > 0
                ):
                    change = {col: val for col, val in zip(segment_cols, segment_name)}
                    change.update(
                        {
                            "change_point": dates[i],
                            "previous_slope": slopes[i - 1],
                            "new_slope": slopes[i],
                            "direction_change": (
                                "positive to negative"
                                if slopes[i - 1] > 0
                                else "negative to positive"
                            ),
                            "change_magnitude": abs(slopes[i] - slopes[i - 1]),
                        }
                    )
                    changes.append(change)

        all_changes.extend(changes)

    # Return DataFrame of trend changes
    return pd.DataFrame(all_changes)


def calculate_growth_rates(
    df: pd.DataFrame, time_col: str, value_col: str, segment_cols: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Calculate growth rates for each segment over time.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze

    Returns:
        DataFrame with growth rates for each segment
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist
    missing_cols = [
        col for col in segment_cols + [time_col, value_col] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Group by segment(s)
    grouped = df.groupby(segment_cols)

    # Store results
    growth_rates = []

    for segment_name, segment_df in grouped:
        # Skip segments with too few points
        if len(segment_df) < 2:
            continue

        # Ensure segment_name is a tuple
        if not isinstance(segment_name, tuple):
            segment_name = (segment_name,)

        # Sort by time
        segment_df = segment_df.sort_values(by=time_col)

        # Calculate period-to-period growth rates
        segment_df = segment_df.copy()
        segment_df["prev_value"] = segment_df[value_col].shift(1)
        segment_df["growth_rate"] = (
            segment_df[value_col] - segment_df["prev_value"]
        ) / segment_df["prev_value"]
        segment_df["growth_rate_pct"] = segment_df["growth_rate"] * 100

        # Skip first row (which has no growth rate)
        segment_df = segment_df.dropna(subset=["growth_rate"])

        # Calculate average growth rate
        avg_growth = segment_df["growth_rate"].mean()
        avg_growth_pct = avg_growth * 100

        # Create result row
        row = {col: val for col, val in zip(segment_cols, segment_name)}
        row.update(
            {
                "avg_growth_rate": avg_growth,
                "avg_growth_rate_pct": avg_growth_pct,
                "min_growth_rate_pct": segment_df["growth_rate_pct"].min(),
                "max_growth_rate_pct": segment_df["growth_rate_pct"].max(),
                "growth_volatility": segment_df["growth_rate"].std(),
                "n_periods": len(segment_df),
                "n_positive_growth": sum(segment_df["growth_rate"] > 0),
                "n_negative_growth": sum(segment_df["growth_rate"] < 0),
            }
        )
        growth_rates.append(row)

    # Convert results to DataFrame
    result_df = pd.DataFrame(growth_rates)

    # Add trend direction based on average growth rate
    if "avg_growth_rate" in result_df.columns:
        result_df["trend_direction"] = result_df["avg_growth_rate"].apply(
            lambda g: "increasing" if g > 0 else ("decreasing" if g < 0 else "flat")
        )

    return result_df


def detect_step_changes(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    segment_cols: Union[str, List[str]],
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect sudden step changes in values between consecutive time periods.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze
        threshold: Z-score threshold for considering a change as significant

    Returns:
        DataFrame with detected step changes for each segment
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist
    missing_cols = [
        col for col in segment_cols + [time_col, value_col] if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Group by segment(s)
    grouped = df.groupby(segment_cols)

    # Store results
    all_steps = []

    for segment_name, segment_df in grouped:
        # Skip segments with too few points
        if len(segment_df) < 3:  # Need at least 3 points to establish a baseline
            continue

        # Ensure segment_name is a tuple
        if not isinstance(segment_name, tuple):
            segment_name = (segment_name,)

        # Sort by time
        segment_df = segment_df.sort_values(by=time_col)

        # Calculate period-to-period changes
        segment_df = segment_df.copy()
        segment_df["prev_value"] = segment_df[value_col].shift(1)
        segment_df["change"] = segment_df[value_col] - segment_df["prev_value"]
        segment_df["pct_change"] = segment_df["change"] / segment_df["prev_value"] * 100

        # Skip first row (which has no change)
        segment_df = segment_df.dropna(subset=["change"])

        # Calculate mean and standard deviation of changes
        mean_change = segment_df["change"].mean()
        std_change = segment_df["change"].std()

        # If std is zero or NaN, skip this segment
        if std_change == 0 or np.isnan(std_change):
            continue

        # Calculate z-scores for each change
        segment_df["change_zscore"] = (segment_df["change"] - mean_change) / std_change

        # Identify significant step changes
        step_changes = segment_df[abs(segment_df["change_zscore"]) > threshold].copy()

        for _, row in step_changes.iterrows():
            step = {col: val for col, val in zip(segment_cols, segment_name)}
            step.update(
                {
                    "time_point": row[time_col],
                    "previous_value": row["prev_value"],
                    "new_value": row[value_col],
                    "change": row["change"],
                    "pct_change": row["pct_change"],
                    "z_score": row["change_zscore"],
                    "direction": "increase" if row["change"] > 0 else "decrease",
                }
            )
            all_steps.append(step)

    # Convert results to DataFrame and sort by absolute z-score
    if all_steps:
        result_df = pd.DataFrame(all_steps)
        result_df["abs_z_score"] = abs(result_df["z_score"])
        result_df = result_df.sort_values("abs_z_score", ascending=False)
        result_df = result_df.drop("abs_z_score", axis=1)
        return result_df
    else:
        # Return empty DataFrame with appropriate columns
        return pd.DataFrame(
            columns=[
                *segment_cols,
                "time_point",
                "previous_value",
                "new_value",
                "change",
                "pct_change",
                "z_score",
                "direction",
            ]
        )
