"""
Trend Metrics for Grouped Data Over Time

A collection of functions to evaluate how "interesting" a group's trend is compared
to the overall trend or other groups. These functions expect pre-aggregated pandas
DataFrames where each row represents a group-time combination with metrics like mean,
count, variance, etc.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats


def trend_slope_divergence(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate how much each group's trend slope diverges from the overall trend.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values
    count_col : str, optional
        Column containing the counts for weighted calculations

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and their slope divergence scores
    """
    if df.empty:
        return pd.DataFrame(
            columns=["group_id", "slope", "overall_slope", "divergence"]
        )

    # Ensure time_col is numeric for regression
    df = df.copy()

    # Check if time_col is already numeric
    if not np.issubdtype(df[time_col].dtype, np.number):
        # Try to convert if it's a timestamp
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Convert to numeric (days since min date)
            min_date = df[time_col].min()
            df["_numeric_time"] = (df[time_col] - min_date).dt.total_seconds() / (
                24 * 3600
            )
        else:
            # Try to extract numeric values or create ordinal encoding
            unique_times = df[time_col].unique()
            time_mapping = {t: i for i, t in enumerate(sorted(unique_times))}
            df["_numeric_time"] = df[time_col].map(time_mapping)
    else:
        df["_numeric_time"] = df[time_col]

    # Function to calculate slope for a group
    def calculate_slope(group_df):
        X = group_df["_numeric_time"].values.reshape(-1, 1)
        y = group_df[mean_col].values

        # Add a constant for intercept
        X_with_const = np.column_stack([np.ones(X.shape[0]), X])

        # Use weighted regression if count_col is provided
        if count_col is not None and count_col in group_df.columns:
            weights = group_df[count_col].values
            # Weighted least squares
            wx = X_with_const * np.sqrt(weights[:, np.newaxis])
            wy = y * np.sqrt(weights)
            beta = np.linalg.lstsq(wx, wy, rcond=None)[0]
        else:
            # Regular least squares
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]

        # Return the slope (beta[1])
        return beta[1]

    # Calculate overall slope
    overall_slope = calculate_slope(df)

    # Calculate slope for each group
    if isinstance(group_col, str):
        group_slopes = df.groupby(group_col).apply(calculate_slope).reset_index()
        group_slopes.columns = [group_col, "slope"]
    else:
        # Handle multi-column grouping
        group_slopes = df.groupby(group_col).apply(calculate_slope).reset_index()
        group_slopes.columns = group_col + ["slope"]

    # Calculate divergence from overall slope
    group_slopes["overall_slope"] = overall_slope
    group_slopes["divergence"] = np.abs(group_slopes["slope"] - overall_slope)

    return group_slopes


def trend_direction_difference(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Identify groups that trend in a different direction from the overall trend.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values
    count_col : str, optional
        Column containing the counts for weighted calculations

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and direction difference metrics
    """
    if df.empty:
        return pd.DataFrame(
            columns=["group_id", "slope", "overall_slope", "direction_difference"]
        )

    # Get slope divergence first to reuse calculations
    slope_df = trend_slope_divergence(df, group_col, time_col, mean_col, count_col)

    # Calculate direction difference
    # 1.0 means opposite direction, 0.0 means same direction
    slope_df["direction_difference"] = (
        slope_df["slope"] * slope_df["overall_slope"] < 0
    ).astype(float)

    # Add a "significance" measure - how large are the opposing slopes
    slope_df["direction_significance"] = 0.0

    # For groups moving in opposite direction, calculate significance
    opposite_mask = slope_df["direction_difference"] > 0
    if opposite_mask.any():
        # Significance = product of absolute slopes (both slopes must be meaningful)
        slope_df.loc[opposite_mask, "direction_significance"] = np.abs(
            slope_df.loc[opposite_mask, "slope"]
        ) * np.abs(slope_df.loc[opposite_mask, "overall_slope"])

    return slope_df


def trend_volatility(
    df: pd.DataFrame, group_col: Union[str, List[str]], time_col: str, mean_col: str
) -> pd.DataFrame:
    """
    Measure how volatile each group's trend is compared to others.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and volatility metrics
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "volatility", "relative_volatility"])

    # Calculate period-to-period changes for each group
    results = []

    # Function to calculate volatility for a group
    def calculate_volatility(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by=time_col)
        # Calculate period-to-period percent changes
        pct_changes = sorted_df[mean_col].pct_change().dropna()

        if len(pct_changes) < 2:
            # Not enough data points
            return {"volatility": np.nan}

        # Measure of volatility: standard deviation of percent changes
        return {"volatility": pct_changes.std()}

    # Calculate volatility for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            volatility = calculate_volatility(group_df)
            results.append({group_col: group, **volatility})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            volatility = calculate_volatility(group_df)
            results.append({**group_dict, **volatility})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Calculate overall volatility
    overall_volatility = np.nanmean(result_df["volatility"])

    # Calculate relative volatility (how much more/less volatile than average)
    result_df["relative_volatility"] = (
        result_df["volatility"] / overall_volatility if overall_volatility != 0 else 0.0
    )

    return result_df


def trend_consistency(
    df: pd.DataFrame, group_col: Union[str, List[str]], time_col: str, mean_col: str
) -> pd.DataFrame:
    """
    Measure how consistent each group's trend is (R² of linear fit).

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and consistency metrics (R²)
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "r_squared", "consistency_score"])

    # Ensure time_col is numeric for regression
    df = df.copy()

    # Check if time_col is already numeric
    if not np.issubdtype(df[time_col].dtype, np.number):
        # Try to convert if it's a timestamp
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Convert to numeric (days since min date)
            min_date = df[time_col].min()
            df["_numeric_time"] = (df[time_col] - min_date).dt.total_seconds() / (
                24 * 3600
            )
        else:
            # Try to extract numeric values or create ordinal encoding
            unique_times = df[time_col].unique()
            time_mapping = {t: i for i, t in enumerate(sorted(unique_times))}
            df["_numeric_time"] = df[time_col].map(time_mapping)
    else:
        df["_numeric_time"] = df[time_col]

    # Function to calculate R² for a group
    def calculate_r_squared(group_df):
        X = group_df["_numeric_time"].values
        y = group_df[mean_col].values

        if len(X) < 3:  # Need at least 3 points for meaningful regression
            return {"r_squared": np.nan}

        # Calculate correlation coefficient
        correlation, _ = stats.pearsonr(X, y)
        # R² is the square of the correlation coefficient
        r_squared = correlation**2

        return {"r_squared": r_squared}

    results = []

    # Calculate R² for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            stats_result = calculate_r_squared(group_df)
            results.append({group_col: group, **stats_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            stats_result = calculate_r_squared(group_df)
            results.append({**group_dict, **stats_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Add consistency score (1 = perfectly consistent, 0 = not consistent)
    result_df["consistency_score"] = result_df["r_squared"]

    return result_df


def acceleration_detection(
    df: pd.DataFrame, group_col: Union[str, List[str]], time_col: str, mean_col: str
) -> pd.DataFrame:
    """
    Detect which groups are accelerating or decelerating in their trend.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and acceleration metrics
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "acceleration", "is_accelerating"])

    # Ensure time_col is numeric
    df = df.copy()

    # Check if time_col is already numeric
    if not np.issubdtype(df[time_col].dtype, np.number):
        # Try to convert if it's a timestamp
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            # Convert to numeric (days since min date)
            min_date = df[time_col].min()
            df["_numeric_time"] = (df[time_col] - min_date).dt.total_seconds() / (
                24 * 3600
            )
        else:
            # Try to extract numeric values or create ordinal encoding
            unique_times = df[time_col].unique()
            time_mapping = {t: i for i, t in enumerate(sorted(unique_times))}
            df["_numeric_time"] = df[time_col].map(time_mapping)
    else:
        df["_numeric_time"] = df[time_col]

    # Function to fit a quadratic model and extract acceleration
    def calculate_acceleration(group_df):
        X = group_df["_numeric_time"].values
        y = group_df[mean_col].values

        if len(X) < 3:  # Need at least 3 points for quadratic fit
            return {"acceleration": 0.0, "is_accelerating": False}

        # Fit quadratic polynomial: y = a*x² + b*x + c
        coeffs = np.polyfit(X, y, 2)

        # The coefficient of x² (coeffs[0]) represents acceleration
        acceleration = coeffs[0]

        # Determine if accelerating (positive) or decelerating (negative)
        is_accelerating = acceleration > 0

        return {"acceleration": acceleration, "is_accelerating": is_accelerating}

    results = []

    # Calculate acceleration for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            accel_result = calculate_acceleration(group_df)
            results.append({group_col: group, **accel_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            accel_result = calculate_acceleration(group_df)
            results.append({**group_dict, **accel_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Add relative acceleration (normalized)
    if not result_df.empty and not result_df["acceleration"].isna().all():
        # Normalize acceleration to [-1, 1] range for easier comparison
        max_abs_accel = np.nanmax(np.abs(result_df["acceleration"]))

        if max_abs_accel > 0:
            result_df["relative_acceleration"] = (
                result_df["acceleration"] / max_abs_accel
            )
        else:
            result_df["relative_acceleration"] = 0.0
    else:
        result_df["relative_acceleration"] = 0.0

    return result_df


def trend_break_detection(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Detect sudden changes or breaks in the trend for each group.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values
    min_periods : int, default=3
        Minimum number of periods before and after to consider a break

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and trend break metrics
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "has_trend_break", "break_magnitude"])

    # Function to detect trend breaks for a group
    def detect_breaks(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by=time_col)
        times = sorted_df[time_col].values
        values = sorted_df[mean_col].values

        if len(times) < (2 * min_periods):
            # Not enough data for meaningful break detection
            return {
                "has_trend_break": False,
                "break_magnitude": 0.0,
                "break_time": None,
            }

        # Look for the point that maximizes the difference in slopes
        max_break_magnitude = 0
        break_point = None

        for i in range(min_periods, len(times) - min_periods):
            # Calculate slope before this point
            before_X = np.arange(i)
            before_y = values[:i]
            before_slope = np.polyfit(before_X, before_y, 1)[0]

            # Calculate slope after this point
            after_X = np.arange(i, len(times))
            after_y = values[i:]
            after_slope = np.polyfit(after_X - i, after_y, 1)[0]

            # Calculate magnitude of the break (absolute difference in slopes)
            break_magnitude = abs(after_slope - before_slope)

            if break_magnitude > max_break_magnitude:
                max_break_magnitude = break_magnitude
                break_point = times[i]

        # Determine if there's a significant break
        # This threshold can be adjusted based on the data
        significance_threshold = np.std(values) * 0.2  # 20% of standard deviation

        has_trend_break = max_break_magnitude > significance_threshold

        return {
            "has_trend_break": has_trend_break,
            "break_magnitude": max_break_magnitude,
            "break_time": break_point,
        }

    results = []

    # Calculate trend breaks for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            break_result = detect_breaks(group_df)
            results.append({group_col: group, **break_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            break_result = detect_breaks(group_df)
            results.append({**group_dict, **break_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Add normalized break magnitude
    if not result_df.empty and not result_df["break_magnitude"].isna().all():
        max_magnitude = np.nanmax(result_df["break_magnitude"])

        if max_magnitude > 0:
            result_df["normalized_break"] = result_df["break_magnitude"] / max_magnitude
        else:
            result_df["normalized_break"] = 0.0
    else:
        result_df["normalized_break"] = 0.0

    return result_df


def trend_divergence_score(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calculate an overall trend divergence score combining multiple metrics.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_col : str or list of str
        Column(s) identifying the groups
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values
    count_col : str, optional
        Column containing the counts for weighted calculations

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and combined trend divergence scores
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "trend_divergence_score"])

    # Calculate individual metrics
    slope_df = trend_slope_divergence(df, group_col, time_col, mean_col, count_col)
    direction_df = trend_direction_difference(
        df, group_col, time_col, mean_col, count_col
    )
    consistency_df = trend_consistency(df, group_col, time_col, mean_col)
    acceleration_df = acceleration_detection(df, group_col, time_col, mean_col)

    # Get group identifier columns
    if isinstance(group_col, str):
        id_cols = [group_col]
    else:
        id_cols = group_col

    # Merge all metrics
    # Start with slope_df
    result_df = slope_df.copy()

    # Add direction difference
    direction_cols = ["direction_difference", "direction_significance"]
    result_df = pd.merge(result_df, direction_df[id_cols + direction_cols], on=id_cols)

    # Add consistency
    consistency_cols = ["consistency_score"]
    result_df = pd.merge(
        result_df, consistency_df[id_cols + consistency_cols], on=id_cols
    )

    # Add acceleration
    acceleration_cols = ["relative_acceleration"]
    result_df = pd.merge(
        result_df, acceleration_df[id_cols + acceleration_cols], on=id_cols
    )

    # Calculate combined score
    # Components:
    # 1. Slope divergence (normalized)
    # 2. Direction difference (0 or 1) weighted by significance
    # 3. Inconsistency (1 - consistency_score)
    # 4. Absolute acceleration (how much curvature)

    # Normalize slope divergence
    max_divergence = result_df["divergence"].max()
    if max_divergence > 0:
        result_df["normalized_divergence"] = result_df["divergence"] / max_divergence
    else:
        result_df["normalized_divergence"] = 0.0

    # Calculate combined score
    # Weights can be adjusted based on which aspects are most important
    weights = {
        "normalized_divergence": 0.4,  # Different slope
        "direction_difference": 0.3,  # Opposite direction
        "inconsistency": 0.1,  # Non-linear pattern
        "abs_acceleration": 0.2,  # Acceleration/deceleration
    }

    result_df["inconsistency"] = 1 - result_df["consistency_score"]
    result_df["abs_acceleration"] = np.abs(result_df["relative_acceleration"])

    # Combined score
    result_df["trend_divergence_score"] = (
        weights["normalized_divergence"] * result_df["normalized_divergence"]
        + weights["direction_difference"]
        * result_df["direction_difference"]
        * np.sqrt(
            result_df["direction_significance"] + 0.01
        )  # Add small constant to avoid zeros
        + weights["inconsistency"] * result_df["inconsistency"]
        + weights["abs_acceleration"] * result_df["abs_acceleration"]
    )

    # Normalize to 0-1 range
    max_score = result_df["trend_divergence_score"].max()
    if max_score > 0:
        result_df["trend_divergence_score"] = (
            result_df["trend_divergence_score"] / max_score
        )

    return result_df
