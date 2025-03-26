"""
Advanced Trend Metrics for Grouped Data Over Time

Advanced statistical metrics to evaluate how interesting or unusual
a group's trend is compared to the overall trend or other groups.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose


def seasonal_pattern_difference(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    period: int,
) -> pd.DataFrame:
    """
    Measure how different a group's seasonal pattern is from the overall pattern.

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
    period : int
        Number of time periods in a seasonal cycle (e.g., 12 for monthly data with yearly seasonality)

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and seasonal pattern difference metrics
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "seasonal_difference"])

    # Function to extract seasonal component
    def extract_seasonal_component(time_series, period):
        # time_series should be a pandas Series with time index
        if len(time_series) < 2 * period:
            # Not enough data for seasonal decomposition
            return pd.Series(np.zeros(len(time_series)), index=time_series.index)

        try:
            # Get seasonal component
            result = seasonal_decompose(time_series, period=period, model="additive")
            seasonal = result.seasonal
            return seasonal
        except:
            # If decomposition fails, return zeros
            return pd.Series(np.zeros(len(time_series)), index=time_series.index)

    # Prepare result container
    results = []

    # Calculate overall seasonal pattern
    # First, create time-indexed series of overall mean
    overall_means_df = df.groupby(time_col)[mean_col].mean().reset_index()

    # Convert to a Series with time_col as index for seasonal_decompose
    overall_means = pd.Series(
        overall_means_df[mean_col].values, index=overall_means_df[time_col]
    )

    # Extract overall seasonal component
    overall_seasonal = extract_seasonal_component(overall_means, period)

    # Function to calculate correlation with overall seasonal pattern
    def calculate_seasonal_difference(group_df):
        # Group data by time and calculate mean
        group_means_df = group_df.groupby(time_col)[mean_col].mean().reset_index()

        # Convert to a Series with time_col as index
        group_means = pd.Series(
            group_means_df[mean_col].values, index=group_means_df[time_col]
        )

        # Extract seasonal component for this group
        if len(group_means) < 2 * period:
            return {"seasonal_difference": 1.0, "has_seasonal_pattern": False}

        group_seasonal = extract_seasonal_component(group_means, period)

        # Calculate correlation with overall seasonal pattern
        # Match up the indexes
        common_index = set(overall_seasonal.index).intersection(
            set(group_seasonal.index)
        )
        if len(common_index) < 3:
            return {"seasonal_difference": 1.0, "has_seasonal_pattern": False}

        # Convert to lists for consistent ordering
        common_index = sorted(list(common_index))

        overall_seasonal_aligned = overall_seasonal.loc[common_index]
        group_seasonal_aligned = group_seasonal.loc[common_index]

        # Calculate correlation
        correlation, _ = stats.pearsonr(
            overall_seasonal_aligned, group_seasonal_aligned
        )

        # Transform correlation to difference score (1 = completely different, 0 = identical)
        seasonal_difference = (1 - correlation) / 2

        # Check if the group has a meaningful seasonal pattern
        # Use variance of the seasonal component compared to the total variance
        group_seasonal_variance = np.var(group_seasonal_aligned)
        group_total_variance = np.var(group_means.loc[common_index])

        has_seasonal_pattern = (
            group_seasonal_variance
            > 0.1 * group_total_variance  # Seasonal component is at least 10% of total
            and len(common_index) >= period  # Have at least one full cycle
        )

        return {
            "seasonal_difference": seasonal_difference,
            "has_seasonal_pattern": has_seasonal_pattern,
        }

    # Calculate seasonal differences for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            seasonal_result = calculate_seasonal_difference(group_df)
            results.append({group_col: group, **seasonal_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            seasonal_result = calculate_seasonal_difference(group_df)
            results.append({**group_dict, **seasonal_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def trend_stationarity_test(
    df: pd.DataFrame, group_col: Union[str, List[str]], time_col: str, mean_col: str
) -> pd.DataFrame:
    """
    Test if a group's time series is stationary or has a clear trend.

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
        DataFrame with group identifiers and stationarity test results
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "is_stationary", "p_value"])

    # Import adfuller here for test
    from statsmodels.tsa.stattools import adfuller

    # Function to test stationarity
    def test_stationarity(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by=time_col)
        values = sorted_df[mean_col].values

        if len(values) < 8:  # Need reasonable number of points for the test
            return {"is_stationary": np.nan, "p_value": np.nan}

        try:
            # Augmented Dickey-Fuller test
            result = adfuller(values)
            p_value = result[1]

            # If p-value < 0.05, reject null hypothesis => stationary
            is_stationary = p_value < 0.05

            return {"is_stationary": is_stationary, "p_value": p_value}
        except:
            # If test fails, return NaN
            return {"is_stationary": np.nan, "p_value": np.nan}

    results = []

    # Test stationarity for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            test_result = test_stationarity(group_df)
            results.append({group_col: group, **test_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            test_result = test_stationarity(group_df)
            results.append({**group_dict, **test_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Calculate stationarity_difference from overall
    # If most groups are stationary (or not), identify the outliers
    if not result_df.empty and not result_df["is_stationary"].isna().all():
        # Calculate proportion of stationary groups
        prop_stationary = result_df["is_stationary"].mean()

        # If most are stationary, then non-stationary groups are interesting
        # If most are non-stationary, then stationary groups are interesting
        result_df["stationarity_difference"] = (
            (prop_stationary >= 0.5) & (~result_df["is_stationary"])
            | (prop_stationary < 0.5) & (result_df["is_stationary"])
        ).astype(float)
    else:
        result_df["stationarity_difference"] = 0.0

    return result_df


def autocorrelation_pattern(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    max_lag: int = 5,
) -> pd.DataFrame:
    """
    Analyze autocorrelation patterns for each group and compare to the overall pattern.

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
    max_lag : int, default=5
        Maximum lag to consider for autocorrelation

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and autocorrelation pattern differences
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "autocorrelation_difference"])

    # Import acf function
    from statsmodels.tsa.stattools import acf

    # Function to calculate autocorrelation
    def calculate_acf(series, max_lag):
        if len(series) <= max_lag:
            return np.zeros(max_lag)

        try:
            # Calculate autocorrelation up to max_lag
            acf_values = acf(series, nlags=max_lag, fft=False)
            # Skip lag 0 (always 1.0)
            return acf_values[1:]
        except:
            # If calculation fails, return zeros
            return np.zeros(max_lag)

    # Calculate overall autocorrelation pattern
    # Group by time and calculate overall mean - with reset_index() to avoid index assumptions
    overall_means = df.groupby(time_col)[mean_col].mean().reset_index()[mean_col].values
    overall_acf = calculate_acf(overall_means, max_lag)

    # Function to compare autocorrelation patterns
    def compare_autocorrelation(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by=time_col)
        values = sorted_df[mean_col].values

        if len(values) <= max_lag:
            return {"autocorrelation_difference": 1.0}

        # Calculate autocorrelation for this group
        group_acf = calculate_acf(values, max_lag)

        # Calculate difference with overall pattern (Euclidean distance)
        difference = np.sqrt(np.sum((group_acf - overall_acf) ** 2)) / max_lag

        # Normalize to [0, 1] range
        normalized_difference = min(1.0, difference)

        return {"autocorrelation_difference": normalized_difference}

    results = []

    # Compare autocorrelation patterns for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            acf_result = compare_autocorrelation(group_df)
            results.append({group_col: group, **acf_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            acf_result = compare_autocorrelation(group_df)
            results.append({**group_dict, **acf_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def outlier_period_detection(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    threshold: float = 2.0,
) -> pd.DataFrame:
    """
    Detect time periods where a group's value deviates significantly from its trend line.

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
    threshold : float, default=2.0
        Z-score threshold for considering a period an outlier

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and outlier period metrics
    """
    if df.empty:
        return pd.DataFrame(
            columns=["group_id", "has_outlier_periods", "outlier_ratio"]
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

    # Function to detect outlier periods
    def detect_outliers(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by="_numeric_time")
        X = sorted_df["_numeric_time"].values.reshape(-1, 1)
        y = sorted_df[mean_col].values

        if len(X) < 4:  # Need some minimum points
            return {
                "has_outlier_periods": False,
                "outlier_ratio": 0.0,
                "outlier_periods": [],
            }

        # Fit a trend line
        X_with_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_with_const).fit()
            y_pred = model.predict(X_with_const)

            # Calculate residuals
            residuals = y - y_pred

            # Identify outliers (residuals beyond threshold standard deviations)
            std_residuals = np.std(residuals)
            if std_residuals == 0:
                return {
                    "has_outlier_periods": False,
                    "outlier_ratio": 0.0,
                    "outlier_periods": [],
                }

            z_scores = residuals / std_residuals
            outliers = np.abs(z_scores) > threshold

            # Extract outlier periods
            outlier_periods = sorted_df[time_col].iloc[outliers].tolist()

            # Calculate outlier ratio
            outlier_ratio = sum(outliers) / len(outliers)

            return {
                "has_outlier_periods": any(outliers),
                "outlier_ratio": outlier_ratio,
                "outlier_periods": outlier_periods,
            }
        except:
            # If model fitting fails, return no outliers
            return {
                "has_outlier_periods": False,
                "outlier_ratio": 0.0,
                "outlier_periods": [],
            }

    results = []

    # Detect outliers for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            outlier_result = detect_outliers(group_df)
            results.append({group_col: group, **outlier_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            outlier_result = detect_outliers(group_df)
            results.append({**group_dict, **outlier_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def periodicity_detection(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    max_period: Optional[int] = None,
) -> pd.DataFrame:
    """
    Detect if a group has a regular cycle or periodic pattern different from others.

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
    max_period : int, optional
        Maximum period length to check for

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and periodicity metrics
    """
    if df.empty:
        return pd.DataFrame(columns=["group_id", "has_periodicity", "period_length"])

    # Function to detect periodicity using FFT
    def detect_periodicity(group_df):
        # Sort by time
        sorted_df = group_df.sort_values(by=time_col)
        values = sorted_df[mean_col].values

        if len(values) < 8:  # Need some minimum number of points
            return {
                "has_periodicity": False,
                "period_length": 0,
                "periodicity_strength": 0.0,
            }

        try:
            # Detrend the series (remove linear trend)
            X = np.arange(len(values))
            coef = np.polyfit(X, values, 1)
            trend = coef[0] * X + coef[1]
            detrended = values - trend

            # Calculate FFT
            fft_values = np.abs(np.fft.rfft(detrended))

            # Get frequencies
            freqs = np.fft.rfftfreq(len(detrended))

            # Limit to max_period if specified
            if max_period is not None and max_period < len(detrended) // 2:
                max_freq = 1.0 / max_period
                mask = freqs > max_freq
                freqs = freqs[mask]
                fft_values = fft_values[mask]

            # Skip DC component (0 frequency)
            if len(freqs) > 1:
                freqs = freqs[1:]
                fft_values = fft_values[1:]
            else:
                return {
                    "has_periodicity": False,
                    "period_length": 0,
                    "periodicity_strength": 0.0,
                }

            # Find dominant frequency
            dominant_idx = np.argmax(fft_values)
            dominant_freq = freqs[dominant_idx]

            # Calculate period length
            if dominant_freq > 0:
                period_length = int(round(1.0 / dominant_freq))
            else:
                period_length = 0

            # Calculate strength of periodicity
            # Compare dominant frequency amplitude to average
            if len(fft_values) > 1:
                avg_amplitude = np.mean(fft_values)
                periodicity_strength = (
                    fft_values[dominant_idx] / avg_amplitude
                    if avg_amplitude > 0
                    else 0.0
                )
            else:
                periodicity_strength = 0.0

            # Check if periodicity is significant
            has_periodicity = (
                period_length > 1  # At least 2 time periods
                and period_length < len(values) // 2  # Can observe at least 2 cycles
                and periodicity_strength
                > 2.0  # Dominant frequency is at least twice the average
            )

            return {
                "has_periodicity": has_periodicity,
                "period_length": period_length,
                "periodicity_strength": periodicity_strength,
            }
        except:
            # If detection fails, return no periodicity
            return {
                "has_periodicity": False,
                "period_length": 0,
                "periodicity_strength": 0.0,
            }

    results = []

    # Detect periodicity for each group
    if isinstance(group_col, str):
        # Single grouping column
        groups = df[group_col].unique()
        for group in groups:
            group_df = df[df[group_col] == group]
            periodicity_result = detect_periodicity(group_df)
            results.append({group_col: group, **periodicity_result})
    else:
        # Multi-column grouping
        unique_groups = df[group_col].drop_duplicates().to_dict("records")
        for group_dict in unique_groups:
            # Create filter for this group combination
            mask = pd.Series(True, index=df.index)
            for col, val in group_dict.items():
                mask &= df[col] == val

            group_df = df[mask]
            periodicity_result = detect_periodicity(group_df)
            results.append({**group_dict, **periodicity_result})

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Calculate periodicity difference score
    # If a group has periodicity different from most other groups, that's interesting
    if not result_df.empty and not result_df["has_periodicity"].isna().all():
        # Calculate proportion of groups with periodicity
        prop_periodic = result_df["has_periodicity"].mean()

        # If most groups have periodicity, then non-periodic groups are interesting
        # If most groups don't have periodicity, then periodic groups are interesting
        result_df["periodicity_difference"] = (
            (prop_periodic >= 0.5) & (~result_df["has_periodicity"])
            | (prop_periodic < 0.5) & (result_df["has_periodicity"])
        ).astype(float)

        # Scale by strength of periodicity if present
        mask = result_df["has_periodicity"]
        if mask.any():
            max_strength = result_df.loc[mask, "periodicity_strength"].max()
            if max_strength > 0:
                result_df.loc[mask, "periodicity_difference"] *= (
                    result_df.loc[mask, "periodicity_strength"] / max_strength
                )
    else:
        result_df["periodicity_difference"] = 0.0

    return result_df


def advanced_trend_divergence_score(
    df, group_col, time_col, mean_col, count_col=None, period=None
):
    """
    Calculate a comprehensive trend divergence score that includes both basic and advanced metrics.

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
    period : int, optional
        Number of time periods in a seasonal cycle, if known

    Returns:
    --------
    pandas DataFrame
        DataFrame with group identifiers and advanced trend divergence scores
    """
    from .trend_metrics import trend_divergence_score

    if df.empty:
        return pd.DataFrame(columns=["group_id", "advanced_trend_divergence_score"])

    # Get basic trend divergence score
    basic_scores = trend_divergence_score(df, group_col, time_col, mean_col, count_col)

    # Get advanced metrics

    # Autocorrelation pattern difference
    acf_diff = autocorrelation_pattern(df, group_col, time_col, mean_col)

    # Outlier periods
    outlier_df = outlier_period_detection(df, group_col, time_col, mean_col)

    # Periodicity detection
    periodicity_df = periodicity_detection(df, group_col, time_col, mean_col)

    # Stationarity test
    stationarity_df = trend_stationarity_test(df, group_col, time_col, mean_col)

    # Get group identifier columns
    if isinstance(group_col, str):
        id_cols = [group_col]
    else:
        id_cols = group_col

    # Seasonal pattern differences, if period is provided
    if period is not None and period > 1:
        seasonal_df = seasonal_pattern_difference(
            df, group_col, time_col, mean_col, period
        )
    else:
        # Create dummy DataFrame with zeros that matches the structure of basic_scores
        seasonal_df = basic_scores[id_cols].copy()
        seasonal_df["seasonal_difference"] = 0.0

    # Merge all metrics
    # Start with basic scores
    result_df = basic_scores.copy()

    # Add autocorrelation
    acf_cols = ["autocorrelation_difference"]
    result_df = pd.merge(result_df, acf_diff[id_cols + acf_cols], on=id_cols)

    # Add outlier periods
    outlier_cols = ["outlier_ratio"]
    result_df = pd.merge(result_df, outlier_df[id_cols + outlier_cols], on=id_cols)

    # Add periodicity
    periodicity_cols = ["periodicity_difference"]
    result_df = pd.merge(
        result_df, periodicity_df[id_cols + periodicity_cols], on=id_cols
    )

    # Add stationarity
    stationarity_cols = ["stationarity_difference"]
    result_df = pd.merge(
        result_df, stationarity_df[id_cols + stationarity_cols], on=id_cols
    )

    # Add seasonal pattern
    seasonal_cols = ["seasonal_difference"]
    result_df = pd.merge(result_df, seasonal_df[id_cols + seasonal_cols], on=id_cols)

    # Calculate combined score
    # Weights can be adjusted based on which aspects are most important
    weights = {
        "trend_divergence_score": 0.4,  # Basic trend metrics
        "autocorrelation_difference": 0.1,  # Different temporal correlations
        "outlier_ratio": 0.1,  # Unusual periods
        "periodicity_difference": 0.1,  # Unique cycle
        "stationarity_difference": 0.1,  # Different stationarity
        "seasonal_difference": 0.2,  # Different seasonal pattern
    }

    # Combined score
    result_df["advanced_trend_divergence_score"] = (
        weights["trend_divergence_score"] * result_df["trend_divergence_score"]
        + weights["autocorrelation_difference"]
        * result_df["autocorrelation_difference"]
        + weights["outlier_ratio"] * result_df["outlier_ratio"]
        + weights["periodicity_difference"] * result_df["periodicity_difference"]
        + weights["stationarity_difference"] * result_df["stationarity_difference"]
        + weights["seasonal_difference"] * result_df["seasonal_difference"]
    )

    # Normalize to 0-1 range
    max_score = result_df["advanced_trend_divergence_score"].max()
    if max_score > 0:
        result_df["advanced_trend_divergence_score"] = (
            result_df["advanced_trend_divergence_score"] / max_score
        )

    return result_df
