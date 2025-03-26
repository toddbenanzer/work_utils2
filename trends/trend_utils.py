"""
Utility functions for the trend metrics package.

These functions help with analyzing, ranking, summarizing, and visualizing
trend-related metrics for grouped data over time.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def rank_groups_by_trend_divergence(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rank groups by their trend divergence score.

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
        DataFrame with groups ranked by trend divergence score
    """
    from .trend_metrics import trend_divergence_score

    # Calculate trend divergence score
    scores = trend_divergence_score(df, group_col, time_col, mean_col, count_col)

    # Sort by score (descending)
    return scores.sort_values("trend_divergence_score", ascending=False)


def plot_group_trends(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
    top_n: Optional[int] = None,
    plot_overall: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot trends over time for different groups.

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
        Column containing the counts for weighted calculations or point sizing
    top_n : int, optional
        Number of top divergent groups to include (if None, plot all)
    plot_overall : bool, default=True
        Whether to plot the overall trend as a reference
    figsize : tuple, default=(10, 6)
        Figure size as (width, height) in inches

    Returns:
    --------
    None
        Displays the plot
    """
    if df.empty:
        print("Empty DataFrame, nothing to plot.")
        return

    # If top_n is specified, get the top divergent groups
    if top_n is not None:
        from .trend_metrics import trend_divergence_score

        # Calculate trend divergence score
        scores = trend_divergence_score(df, group_col, time_col, mean_col, count_col)

        # Limit to top N groups
        top_groups = scores.nlargest(top_n, "trend_divergence_score")

        # Filter data to only include these groups
        if isinstance(group_col, str):
            df_filtered = df[df[group_col].isin(top_groups[group_col])]
        else:
            # For multi-column grouping, create a mask
            mask = pd.Series(False, index=df.index)
            for _, row in top_groups.iterrows():
                row_mask = pd.Series(True, index=df.index)
                for col in group_col:
                    row_mask &= df[col] == row[col]
                mask |= row_mask

            df_filtered = df[mask]
    else:
        df_filtered = df

    # Create figure
    plt.figure(figsize=figsize)

    # Check if time_col is a date/datetime
    is_datetime = pd.api.types.is_datetime64_any_dtype(df[time_col])

    # Pre-convert time column for consistent sorting
    if is_datetime:
        time_values = sorted(df[time_col].unique())
    else:
        # Try to sort numerically if possible
        unique_times = df[time_col].unique()
        try:
            time_values = sorted(unique_times)
        except:
            # If sorting fails, use original order
            time_values = unique_times

    # Plot overall trend if requested
    if plot_overall:
        # Calculate overall mean for each time period
        overall = df.groupby(time_col)[mean_col].mean().reset_index()

        # Plot overall trend as a thicker, black line
        if is_datetime:
            plt.plot(
                overall[time_col], overall[mean_col], "k-", linewidth=2, label="Overall"
            )
        else:
            plt.plot(overall[mean_col], "k-", linewidth=2, label="Overall")
            plt.xticks(range(len(time_values)), time_values, rotation=45)

    # Plot each group's trend
    if isinstance(group_col, str):
        # Single group column
        groups = df_filtered[group_col].unique()

        for group in groups:
            group_data = df_filtered[df_filtered[group_col] == group]

            # Sort by time
            group_data = group_data.sort_values(by=time_col)

            # Different line style for each group
            if is_datetime:
                plt.plot(
                    group_data[time_col],
                    group_data[mean_col],
                    "o-",
                    label=f"{group_col}={group}",
                )
            else:
                # Convert time values to their position in the sorted list
                x_positions = [time_values.index(t) for t in group_data[time_col]]
                plt.plot(
                    x_positions,
                    group_data[mean_col],
                    "o-",
                    label=f"{group_col}={group}",
                )
    else:
        # Multi-column grouping
        # Create a unique identifier for each group
        def group_id(row):
            return ", ".join(f"{col}={row[col]}" for col in group_col)

        # Get unique group combinations
        group_df = df_filtered[group_col].drop_duplicates()
        group_df["group_id"] = group_df.apply(group_id, axis=1)

        for _, row in group_df.iterrows():
            # Create filter for this group combination
            mask = pd.Series(True, index=df_filtered.index)
            for col in group_col:
                mask &= df_filtered[col] == row[col]

            group_data = df_filtered[mask].sort_values(by=time_col)

            if is_datetime:
                plt.plot(
                    group_data[time_col],
                    group_data[mean_col],
                    "o-",
                    label=row["group_id"],
                )
            else:
                # Convert time values to their position in the sorted list
                x_positions = [time_values.index(t) for t in group_data[time_col]]
                plt.plot(x_positions, group_data[mean_col], "o-", label=row["group_id"])

    # Set labels and title
    plt.xlabel(time_col)
    plt.ylabel(mean_col)
    if top_n is not None:
        plt.title(f"Trend Comparison for Top {top_n} Most Divergent Groups")
    else:
        plt.title(f"Trend Comparison by {group_col}")

    # Format x-axis for dates
    if is_datetime:
        plt.gcf().autofmt_xdate()

        # Choose appropriate date formatter based on range
        date_range = max(time_values) - min(time_values)
        if date_range.days > 365 * 2:  # More than 2 years
            date_format = mdates.DateFormatter("%Y")
        elif date_range.days > 90:  # More than 3 months
            date_format = mdates.DateFormatter("%b %Y")
        elif date_range.days > 30:  # More than a month
            date_format = mdates.DateFormatter("%d %b")
        else:
            date_format = mdates.DateFormatter("%d %b")

        plt.gca().xaxis.set_major_formatter(date_format)

    # Add legend with smaller font size and outside the plot
    plt.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Show plot
    plt.show()


def evaluate_all_trend_metrics(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
    var_col: Optional[str] = None,
    period: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate multiple trend metrics and return a comprehensive evaluation.

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
    var_col : str, optional
        Column containing the variance for each group-time combination
    period : int, optional
        Number of time periods in a seasonal cycle, if known

    Returns:
    --------
    pandas DataFrame
        DataFrame with all calculated trend metrics for each group
    """
    from .trend_metrics import (
        acceleration_detection,
        trend_break_detection,
        trend_consistency,
        trend_direction_difference,
        trend_divergence_score,
        trend_slope_divergence,
        trend_volatility,
    )

    try:
        from .trend_metrics_advanced import (
            advanced_trend_divergence_score,
            autocorrelation_pattern,
            outlier_period_detection,
            periodicity_detection,
            seasonal_pattern_difference,
            trend_stationarity_test,
        )

        has_advanced_metrics = True
    except ImportError:
        has_advanced_metrics = False

    if df.empty:
        return pd.DataFrame()

    # Get group identifier columns
    if isinstance(group_col, str):
        id_cols = [group_col]
    else:
        id_cols = group_col

    # Ensure all groups have the same time points for fair comparison
    # (fill in missing combinations with NaN, then drop them)
    all_groups = df[group_col].drop_duplicates()
    all_times = df[time_col].drop_duplicates()

    # Calculate core metrics

    # Slope divergence
    slope_df = trend_slope_divergence(df, group_col, time_col, mean_col, count_col)

    # Direction difference
    direction_df = trend_direction_difference(
        df, group_col, time_col, mean_col, count_col
    )

    # Volatility
    volatility_df = trend_volatility(df, group_col, time_col, mean_col)

    # Consistency
    consistency_df = trend_consistency(df, group_col, time_col, mean_col)

    # Acceleration
    acceleration_df = acceleration_detection(df, group_col, time_col, mean_col)

    # Trend breaks
    break_df = trend_break_detection(df, group_col, time_col, mean_col)

    # Combined score
    score_df = trend_divergence_score(df, group_col, time_col, mean_col, count_col)

    # Merge all metrics
    # Start with slope divergence
    result_df = slope_df.copy()

    # Add other metrics
    metrics_to_add = [
        (direction_df, ["direction_difference", "direction_significance"]),
        (volatility_df, ["volatility", "relative_volatility"]),
        (consistency_df, ["r_squared", "consistency_score"]),
        (acceleration_df, ["acceleration", "is_accelerating", "relative_acceleration"]),
        (break_df, ["has_trend_break", "break_magnitude", "normalized_break"]),
        (score_df, ["trend_divergence_score"]),
    ]

    for df_to_add, cols_to_add in metrics_to_add:
        result_df = pd.merge(result_df, df_to_add[id_cols + cols_to_add], on=id_cols)

    # Add advanced metrics if available
    if has_advanced_metrics:
        advanced_metrics_to_add = []

        # ACF pattern difference
        acf_df = autocorrelation_pattern(df, group_col, time_col, mean_col)
        advanced_metrics_to_add.append((acf_df, ["autocorrelation_difference"]))

        # Outlier periods
        outlier_df = outlier_period_detection(df, group_col, time_col, mean_col)
        advanced_metrics_to_add.append(
            (outlier_df, ["has_outlier_periods", "outlier_ratio"])
        )

        # Periodicity
        periodicity_df = periodicity_detection(df, group_col, time_col, mean_col)
        advanced_metrics_to_add.append(
            (
                periodicity_df,
                [
                    "has_periodicity",
                    "period_length",
                    "periodicity_strength",
                    "periodicity_difference",
                ],
            )
        )

        # Stationarity
        stationarity_df = trend_stationarity_test(df, group_col, time_col, mean_col)
        advanced_metrics_to_add.append(
            (stationarity_df, ["is_stationary", "p_value", "stationarity_difference"])
        )

        # Seasonal patterns if period is provided
        if period is not None and period > 1:
            seasonal_df = seasonal_pattern_difference(
                df, group_col, time_col, mean_col, period
            )
            advanced_metrics_to_add.append(
                (seasonal_df, ["seasonal_difference", "has_seasonal_pattern"])
            )

        # Add all advanced metrics
        for df_to_add, cols_to_add in advanced_metrics_to_add:
            available_cols = [col for col in cols_to_add if col in df_to_add.columns]
            if available_cols:
                result_df = pd.merge(
                    result_df, df_to_add[id_cols + available_cols], on=id_cols
                )

        # Add advanced score
        if period is not None and period > 1:
            advanced_score_df = advanced_trend_divergence_score(
                df, group_col, time_col, mean_col, count_col, period
            )
        else:
            advanced_score_df = advanced_trend_divergence_score(
                df, group_col, time_col, mean_col, count_col
            )

        result_df = pd.merge(
            result_df,
            advanced_score_df[id_cols + ["advanced_trend_divergence_score"]],
            on=id_cols,
        )

    return result_df


def trend_analysis_report(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
    var_col: Optional[str] = None,
    period: Optional[int] = None,
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Generate a comprehensive report of trend metrics with visualizations.

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
    var_col : str, optional
        Column containing the variance for each group-time combination
    period : int, optional
        Number of time periods in a seasonal cycle, if known
    top_n : int, default=3
        Number of top interesting groups to highlight

    Returns:
    --------
    pandas DataFrame
        Summary DataFrame with trend divergence metrics
    """
    if df.empty:
        print("Empty DataFrame, nothing to analyze.")
        return pd.DataFrame()

    # Calculate all metrics
    all_metrics = evaluate_all_trend_metrics(
        df, group_col, time_col, mean_col, count_col, var_col, period
    )

    # Check if advanced metrics are available
    has_advanced = "advanced_trend_divergence_score" in all_metrics.columns

    # Determine which score to use for ranking
    if has_advanced:
        score_col = "advanced_trend_divergence_score"
    else:
        score_col = "trend_divergence_score"

    # Get top interesting groups
    top_groups = all_metrics.nlargest(top_n, score_col)

    # Print summary header
    print("\n===== TREND ANALYSIS REPORT =====\n")

    # Visualize trends for all groups
    print("Overall Trend Comparison:")
    plot_group_trends(df, group_col, time_col, mean_col, count_col=count_col)

    # Visualize top interesting groups
    print(f"\nTop {top_n} Most Interesting Trends:")
    plot_group_trends(
        df, group_col, time_col, mean_col, count_col=count_col, top_n=top_n
    )

    # Print details about top groups
    print("\nDetailed Analysis of Top Groups:")
    for i, (_, row) in enumerate(top_groups.iterrows()):
        # Create group identifier string
        if isinstance(group_col, str):
            group_id = f"{group_col}={row[group_col]}"
        else:
            group_id = ", ".join(f"{col}={row[col]}" for col in group_col)

        print(f"\n{i+1}. {group_id}")

        # Print key metrics
        print(f"   Overall Divergence Score: {row[score_col]:.3f}")
        print(f"   Slope: {row['slope']:.3f} (Overall: {row['overall_slope']:.3f})")

        # Direction difference
        if row["direction_difference"] > 0:
            print("   ✓ Trends in OPPOSITE direction from overall trend")

        # Consistency
        consistency = row["consistency_score"]
        if consistency < 0.5:
            print(f"   ✓ Inconsistent trend (R² = {consistency:.2f})")
        else:
            print(f"   Consistent trend (R² = {consistency:.2f})")

        # Volatility
        relative_vol = row["relative_volatility"]
        if relative_vol > 1.5:
            print(f"   ✓ High volatility ({relative_vol:.1f}x average)")
        elif relative_vol < 0.5:
            print(f"   ✓ Unusually stable ({relative_vol:.1f}x average)")

        # Acceleration
        if row["is_accelerating"]:
            print("   ✓ Accelerating trend")
        elif row["relative_acceleration"] < -0.5:
            print("   ✓ Decelerating trend")

        # Trend breaks
        if row["has_trend_break"]:
            print("   ✓ Shows a trend break/change point")

        # Advanced metrics if available
        if has_advanced:
            if "has_periodicity" in row and row["has_periodicity"]:
                print(
                    f"   ✓ Shows periodic pattern (period: {row['period_length']} units)"
                )

            if "has_outlier_periods" in row and row["has_outlier_periods"]:
                print("   ✓ Contains outlier time periods")

            if "seasonal_difference" in row and row["seasonal_difference"] > 0.6:
                print("   ✓ Shows unusual seasonal pattern")

    # Print overall insights
    print("\nOverall Insights:")

    # Check if there are groups with opposite direction
    opposite_direction = all_metrics[all_metrics["direction_difference"] > 0]
    if not opposite_direction.empty:
        print(
            f"- {len(opposite_direction)} groups trend in the opposite direction from the overall trend."
        )

    # Check for acceleration patterns
    accelerating = all_metrics[all_metrics["is_accelerating"]]
    decelerating = all_metrics[all_metrics["relative_acceleration"] < -0.3]
    if not accelerating.empty or not decelerating.empty:
        print(
            f"- {len(accelerating)} groups show accelerating trends, while {len(decelerating)} show decelerating trends."
        )

    # Check for high volatility
    volatile = all_metrics[all_metrics["relative_volatility"] > 1.5]
    if not volatile.empty:
        print(f"- {len(volatile)} groups show unusually high volatility.")

    # Check for trend breaks
    breaks = all_metrics[all_metrics["has_trend_break"]]
    if not breaks.empty:
        print(f"- {len(breaks)} groups show significant trend breaks or change points.")

    # Advanced insights
    if has_advanced:
        if "has_periodicity" in all_metrics.columns:
            periodic = all_metrics[all_metrics["has_periodicity"]]
            if not periodic.empty:
                print(f"- {len(periodic)} groups show periodic patterns.")

        if "has_seasonal_pattern" in all_metrics.columns:
            seasonal = all_metrics[all_metrics["has_seasonal_pattern"]]
            if not seasonal.empty:
                print(f"- {len(seasonal)} groups show detectable seasonal patterns.")

    # Return the full metrics DataFrame sorted by interestingness score
    return all_metrics.sort_values(score_col, ascending=False)


def compare_trends_across_dimensions(
    df: pd.DataFrame,
    group_dimensions: List[Union[str, List[str]]],
    time_col: str,
    mean_col: str,
    count_col: Optional[str] = None,
    var_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare how interesting trends are across different grouping dimensions.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group-time combination
    group_dimensions : list of str or list of lists
        Different grouping columns or combinations to compare
    time_col : str
        Column identifying the time periods
    mean_col : str
        Column containing the mean values
    count_col : str, optional
        Column containing the counts for weighted calculations
    var_col : str, optional
        Column containing the variance for each group-time combination

    Returns:
    --------
    pandas DataFrame
        Comparison of divergence scores across different grouping dimensions
    """
    from .trend_metrics import trend_divergence_score

    if df.empty:
        return pd.DataFrame()

    results = []

    # Evaluate each grouping dimension
    for dimension in group_dimensions:
        # Calculate divergence score for this grouping
        scores = trend_divergence_score(df, dimension, time_col, mean_col, count_col)

        # Get statistics about the scores
        stats = {
            "dimension": (
                dimension if isinstance(dimension, str) else "+".join(dimension)
            ),
            "num_groups": len(scores),
            "max_score": scores["trend_divergence_score"].max(),
            "mean_score": scores["trend_divergence_score"].mean(),
            "median_score": scores["trend_divergence_score"].median(),
            "std_score": scores["trend_divergence_score"].std(),
            "num_interesting": sum(scores["trend_divergence_score"] > 0.7),
            "pct_interesting": sum(scores["trend_divergence_score"] > 0.7)
            / len(scores)
            * 100,
        }

        # Add count of unique groups
        if isinstance(dimension, str):
            stats["unique_groups"] = df[dimension].nunique()
        else:
            stats["unique_groups"] = df.groupby(dimension).ngroups

        results.append(stats)

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Sort by interestingness metrics
    result_df = result_df.sort_values("max_score", ascending=False)

    return result_df


def detect_trend_anomalies(
    df: pd.DataFrame,
    group_col: Union[str, List[str]],
    time_col: str,
    mean_col: str,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Detect groups with anomalous trend patterns.

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
    threshold : float, default=0.7
        Threshold for considering a trend anomalous

    Returns:
    --------
    pandas DataFrame
        Groups with anomalous trends and their metrics
    """
    # Get comprehensive trend metrics
    all_metrics = evaluate_all_trend_metrics(df, group_col, time_col, mean_col)

    # Check if advanced metrics are available
    has_advanced = "advanced_trend_divergence_score" in all_metrics.columns

    # Determine which score to use
    if has_advanced:
        score_col = "advanced_trend_divergence_score"
    else:
        score_col = "trend_divergence_score"

    # Filter for anomalous groups
    anomalies = all_metrics[all_metrics[score_col] > threshold].copy()

    # Sort by score (most anomalous first)
    anomalies = anomalies.sort_values(score_col, ascending=False)

    if len(anomalies) == 0:
        print("No trend anomalies detected with the current threshold.")
    else:
        print(f"Detected {len(anomalies)} groups with anomalous trends.")

    return anomalies
