"""
Functions for ranking and comparing segment combinations based on interestingness measures.
This module helps identify which segmentations reveal the most insights in the data.
"""

import itertools
from typing import List, Optional, Union

import pandas as pd

from ..interestingness.anova import (
    calculate_anova_f_statistic,
    calculate_between_within_variance_ratio,
    calculate_eta_squared,
)
from ..interestingness.variance_measures import (
    calculate_coefficient_variation,
    calculate_max_deviation,
    calculate_weighted_variance,
)


def rank_segments_by_interestingness(
    df: pd.DataFrame,
    value_col: str,
    segment_candidates: List[Union[str, List[str]]],
    methods: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Rank different segment combinations by interestingness measures.

    Args:
        df: DataFrame with data
        value_col: Column containing the metric to analyze
        segment_candidates: List of segment columns or column combinations to evaluate
            Each item can be a single column name or a list of column names for multi-column segments
        methods: List of interestingness measures to calculate
            Options: 'weighted_variance', 'coefficient_variation', 'max_deviation',
                     'f_statistic', 'eta_squared', 'variance_ratio'
            Default (None): uses all methods
        weight_col: Optional column containing weights (e.g., 'count' for segment size)

    Returns:
        DataFrame with interestingness measures for each segment combination, ranked by a composite score
    """
    # Default methods if none provided
    if methods is None:
        methods = [
            "weighted_variance",
            "coefficient_variation",
            "max_deviation",
            "f_statistic",
            "eta_squared",
            "variance_ratio",
        ]

    # Validate methods
    valid_methods = [
        "weighted_variance",
        "coefficient_variation",
        "max_deviation",
        "f_statistic",
        "eta_squared",
        "variance_ratio",
    ]

    invalid_methods = [m for m in methods if m not in valid_methods]
    if invalid_methods:
        raise ValueError(
            f"Invalid interestingness methods: {invalid_methods}. "
            f"Valid options are: {valid_methods}"
        )

    # Check if value_col exists
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame")

    # Check if weight_col exists if provided
    if weight_col is not None and weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    # Store results for each segment combination
    results = []

    # Evaluate each segment candidate
    for segment_cols in segment_candidates:
        # Ensure segment_cols is a list
        if isinstance(segment_cols, str):
            segment_cols = [segment_cols]

        # Check if all segment columns exist
        missing_cols = [col for col in segment_cols if col not in df.columns]
        if missing_cols:
            print(
                f"Warning: Skipping segment combination with missing columns: {missing_cols}"
            )
            continue

        # Calculate each requested interestingness measure
        measures = {}

        try:
            if "weighted_variance" in methods:
                result = calculate_weighted_variance(
                    df, value_col, segment_cols, weight_col
                )
                measures["weighted_variance"] = result["weighted_variance"]

            if "coefficient_variation" in methods:
                result = calculate_coefficient_variation(
                    df, value_col, segment_cols, weight_col
                )
                measures["coefficient_variation"] = result["coefficient_variation"]

            if "max_deviation" in methods:
                result = calculate_max_deviation(
                    df, value_col, segment_cols, weight_col
                )
                measures["max_deviation"] = result["max_deviation"]
                measures["max_deviation_relative"] = result["max_deviation_relative"]
                measures["max_deviation_segment"] = result["max_deviation_segment"]

            if "f_statistic" in methods:
                result = calculate_anova_f_statistic(df, value_col, segment_cols)
                if "error" not in result:
                    measures["f_statistic"] = result["f_statistic"]
                    measures["f_p_value"] = result["p_value"]
                    measures["f_significant"] = result["p_value"] < 0.05

            if "eta_squared" in methods:
                result = calculate_eta_squared(df, value_col, segment_cols)
                if "error" not in result:
                    measures["eta_squared"] = result["eta_squared"]
                    measures["effect_size"] = result.get("effect_size", "unknown")

            if "variance_ratio" in methods:
                result = calculate_between_within_variance_ratio(
                    df, value_col, segment_cols
                )
                if "error" not in result:
                    measures["variance_ratio"] = result["variance_ratio"]

            # Add metadata about the segmentation
            segment_str = "+".join(segment_cols)
            measures["segment_columns"] = segment_str
            measures["segment_list"] = segment_cols
            measures["n_segments"] = df.groupby(segment_cols).ngroups

            # Calculate composite score based on normalized rank of each measure
            # We'll add this after collecting all results

            results.append(measures)

        except Exception as e:
            print(f"Error calculating interestingness for {segment_cols}: {str(e)}")

    # Convert results to DataFrame
    if not results:
        return pd.DataFrame()  # Return empty DataFrame if no valid results

    result_df = pd.DataFrame(results)

    # Calculate composite score based on normalized ranks
    # First, identify which measures we have in the results
    available_measures = [
        col
        for col in [
            "weighted_variance",
            "coefficient_variation",
            "max_deviation",
            "f_statistic",
            "eta_squared",
            "variance_ratio",
        ]
        if col in result_df.columns
    ]

    if available_measures:
        # For each measure, compute percentile ranks (0-1)
        # Higher values of each measure = more interesting
        for measure in available_measures:
            rank_col = f"{measure}_rank"
            # Create normalized rank (0-1, higher = more interesting)
            result_df[rank_col] = result_df[measure].rank(pct=True)

        # Composite score = average of normalized ranks
        rank_cols = [f"{measure}_rank" for measure in available_measures]
        result_df["composite_score"] = result_df[rank_cols].mean(axis=1)

        # Sort by composite score (descending)
        result_df = result_df.sort_values("composite_score", ascending=False)

    return result_df


def generate_segment_combinations(
    df: pd.DataFrame, segment_cols: List[str], max_combination_size: int = 2
) -> List[List[str]]:
    """
    Generate all possible combinations of segment columns up to a specified size.

    Args:
        df: DataFrame with data
        segment_cols: List of columns to create combinations from
        max_combination_size: Maximum number of columns to combine (default: 2)

    Returns:
        List of column combinations to evaluate
    """
    # Check if all columns exist
    missing_cols = [col for col in segment_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Generate all combinations from size 1 to max_combination_size
    combinations = []
    for r in range(1, min(max_combination_size + 1, len(segment_cols) + 1)):
        # Add each r-combination as a list
        combinations.extend(list(itertools.combinations(segment_cols, r)))

    # Convert tuples to lists
    return [list(combo) for combo in combinations]


def compare_segment_combinations(
    df: pd.DataFrame,
    value_col: str,
    segment_combinations: List[List[str]],
    method: str = "weighted_variance",
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple segment combinations using a single interestingness method.

    Args:
        df: DataFrame with data
        value_col: Column containing the metric to analyze
        segment_combinations: List of segment column combinations to compare
        method: Interestingness method to use (default: 'weighted_variance')
        weight_col: Optional column containing weights

    Returns:
        DataFrame with interestingness measure for each segment combination
    """
    # Validate method
    valid_methods = [
        "weighted_variance",
        "coefficient_variation",
        "max_deviation",
        "f_statistic",
        "eta_squared",
        "variance_ratio",
    ]

    if method not in valid_methods:
        raise ValueError(
            f"Invalid method: {method}. Valid options are: {valid_methods}"
        )

    # Check if value_col exists
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in DataFrame")

    # Check if weight_col exists if provided
    if weight_col is not None and weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    # Use rank_segments_by_interestingness with a single method
    return rank_segments_by_interestingness(
        df=df,
        value_col=value_col,
        segment_candidates=segment_combinations,
        methods=[method],
        weight_col=weight_col,
    )


def find_most_interesting_segments(
    df: pd.DataFrame,
    value_col: str,
    segment_cols: List[str],
    top_n: int = 5,
    max_combination_size: int = 2,
    methods: Optional[List[str]] = None,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Convenience function to find the most interesting segment combinations.
    Generates all possible segment combinations, calculates interestingness,
    and returns the top N most interesting combinations.

    Args:
        df: DataFrame with data
        value_col: Column containing the metric to analyze
        segment_cols: List of columns to create combinations from
        top_n: Number of top combinations to return (default: 5)
        max_combination_size: Maximum number of columns to combine (default: 2)
        methods: List of interestingness methods to calculate (default: None = all)
        weight_col: Optional column containing weights

    Returns:
        DataFrame with top N most interesting segment combinations
    """
    # Generate all possible segment combinations
    combinations = generate_segment_combinations(df, segment_cols, max_combination_size)

    # Rank segment combinations by interestingness
    rankings = rank_segments_by_interestingness(
        df=df,
        value_col=value_col,
        segment_candidates=combinations,
        methods=methods,
        weight_col=weight_col,
    )

    # Return top N combinations
    return rankings.head(top_n)


def evaluate_segment_over_time(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    segment_cols: Union[str, List[str]],
    method: str = "weighted_variance",
    weight_col: Optional[str] = None,
    window_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate interestingness of a segmentation over time.

    Args:
        df: DataFrame with time series data
        time_col: Column containing time values
        value_col: Column containing the metric to analyze
        segment_cols: Column(s) defining the segments to analyze
        method: Interestingness method to use (default: 'weighted_variance')
        weight_col: Optional column containing weights
        window_size: Optional rolling window size to calculate interestingness over
                     If None, calculates for each unique time period separately

    Returns:
        DataFrame with interestingness measures for each time period
    """
    # Ensure segment_cols is a list
    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    # Check if columns exist
    required_cols = [time_col, value_col] + segment_cols
    if weight_col:
        required_cols.append(weight_col)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {missing_cols}")

    # Group by time periods
    df_sorted = df.sort_values(by=time_col)

    if window_size is None:
        # Calculate interestingness for each unique time period
        time_periods = df_sorted[time_col].unique()
        results = []

        for period in time_periods:
            period_df = df_sorted[df_sorted[time_col] == period]

            # Skip periods with too few data points
            if len(period_df) < 2:
                continue

            try:
                # Calculate interestingness for this period
                method_func = {
                    "weighted_variance": calculate_weighted_variance,
                    "coefficient_variation": calculate_coefficient_variation,
                    "max_deviation": calculate_max_deviation,
                    "f_statistic": calculate_anova_f_statistic,
                    "eta_squared": calculate_eta_squared,
                    "variance_ratio": calculate_between_within_variance_ratio,
                }[method]

                result = method_func(period_df, value_col, segment_cols, weight_col)

                # Add period information
                result_dict = {"time_period": period}

                # Extract the main measure value based on method
                measure_key = method
                if method == "f_statistic":
                    measure_key = "f_statistic"
                    result_dict["p_value"] = result["p_value"]
                    result_dict["significant"] = result["p_value"] < 0.05

                result_dict["measure"] = result[measure_key]
                result_dict["n_data_points"] = len(period_df)

                results.append(result_dict)
            except Exception as e:
                print(f"Error calculating {method} for period {period}: {str(e)}")

        # Convert to DataFrame and sort by time period
        if results:
            results_df = pd.DataFrame(results)
            return results_df.sort_values(by="time_period")
        else:
            return pd.DataFrame()

    else:
        # Calculate interestingness using a rolling window
        unique_periods = df_sorted[time_col].unique()
        if len(unique_periods) < window_size:
            raise ValueError(f"Not enough time periods for window size {window_size}")

        results = []

        for i in range(len(unique_periods) - window_size + 1):
            window_periods = unique_periods[i : i + window_size]
            window_df = df_sorted[df_sorted[time_col].isin(window_periods)]

            try:
                # Calculate interestingness for this window
                method_func = {
                    "weighted_variance": calculate_weighted_variance,
                    "coefficient_variation": calculate_coefficient_variation,
                    "max_deviation": calculate_max_deviation,
                    "f_statistic": calculate_anova_f_statistic,
                    "eta_squared": calculate_eta_squared,
                    "variance_ratio": calculate_between_within_variance_ratio,
                }[method]

                result = method_func(window_df, value_col, segment_cols, weight_col)

                # Add window information
                result_dict = {
                    "window_start": window_periods[0],
                    "window_end": window_periods[-1],
                    "window_center": window_periods[window_size // 2],
                }

                # Extract the main measure value based on method
                measure_key = method
                if method == "f_statistic":
                    measure_key = "f_statistic"
                    result_dict["p_value"] = result["p_value"]
                    result_dict["significant"] = result["p_value"] < 0.05

                result_dict["measure"] = result[measure_key]
                result_dict["n_data_points"] = len(window_df)

                results.append(result_dict)
            except Exception as e:
                print(
                    f"Error calculating {method} for window {window_periods[0]} to {window_periods[-1]}: {str(e)}"
                )

        # Convert to DataFrame and sort by window center
        if results:
            results_df = pd.DataFrame(results)
            return results_df.sort_values(by="window_center")
        else:
            return pd.DataFrame()
