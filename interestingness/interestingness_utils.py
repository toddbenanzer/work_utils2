"""
Utility functions for the interestingness metrics package.

These functions help with ranking, summarizing, and visualizing interestingness
metrics for grouped data.
"""

from typing import Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def rank_groups_by_metric(
    df: pd.DataFrame,
    metric_func: Callable,
    metric_kwargs: Dict,
    group_col: Union[str, List[str]],
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Ranks groups by a specified interestingness metric.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    metric_func : callable
        Function from the interestingness metrics package to calculate the metric
    metric_kwargs : dict
        Keyword arguments to pass to the metric function
    group_col : str or list of str
        Column(s) containing the group identifiers
    ascending : bool, default=False
        Whether to sort in ascending order (False = most interesting first)

    Returns:
    --------
    pandas DataFrame
        DataFrame with groups and their interestingness scores, sorted by score
    """
    if df.empty:
        return pd.DataFrame()

    # Calculate the metric for each group
    results = []

    if isinstance(group_col, str):
        group_names = df[group_col].unique()
        for group in group_names:
            group_df = df[df[group_col] == group]
            score = metric_func(group_df, **metric_kwargs)

            # Handle both scalar and dictionary returns
            if isinstance(score, dict):
                # For metrics that return dictionaries, extract the main score value
                if "f_statistic" in score:
                    score_value = score["f_statistic"]
                elif "h_statistic" in score:
                    score_value = score["h_statistic"]
                elif "p_value" in score:
                    score_value = (
                        1.0 - score["p_value"]
                    )  # Convert p-value to score (lower p = higher score)
                else:
                    # Use the first value as the score
                    score_value = list(score.values())[0]
            else:
                score_value = score

            results.append({"group": group, "metric_score": score_value})
    else:
        # Handle multi-column grouping
        groups = df[group_col].drop_duplicates().values.tolist()
        for group_values in groups:
            # Create filter for this group combination
            mask = True
            for col, val in zip(group_col, group_values):
                mask = mask & (df[col] == val)

            group_df = df[mask]
            score = metric_func(group_df, **metric_kwargs)

            # Handle both scalar and dictionary returns
            if isinstance(score, dict):
                # For metrics that return dictionaries, extract the main score value
                if "f_statistic" in score:
                    score_value = score["f_statistic"]
                elif "h_statistic" in score:
                    score_value = score["h_statistic"]
                elif "p_value" in score:
                    score_value = (
                        1.0 - score["p_value"]
                    )  # Convert p-value to score (lower p = higher score)
                else:
                    # Use the first value as the score
                    score_value = list(score.values())[0]
            else:
                score_value = score

            result_dict = {col: val for col, val in zip(group_col, group_values)}
            result_dict["metric_score"] = score_value
            results.append(result_dict)

    # Create result DataFrame
    result_df = pd.DataFrame(results)

    # Sort by metric score
    return result_df.sort_values("metric_score", ascending=ascending)


def evaluate_all_metrics(
    df: pd.DataFrame,
    mean_col: str,
    count_col: str,
    var_col: Optional[str] = None,
    percentile_25_col: Optional[str] = None,
    percentile_75_col: Optional[str] = None,
) -> Dict:
    """
    Evaluate multiple interestingness metrics and return a comprehensive report.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count for each group
    var_col : str, optional
        Column name containing the variance for each group
    percentile_25_col : str, optional
        Column name containing the 25th percentile for each group
    percentile_75_col : str, optional
        Column name containing the 75th percentile for each group

    Returns:
    --------
    dict
        Dictionary containing all calculated metrics
    """
    # Import metrics here to avoid circular imports
    from .interestingness_metrics import (
        anova_f_statistic,
        coefficient_of_variation,
        effect_size_f,
        gini_coefficient,
        group_variance,
        max_deviation_ratio,
        range_to_mean_ratio,
    )
    from .interestingness_metrics_advanced import (
        discriminative_power,
        entropy_reduction,
        group_separation_index,
        kruskal_wallis_h,
        outlier_group_score,
    )

    results = {}

    # Basic metrics
    results["group_variance"] = group_variance(df, mean_col, count_col)
    results["coefficient_of_variation"] = coefficient_of_variation(
        df, mean_col, count_col
    )
    results["max_deviation_ratio"] = max_deviation_ratio(df, mean_col)
    results["range_to_mean_ratio"] = range_to_mean_ratio(df, mean_col)
    results["gini_coefficient"] = gini_coefficient(df, mean_col, count_col)

    # ANOVA-based metrics
    if var_col is not None:
        results["anova"] = anova_f_statistic(df, mean_col, count_col, var_col)
        results["effect_size_f"] = effect_size_f(df, mean_col, count_col, var_col)
        results["discriminative_power"] = discriminative_power(
            df, mean_col, var_col, count_col
        )

    # Advanced metrics
    results["entropy_reduction"] = entropy_reduction(df, mean_col, count_col)

    if percentile_25_col is not None and percentile_75_col is not None:
        results["kruskal_wallis"] = kruskal_wallis_h(
            df, mean_col, count_col, percentile_25_col, percentile_75_col
        )

    results["outlier_groups"] = outlier_group_score(df, mean_col, count_col)
    results["group_separation"] = group_separation_index(df, mean_col, count_col)

    return results


def plot_group_means(
    df: pd.DataFrame,
    mean_col: str,
    group_col: Union[str, List[str]],
    count_col: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> None:
    """
    Create a visualization of group means to display their differences.
    Optionally sizes points by group count.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    group_col : str or list of str
        Column(s) containing the group identifiers
    count_col : str, optional
        Column name containing the count for each group (for sizing points)
    title : str, optional
        Plot title
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

    plt.figure(figsize=figsize)

    # For point sizing
    if count_col is not None:
        # Normalize counts for point sizing (between 50 and 300)
        min_count = df[count_col].min()
        max_count = df[count_col].max()

        if max_count > min_count:
            sizes = 50 + 250 * (df[count_col] - min_count) / (max_count - min_count)
        else:
            sizes = 100  # Default size if all counts are the same
    else:
        sizes = 100  # Default size if count_col not provided

    # Handle single vs. multiple grouping columns
    if isinstance(group_col, str):
        # Create bar chart for single grouping variable
        plt.figure(figsize=figsize)

        # Sort by mean value
        sorted_df = df.sort_values(mean_col)

        # Create bar chart
        bars = plt.bar(range(len(sorted_df)), sorted_df[mean_col], alpha=0.7)

        # Add group labels
        plt.xticks(range(len(sorted_df)), sorted_df[group_col], rotation=45, ha="right")

        # Add labels and title
        plt.xlabel(group_col)
        plt.ylabel(mean_col)
        plt.title(title or f"Mean {mean_col} by {group_col}")

        # Add count labels if provided
        if count_col is not None:
            for i, bar in enumerate(bars):
                count = sorted_df.iloc[i][count_col]
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05 * plt.ylim()[1],
                    f"n={int(count)}",
                    ha="center",
                    va="bottom",
                    rotation=0,
                )
    else:
        # For multi-column grouping, use scatter plot
        # Convert to strings for display
        df = df.copy()

        # Create combined group label for display
        df["group_label"] = df[group_col].apply(
            lambda row: ", ".join(str(val) for val in row), axis=1
        )

        # Create scatter plot
        plt.scatter(range(len(df)), df[mean_col], s=sizes, alpha=0.7)

        # Add labels
        plt.xticks(range(len(df)), df["group_label"], rotation=45, ha="right")
        plt.xlabel("Group")
        plt.ylabel(mean_col)
        plt.title(title or f"Mean {mean_col} by Group")

        # Add count labels if provided
        if count_col is not None:
            for i, row in df.iterrows():
                plt.annotate(
                    f"n={int(row[count_col])}",
                    (i, row[mean_col]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                )

    # Add overall mean reference line
    overall_mean = df[mean_col].mean()
    plt.axhline(
        y=overall_mean,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Overall Mean: {overall_mean:.2f}",
    )

    plt.legend()
    plt.tight_layout()
    plt.show()


def interestingness_report(
    df: pd.DataFrame,
    mean_col: str,
    count_col: str,
    group_col: Union[str, List[str]],
    var_col: Optional[str] = None,
    percentile_25_col: Optional[str] = None,
    percentile_75_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a comprehensive interestingness report for the grouping,
    including visualizations and metric evaluations.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count for each group
    group_col : str or list of str
        Column(s) containing the group identifiers
    var_col : str, optional
        Column name containing the variance for each group
    percentile_25_col : str, optional
        Column name containing the 25th percentile for each group
    percentile_75_col : str, optional
        Column name containing the 75th percentile for each group

    Returns:
    --------
    pandas DataFrame
        Summary DataFrame with interestingness metrics
    """
    if df.empty:
        print("Empty DataFrame, nothing to analyze.")
        return pd.DataFrame()

    # Visualize the group means
    print(f"Visualizing mean {mean_col} across groups:")
    plot_group_means(df, mean_col, group_col, count_col)

    # Calculate all metrics
    results = evaluate_all_metrics(
        df, mean_col, count_col, var_col, percentile_25_col, percentile_75_col
    )

    # Extract scalar metrics
    scalar_metrics = {
        "Group Variance": results["group_variance"],
        "Coefficient of Variation": results["coefficient_of_variation"],
        "Max Deviation Ratio": results["max_deviation_ratio"],
        "Range to Mean Ratio": results["range_to_mean_ratio"],
        "Gini Coefficient": results["gini_coefficient"],
        "Entropy Reduction": results["entropy_reduction"],
        "Group Separation": results["group_separation"],
    }

    # Add ANOVA metrics if variance is provided
    if var_col is not None:
        scalar_metrics.update(
            {
                "ANOVA F-statistic": results["anova"]["f_statistic"],
                "ANOVA p-value": results["anova"]["p_value"],
                "Effect Size (Cohen's f)": results["effect_size_f"],
                "Discriminative Power": results["discriminative_power"],
            }
        )

    # Create summary DataFrame
    summary_df = pd.DataFrame(
        {"Metric": list(scalar_metrics.keys()), "Value": list(scalar_metrics.values())}
    )

    # Print detailed results
    print("\n----- Interestingness Analysis Results -----")

    # ANOVA results if available
    if var_col is not None and "anova" in results:
        sig_status = (
            "SIGNIFICANT" if results["anova"]["significant"] else "NOT SIGNIFICANT"
        )
        print(f"\nANOVA Test: {sig_status}")
        print(f"F-statistic: {results['anova']['f_statistic']:.3f}")
        print(f"p-value: {results['anova']['p_value']:.5f}")
        print(f"Effect size (Cohen's f): {results['effect_size_f']:.3f}")

        # Interpret effect size
        if results["effect_size_f"] < 0.1:
            effect_interpretation = "minimal effect"
        elif results["effect_size_f"] < 0.25:
            effect_interpretation = "small effect"
        elif results["effect_size_f"] < 0.4:
            effect_interpretation = "medium effect"
        else:
            effect_interpretation = "large effect"

        print(f"Interpretation: {effect_interpretation}")

    # Outlier analysis
    outliers = results["outlier_groups"]
    if outliers["outlier_count"] > 0:
        print(
            f"\nOutlier Analysis: {outliers['outlier_count']} outlier groups detected"
        )
        print(f"Maximum Z-score: {outliers['max_z_score']:.2f}")
        if len(outliers["outlier_indices"]) <= 5:  # Only show if few outliers
            print(f"Outlier indices: {outliers['outlier_indices']}")
    else:
        print("\nOutlier Analysis: No outlier groups detected")

    # Interpretations
    print("\nInterpretation of Metrics:")
    if var_col is not None and "anova" in results and results["anova"]["significant"]:
        print("✓ Group means differ significantly (ANOVA p-value < 0.05)")
    elif var_col is not None and "anova" in results:
        print("✗ Group means do not differ significantly (ANOVA p-value >= 0.05)")

    if results["coefficient_of_variation"] > 0.3:
        print("✓ High variation across group means (CV > 0.3)")
    else:
        print("✗ Low variation across group means (CV <= 0.3)")

    if results["entropy_reduction"] > 0.5:
        print("✓ Grouping explains substantial variation in the data")
    else:
        print("✗ Grouping explains limited variation in the data")

    if var_col is not None and results["discriminative_power"] > 1.0:
        print(
            "✓ Good discriminative power (between-group variance > within-group variance)"
        )
    elif var_col is not None:
        print("✗ Poor discriminative power (within-group variance dominates)")

    # Overall assessment
    print("\nOverall Assessment:")

    # Calculate an overall interestingness score (simplified)
    scores = []
    if var_col is not None and "anova" in results:
        # Add score based on statistical significance
        scores.append(1.0 if results["anova"]["significant"] else 0.0)
        # Add score based on effect size
        scores.append(min(1.0, results["effect_size_f"] / 0.4))  # Normalize to 0-1

    # Add scores from other metrics
    scores.append(min(1.0, results["coefficient_of_variation"] / 0.3))
    scores.append(min(1.0, results["max_deviation_ratio"]))
    scores.append(results["entropy_reduction"])

    if scores:
        overall_score = sum(scores) / len(scores)

        if overall_score > 0.7:
            print("This grouping appears to be HIGHLY INTERESTING.")
        elif overall_score > 0.4:
            print("This grouping appears to be MODERATELY INTERESTING.")
        else:
            print("This grouping appears to be MINIMALLY INTERESTING.")

    # Return the summary DataFrame
    return summary_df


def flatten_dict(nested_dict, parent_key="", sep="_"):
    """
    Flatten a nested dictionary structure into a single-level dictionary.

    Args:
        nested_dict (dict): The dictionary to flatten
        parent_key (str): The parent key for nested values (used in recursion)
        sep (str): Separator to use between parent and child keys

    Returns:
        dict: A flattened dictionary with no nested structures
    """
    flat_dict = {}

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            flat_dict.update(flatten_dict(value, new_key, sep))
        elif isinstance(value, list):
            # Convert list to a string representation
            flat_dict[new_key] = str(value)
        else:
            # For simple values, just add them to the result
            flat_dict[new_key] = value

    return flat_dict
