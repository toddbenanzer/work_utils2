import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def analyze_offer_differences(df, target_variable, dimensions, weights="count"):
    """
    Analyze which dimensions best explain differences between offer and non-offer groups.

    Parameters:
    -----------
    df : pandas DataFrame
        Account-level data containing 'offer_status' and other dimensions
    target_variable : str
        Column name for the target variable (e.g., 'average_balance' or 'has_credit_card')
    dimensions : list
        List of column names to analyze (e.g., ['age_bucket', 'income_range', 'region'])
    weights : str, default 'count'
        How to weight the importance. Options: 'count', 'effect_size', 'statistical'

    Returns:
    --------
    pandas DataFrame
        Ranked dimensions with effect sizes, p-values, and importance scores
    """
    results = []

    # Check if target is binary (for classification metrics like credit card adoption)
    is_binary = df[target_variable].nunique() <= 2

    for dimension in dimensions:
        # Aggregate data by dimension and offer_status
        if is_binary:
            # For binary target (like credit card ownership), calculate adoption rate
            agg_df = (
                df.groupby([dimension, "offer_status"])
                .agg(
                    count=pd.NamedAgg(target_variable, "count"),
                    sum=pd.NamedAgg(target_variable, "sum"),
                )
                .reset_index()
            )
            agg_df["rate"] = agg_df["sum"] / agg_df["count"]
        else:
            # For continuous target (like average balance), calculate mean
            agg_df = (
                df.groupby([dimension, "offer_status"])
                .agg(
                    count=pd.NamedAgg(target_variable, "count"),
                    mean=pd.NamedAgg(target_variable, "mean"),
                )
                .reset_index()
            )

        # Calculate effect size for each value in this dimension
        dimension_results = []

        # Get unique values in this dimension
        dim_values = df[dimension].unique()

        for value in dim_values:
            subset = agg_df[agg_df[dimension] == value]

            # Skip if we don't have both offer and non-offer for this value
            if subset.shape[0] < 2:
                continue

            offer_row = subset[subset["offer_status"] == "marketing offer"]
            non_offer_row = subset[subset["offer_status"] == "not marketing offer"]

            # Skip if either group is missing
            if offer_row.empty or non_offer_row.empty:
                continue

            offer_count = offer_row["count"].values[0]
            non_offer_count = non_offer_row["count"].values[0]

            if is_binary:
                # For binary target: calculate difference in rates
                offer_rate = offer_row["rate"].values[0]
                non_offer_rate = non_offer_row["rate"].values[0]

                # Calculate effect size (difference in proportions)
                effect_size = offer_rate - non_offer_rate

                # Calculate statistical significance using proportions z-test
                offer_successes = offer_row["sum"].values[0]
                non_offer_successes = non_offer_row["sum"].values[0]

                counts = np.array([offer_successes, non_offer_successes])
                nobs = np.array([offer_count, non_offer_count])

                try:
                    z_stat, p_value = proportions_ztest(counts, nobs)
                except:
                    # Handle potential errors with small sample sizes
                    p_value = 1.0
            else:
                # For continuous target: calculate difference in means
                offer_mean = offer_row["mean"].values[0]
                non_offer_mean = non_offer_row["mean"].values[0]

                # Calculate effect size (percent difference)
                effect_size = (offer_mean - non_offer_mean) / non_offer_mean

                # We would need to do a t-test, but we don't have raw data, just means
                # So we'll use a simplified approach with the aggregated data
                # For a better approach, would need to do the t-test on raw data
                try:
                    # This is a simplified approximation assuming normal distribution
                    t_stat = (offer_mean - non_offer_mean) / np.sqrt(
                        (offer_mean**2 / offer_count)
                        + (non_offer_mean**2 / non_offer_count)
                    )
                    # 2-sided p-value
                    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
                except:
                    p_value = 1.0

            # Calculate weight for this dimension value
            total_count = offer_count + non_offer_count

            # Store results for this dimension value
            dimension_results.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "effect_size": effect_size,
                    "offer_count": offer_count,
                    "non_offer_count": non_offer_count,
                    "total_count": total_count,
                    "p_value": p_value,
                    "log10_p": -np.log10(p_value) if p_value > 0 else 0,
                }
            )

        # If we have results for this dimension
        if dimension_results:
            # Convert to DataFrame
            dim_df = pd.DataFrame(dimension_results)

            # Calculate overall importance for this dimension
            if weights == "count":
                # Weight by count (proportion of total accounts)
                total_accounts = dim_df["total_count"].sum()
                dim_df["weight"] = dim_df["total_count"] / total_accounts

            elif weights == "effect_size":
                # Weight by absolute effect size
                dim_df["weight"] = np.abs(dim_df["effect_size"])

            elif weights == "statistical":
                # Weight by statistical significance (log p-value)
                dim_df["weight"] = dim_df["log10_p"]

            # Calculate weighted importance
            dim_df["weighted_importance"] = dim_df["weight"] * np.abs(
                dim_df["effect_size"]
            )

            # Aggregate for the dimension
            overall_importance = dim_df["weighted_importance"].sum()
            avg_effect_size = np.abs(dim_df["effect_size"]).mean()

            # Store overall dimension result
            results.append(
                {
                    "dimension": dimension,
                    "importance_score": overall_importance,
                    "avg_effect_size": avg_effect_size,
                    "median_p_value": dim_df["p_value"].median(),
                    "detail": dim_df,
                }
            )

    # Convert to DataFrame and rank by importance
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("importance_score", ascending=False)

    return result_df


def calculate_importance(
    agg_df,
    dimension_col,
    offer_status_col="offer_status",
    metric_col="metric_value",  # Your pre-calculated metric (avg balance or adoption rate)
    count_col="count",  # Number of accounts in each bucket
    weights="statistical",  # How to weight importance: 'count', 'effect_size', or 'statistical'
):
    """
    Calculate importance scores for dimensions explaining differences between offer/non-offer groups

    Parameters:
    -----------
    agg_df : pandas DataFrame
        Pre-aggregated data with dimensions, offer status, metrics and counts
    dimension_col : str
        Column name containing dimension values (e.g., 'age_bucket', 'income_range')
    offer_status_col : str
        Column name containing offer status (should have 'marketing offer' and 'not marketing offer')
    metric_col : str
        Column name containing the pre-calculated metric (avg balance or adoption rate)
    count_col : str
        Column name containing the count of accounts in each bucket
    weights : str
        Weighting method: 'count', 'effect_size', or 'statistical'

    Returns:
    --------
    dict
        Dictionary with importance score, average effect size, and median p-value
    """
    # Get unique dimension values
    dimensions = agg_df[dimension_col].unique()

    results = []

    for dim_value in dimensions:
        # Get data for this dimension value
        dim_df = agg_df[agg_df[dimension_col] == dim_value]

        # Skip if we don't have both offer and non-offer
        if dim_df[offer_status_col].nunique() < 2:
            continue

        # Get offer and non-offer rows
        offer_row = dim_df[dim_df[offer_status_col] == "marketing offer"]
        non_offer_row = dim_df[dim_df[offer_status_col] == "not marketing offer"]

        # Skip if either group is missing
        if offer_row.empty or non_offer_row.empty:
            continue

        # Get values
        offer_value = offer_row[metric_col].values[0]
        non_offer_value = non_offer_row[metric_col].values[0]

        # Get counts
        offer_count = offer_row[count_col].values[0]
        non_offer_count = non_offer_row[count_col].values[0]

        # Calculate effect size (absolute or relative difference depending on metric)
        effect_size = offer_value - non_offer_value

        # For percentage metrics, use absolute difference
        # For monetary values, might want to use percent difference:
        # effect_size = (offer_value - non_offer_value) / non_offer_value

        # Calculate p-value (simplified approach - would need more statistics for better calculation)
        # This assumes normal distribution and is just an approximation
        # For real implementation, would need t-test for means or z-test for proportions
        try:
            # If standard deviation columns are available
            if "std_dev" in dim_df.columns:
                offer_std = offer_row["std_dev"].values[0]
                non_offer_std = non_offer_row["std_dev"].values[0]

                # t-statistic for difference of means
                t_stat = (offer_value - non_offer_value) / np.sqrt(
                    (offer_std**2 / offer_count) + (non_offer_std**2 / non_offer_count)
                )
                # Degrees of freedom (simplified)
                df = offer_count + non_offer_count - 2

                # 2-sided p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            else:
                # Simplified approximation without standard deviations
                # Less accurate but works with just means and counts
                pooled_value = (offer_value + non_offer_value) / 2

                # Estimation of variance based on means and counts
                # This is a rough approximation
                t_stat = (offer_value - non_offer_value) / (
                    pooled_value * np.sqrt((1 / offer_count) + (1 / non_offer_count))
                )

                # Degrees of freedom (simplified)
                df = offer_count + non_offer_count - 2

                # 2-sided p-value
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        except:
            # Default if calculation fails
            p_value = 1.0

        # Calculate total count for this dimension value
        total_count = offer_count + non_offer_count

        # Store results for this dimension value
        results.append(
            {
                "dimension_value": dim_value,
                "effect_size": effect_size,
                "abs_effect_size": abs(effect_size),
                "offer_count": offer_count,
                "non_offer_count": non_offer_count,
                "total_count": total_count,
                "p_value": p_value,
                "log10_p": -np.log10(p_value) if p_value > 0 else 0,
            }
        )

    # If no results, return empty
    if not results:
        return {"importance_score": 0, "avg_effect_size": 0, "median_p_value": 1.0}

    # Convert to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)

    # Calculate weights based on specified method
    if weights == "count":
        # Weight by count (proportion of total accounts)
        total_accounts = results_df["total_count"].sum()
        results_df["weight"] = results_df["total_count"] / total_accounts
    elif weights == "effect_size":
        # Weight by absolute effect size
        results_df["weight"] = (
            results_df["abs_effect_size"] / results_df["abs_effect_size"].sum()
        )
    elif weights == "statistical":
        # Weight by statistical significance (log p-value)
        if results_df["log10_p"].sum() > 0:
            results_df["weight"] = results_df["log10_p"] / results_df["log10_p"].sum()
        else:
            results_df["weight"] = 1 / len(results_df)

    # Calculate weighted importance
    results_df["weighted_importance"] = (
        results_df["weight"] * results_df["abs_effect_size"]
    )

    # Calculate overall metrics
    importance_score = results_df["weighted_importance"].sum()
    avg_effect_size = results_df["abs_effect_size"].mean()
    median_p_value = results_df["p_value"].median()

    return {
        "importance_score": importance_score,
        "avg_effect_size": avg_effect_size,
        "median_p_value": median_p_value,
        "detail": results_df,  # Optional: include detailed breakdown by dimension value
    }
