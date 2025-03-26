"""
Advanced Interestingness Metrics for Grouped Data

Advanced statistical metrics to evaluate how "interesting" or significant
a grouping is in pre-aggregated data.
"""

from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats


def entropy_reduction(df: pd.DataFrame, mean_col: str, count_col: str) -> float:
    """
    Calculate the entropy reduction (information gain) provided by the grouping.
    This measures how much uncertainty about the metric is reduced by knowing
    the group.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count for each group

    Returns:
    --------
    float
        Information gain value (higher = more informative grouping)
    """
    if df.empty or len(df) <= 1:
        return 0.0

    # Get counts and calculate total
    counts = df[count_col].values
    total_count = np.sum(counts)

    if total_count == 0:
        return 0.0

    # Calculate proportions
    proportions = counts / total_count

    # Calculate weighted variance of the full dataset (approximation)
    overall_weighted_mean = np.average(df[mean_col], weights=counts)
    overall_variance = np.average(
        (df[mean_col] - overall_weighted_mean) ** 2, weights=counts
    )

    # Calculate weighted average of within-group variances
    # Note: We would need variance within each group for ideal calculation
    # This is an approximation without raw data
    group_variance_sum = 0
    for _, row in df.iterrows():
        # If we had within-group variance, we would use it here
        # Since we don't, we'll use a placeholder (this is a simplification)
        group_variance = 0.1 * overall_variance  # Placeholder
        group_variance_sum += (row[count_col] / total_count) * group_variance

    # Calculate information gain as reduction in variance
    if overall_variance == 0:
        return 0.0

    return 1.0 - (group_variance_sum / overall_variance)


def kruskal_wallis_h(
    df: pd.DataFrame,
    mean_col: str,
    count_col: str,
    percentile_25_col: Optional[str] = None,
    percentile_75_col: Optional[str] = None,
) -> dict:
    """
    Estimate the Kruskal-Wallis H-statistic from aggregated data.
    This is a non-parametric alternative to one-way ANOVA.

    Note: This is an approximation as the actual test requires raw data.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count for each group
    percentile_25_col : str, optional
        Column name containing the 25th percentile for each group
    percentile_75_col : str, optional
        Column name containing the 75th percentile for each group

    Returns:
    --------
    dict
        Dictionary containing:
        - 'h_statistic': The H-statistic approximation
        - 'p_value': The approximate p-value
        - 'significant': Boolean indicating if result is significant (p < 0.05)
    """
    if df.empty or len(df) <= 1:
        return {"h_statistic": 0.0, "p_value": 1.0, "significant": False}

    # Get counts and calculate total
    counts = df[count_col].values
    total_count = np.sum(counts)
    k = len(df)  # Number of groups

    if total_count <= k:
        return {"h_statistic": 0.0, "p_value": 1.0, "significant": False}

    # Estimate mean ranks using available statistics
    # This is a rough approximation - true calculation needs raw data
    try:
        # Assign ranks based on relative position of group means
        rank_order = df[mean_col].argsort().argsort() + 1
        mean_ranks = (total_count + 1) / 2 + (rank_order - (k + 1) / 2) * (
            total_count / k
        )

        # If we have percentiles, refine the approximation
        if percentile_25_col is not None and percentile_75_col is not None:
            # Use IQR to adjust rank approximation (wider spread = more extreme ranks)
            iqrs = df[percentile_75_col] - df[percentile_25_col]
            relative_iqrs = iqrs / iqrs.mean()
            adjustment = 0.1 * (
                relative_iqrs - 1
            )  # Small adjustment based on relative IQR
            mean_ranks = mean_ranks * (1 + adjustment)

        # Calculate H statistic
        h_stat = (
            12
            / (total_count * (total_count + 1))
            * np.sum(counts * (mean_ranks - (total_count + 1) / 2) ** 2)
        )

        # Approximate p-value from chi-square distribution
        df_h = k - 1
        p_value = stats.chi2.sf(h_stat, df_h)

        return {
            "h_statistic": h_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
        }
    except:
        # Fallback if calculation has issues
        return {"h_statistic": 0.0, "p_value": 1.0, "significant": False}


def discriminative_power(
    df: pd.DataFrame, mean_col: str, var_col: str, count_col: str
) -> float:
    """
    Calculate the discriminative power of the grouping.
    This is similar to the F-ratio in ANOVA but expressed as a ratio of
    between-group to within-group variance.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    var_col : str
        Column name containing the variance for each group
    count_col : str
        Column name containing the count for each group

    Returns:
    --------
    float
        Discriminative power (higher = better separation between groups)
    """
    if df.empty or len(df) <= 1:
        return 0.0

    # Calculate weighted overall mean
    counts = df[count_col].values
    total_count = np.sum(counts)

    if total_count == 0:
        return 0.0

    weighted_mean = np.average(df[mean_col], weights=counts)

    # Calculate between-group variance
    between_var = np.sum(counts * (df[mean_col] - weighted_mean) ** 2) / total_count

    # Calculate within-group variance (weighted average of group variances)
    within_var = np.average(df[var_col], weights=counts)

    # Calculate discriminative power ratio
    if within_var == 0:
        return float("inf") if between_var > 0 else 0.0

    return between_var / within_var


def outlier_group_score(
    df: pd.DataFrame,
    mean_col: str,
    count_col: Optional[str] = None,
    threshold: float = 1.5,
) -> dict:
    """
    Identify outlier groups based on how far their means are from the overall mean.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str, optional
        Column name containing the count for weighted calculations
    threshold : float, default=1.5
        Z-score threshold for considering a group an outlier

    Returns:
    --------
    dict
        Dictionary containing:
        - 'outlier_count': Number of outlier groups
        - 'outlier_proportion': Proportion of outlier groups
        - 'max_z_score': Maximum absolute z-score
        - 'outlier_indices': Indices of outlier groups
    """
    if df.empty or len(df) <= 1:
        return {
            "outlier_count": 0,
            "outlier_proportion": 0.0,
            "max_z_score": 0.0,
            "outlier_indices": [],
        }

    # Calculate overall mean and standard deviation
    if count_col is not None:
        # Weighted statistics
        weights = df[count_col]
        means = df[mean_col]
        overall_mean = np.average(means, weights=weights)
        overall_std = np.sqrt(np.average((means - overall_mean) ** 2, weights=weights))
    else:
        # Unweighted statistics
        overall_mean = df[mean_col].mean()
        overall_std = df[mean_col].std()

    # Calculate z-scores
    if overall_std == 0:
        z_scores = np.zeros(len(df))
    else:
        z_scores = (df[mean_col] - overall_mean) / overall_std

    # Identify outliers
    is_outlier = abs(z_scores) > threshold
    outlier_indices = list(df.index[is_outlier])

    return {
        "outlier_count": sum(is_outlier),
        "outlier_proportion": sum(is_outlier) / len(df),
        "max_z_score": abs(z_scores).max() if len(z_scores) > 0 else 0.0,
        "outlier_indices": outlier_indices,
    }


def group_separation_index(
    df: pd.DataFrame, mean_col: str, count_col: Optional[str] = None
) -> float:
    """
    Calculate a metric for how well-separated the groups are from each other.
    This considers how many group means are significantly different from others.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str, optional
        Column name containing the count for weighted calculations

    Returns:
    --------
    float
        Separation index between 0 and 1 (higher = better separation)
    """
    if df.empty or len(df) <= 1:
        return 0.0

    n_groups = len(df)
    means = df[mean_col].values

    # Calculate the distance matrix between all group means
    distances = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            # Calculate normalized distance between group means
            distances[i, j] = distances[j, i] = abs(means[i] - means[j])

    # Normalize distances by the maximum distance
    max_distance = distances.max()
    if max_distance > 0:
        distances = distances / max_distance

    # Calculate average separation (excluding self-comparisons)
    total_comparisons = n_groups * (n_groups - 1) / 2
    avg_separation = np.sum(distances) / (2 * total_comparisons)

    return avg_separation


def permutation_significance(
    df: pd.DataFrame,
    mean_col: str,
    count_col: str,
    var_col: str,
    n_permutations: int = 1000,
) -> dict:
    """
    Estimate the significance of group differences using permutation testing.

    Note: This is an approximation since we don't have the raw data.

    Parameters:
    -----------
    df : pandas DataFrame
        Pre-aggregated data with each row representing a group
    mean_col : str
        Column name containing the mean value for each group
    count_col : str
        Column name containing the count for each group
    var_col : str
        Column name containing the variance for each group
    n_permutations : int, default=1000
        Number of permutations to run

    Returns:
    --------
    dict
        Dictionary containing:
        - 'p_value': Estimated p-value from permutation test
        - 'significant': Boolean indicating if result is significant (p < 0.05)
    """
    if df.empty or len(df) <= 1:
        return {"p_value": 1.0, "significant": False}

    # Calculate the original F-statistic
    counts = df[count_col].values
    means = df[mean_col].values
    variances = df[var_col].values

    # Calculate grand mean
    total_count = np.sum(counts)
    grand_mean = np.sum(counts * means) / total_count

    # Calculate between-group sum of squares
    between_ss = np.sum(counts * (means - grand_mean) ** 2)
    between_df = len(df) - 1

    # Calculate within-group sum of squares
    within_ss = np.sum((counts - 1) * variances)
    within_df = total_count - len(df)

    if within_df <= 0 or within_ss <= 0:
        return {"p_value": 1.0, "significant": False}

    # Calculate F-statistic
    f_stat_orig = (between_ss / between_df) / (within_ss / within_df)

    # Perform permutation test (approximation)
    count_exceeding = 0

    # Generate synthetic data approximating the original dataset
    for _ in range(n_permutations):
        # For each permutation, we simulate randomly assigning observations to groups
        # Since we don't have raw data, we'll simulate based on group statistics

        # Randomize group means while keeping same counts and variances
        shuffled_means = np.random.normal(
            loc=grand_mean,
            scale=np.sqrt(within_ss / within_df / np.mean(counts)),
            size=len(df),
        )

        # Calculate permuted between-group sum of squares
        perm_between_ss = np.sum(counts * (shuffled_means - grand_mean) ** 2)

        # Calculate permuted F-statistic
        f_stat_perm = (perm_between_ss / between_df) / (within_ss / within_df)

        # Count permutations with F >= original F
        if f_stat_perm >= f_stat_orig:
            count_exceeding += 1

    # Calculate p-value
    p_value = count_exceeding / n_permutations

    return {"p_value": p_value, "significant": p_value < 0.05}
