"""
Advanced interestingness metrics.

This module contains more sophisticated metrics that can help identify
complex patterns, anomalies, and interestingness in aggregated data.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_aggregated, normalize_weights


def gini_impurity(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
) -> Dict[str, float]:
    """
    Calculate Gini impurity for each group, which measures how mixed or "impure"
    the values are within each segment.

    Lower Gini impurity indicates more homogeneity within a group.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    count_col : str
        Column name with the count per group

    Returns:
    --------
    dict
        Dictionary with group values as keys and Gini impurity scores as values
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # We need to discretize the target for Gini calculation if it's continuous
    n_bins = min(10, len(agg_df) // 5)  # Rule of thumb: at least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(
        agg_df[target_col], q=n_bins, duplicates="drop", labels=False
    )

    # Calculate Gini impurity for each group
    result = {}

    for _, group in agg_df.groupby(group_cols):
        # Get group key
        group_key = []
        for col in group_cols:
            group_key.append(str(group[col].iloc[0]))
        group_name = "__".join(group_key)

        # Get counts for each bin within this group
        bin_counts = group.groupby("target_binned")[count_col].sum()
        total_count = bin_counts.sum()

        # Calculate probabilities
        if total_count > 0:
            probs = bin_counts / total_count

            # Calculate Gini impurity: sum(p * (1 - p))
            gini = sum(p * (1 - p) for p in probs)
            result[group_name] = gini

    return result


def cramers_v(
    df: pd.DataFrame, target_col: str, group_col: str, count_col: str = "count"
) -> float:
    """
    Calculate Cramér's V statistic, which measures the association
    between two categorical variables (similar to correlation for categorical data).

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_col : str
        Column name for the grouping variable
    count_col : str
        Column name with the count per group

    Returns:
    --------
    float
        Cramér's V statistic (0 to 1)
    """
    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # We need to discretize the target for contingency table if it's continuous
    n_bins = min(10, len(agg_df) // 5)  # Rule of thumb: at least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(
        agg_df[target_col], q=n_bins, duplicates="drop", labels=False
    )

    # Create contingency table
    contingency = pd.crosstab(
        agg_df[group_col],
        agg_df["target_binned"],
        values=agg_df[count_col],
        aggfunc="sum",
    ).fillna(0)

    # Calculate chi-squared statistic
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    # Calculate Cramér's V
    n = contingency.sum().sum()

    # Get number of rows and columns for correction
    r, k = contingency.shape

    # Calculate Cramér's V with correction
    v = (
        np.sqrt(chi2 / (n * min(r - 1, k - 1)))
        if n > 0 and min(r - 1, k - 1) > 0
        else 0
    )

    return v


def theil_index(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
) -> Dict[str, Dict[str, float]]:
    """
    Calculate Theil's inequality index, which measures inequality or dispersion
    within and between groups.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    count_col : str
        Column name with the count per group

    Returns:
    --------
    dict
        Dictionary with results containing:
        - 'total': Total Theil index
        - 'within': Within-group component
        - 'between': Between-group component
        - 'groups': Theil indices for each group
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # Ensure values are positive (Theil index requires positive values)
    if (agg_df[target_col] <= 0).any():
        # Shift values to ensure all are positive
        min_val = agg_df[target_col].min()
        if min_val <= 0:
            shift = abs(min_val) + 1  # Add 1 to ensure all values are positive
            agg_df[target_col] = agg_df[target_col] + shift

    # Calculate overall mean
    weights = normalize_weights(agg_df, count_col)
    overall_mean = np.average(agg_df[target_col], weights=weights)

    # Calculate total Theil index
    if overall_mean <= 0:
        return {"total": 0, "within": 0, "between": 0, "groups": {}}

    # Total Theil = average of (x_i/mean) * log(x_i/mean)
    theil_values = (agg_df[target_col] / overall_mean) * np.log(
        agg_df[target_col] / overall_mean
    )
    total_theil = np.average(theil_values, weights=weights)

    # Calculate within-group and between-group components
    group_theils = {}
    within_component = 0
    between_component = 0

    for _, group in agg_df.groupby(group_cols):
        # Get group key
        group_key = []
        for col in group_cols:
            group_key.append(str(group[col].iloc[0]))
        group_name = "__".join(group_key)

        # Calculate group statistics
        group_weights = normalize_weights(group, count_col)
        group_mean = np.average(group[target_col], weights=group_weights)

        # Within-group Theil
        if group_mean > 0:
            group_theil_values = (group[target_col] / group_mean) * np.log(
                group[target_col] / group_mean
            )
            group_theil = np.average(group_theil_values, weights=group_weights)
        else:
            group_theil = 0

        # Store group Theil
        group_theils[group_name] = group_theil

        # Calculate contribution to within-group component
        group_weight = group[count_col].sum() / agg_df[count_col].sum()
        within_component += group_weight * group_theil

        # Calculate contribution to between-group component
        if overall_mean > 0 and group_mean > 0:
            between_term = (group_mean / overall_mean) * np.log(
                group_mean / overall_mean
            )
            between_component += group_weight * between_term

    return {
        "total": total_theil,
        "within": within_component,
        "between": between_component,
        "groups": group_theils,
    }


def concentration_ratio(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    top_n: int = 3,
    count_col: str = "count",
) -> Dict[str, float]:
    """
    Calculate concentration ratio, which measures how concentrated the values are
    among the top N groups. Similar to market concentration.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    top_n : int
        Number of top groups to consider
    count_col : str
        Column name with the count per group

    Returns:
    --------
    dict
        Dictionary with results including concentration ratios and Herfindahl index
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # Calculate total metric sum weighted by count
    weighted_values = agg_df[target_col] * agg_df[count_col]
    total_weighted_sum = weighted_values.sum()

    if total_weighted_sum == 0:
        return {f"CR{top_n}": 0, "herfindahl": 0}

    # Calculate weighted sum for each group
    group_sums = agg_df.groupby(group_cols).apply(
        lambda x: (x[target_col] * x[count_col]).sum()
    )

    # Calculate market shares
    market_shares = group_sums / total_weighted_sum

    # Sort market shares in descending order
    sorted_shares = market_shares.sort_values(ascending=False)

    # Calculate concentration ratio for top N groups
    top_n = min(top_n, len(sorted_shares))
    concentration = sorted_shares.iloc[:top_n].sum()

    # Calculate Herfindahl index (sum of squared market shares)
    herfindahl = (market_shares**2).sum()

    return {f"CR{top_n}": concentration, "herfindahl": herfindahl}


def anomaly_score(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    method: str = "combined",
    count_col: str = "count",
    std_col: Optional[str] = "std",
) -> pd.DataFrame:
    """
    Calculate a combined anomaly score for each group based on multiple metrics.

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    method : str
        Method to use for anomaly scoring: 'combined', 'z_score', 'iqr', or 'isolation_forest'
    count_col : str
        Column name with the count per group
    std_col : str, optional
        Column name with the standard deviation per group

    Returns:
    --------
    DataFrame
        DataFrame with groups and their anomaly scores
    """
    from .distribution import distribution_outliers, kl_divergence

    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col, std_col=std_col)

    # For the combined approach, we'll use multiple metrics
    if method == "combined":
        # Get distribution outliers
        outliers_df = distribution_outliers(
            agg_df,
            target_col,
            group_cols,
            method="z_score",
            threshold=0,
            count_col=count_col,
        )

        # If we got empty results, fall back to z-score method
        if len(outliers_df) == 0:
            return anomaly_score(
                df,
                target_col,
                group_cols,
                method="z_score",
                count_col=count_col,
                std_col=std_col,
            )

        # Get KL divergence for each group
        if len(group_cols) == 1:
            kl_div = kl_divergence(
                agg_df, target_col, group_cols[0], count_col=count_col
            )

            # Normalize KL divergence to 0-1 scale
            max_kl = max(kl_div.values()) if kl_div else 0
            kl_scores = {k: v / max_kl if max_kl > 0 else 0 for k, v in kl_div.items()}

            # Add KL divergence to outliers DataFrame
            outliers_df["kl_score"] = outliers_df[group_cols[0]].apply(
                lambda x: kl_scores.get(str(x), 0)
            )
        else:
            # For multiple grouping columns, we don't calculate KL divergence
            outliers_df["kl_score"] = 0

        # Calculate final anomaly score (combination of outlier score and KL divergence)
        outliers_df["anomaly_score"] = (
            outliers_df["outlier_score"] + outliers_df["kl_score"]
        ) / 2

        # Sort by anomaly score
        return outliers_df.sort_values("anomaly_score", ascending=False)

    elif method == "isolation_forest":
        # We'll implement a simplified version that works with aggregated data
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "scikit-learn is required for isolation_forest method. "
                "Install it with 'pip install scikit-learn'."
            )

        # First, we need to create a feature matrix from the grouping variables
        # We'll use one-hot encoding for categorical variables
        features = pd.get_dummies(agg_df[group_cols], drop_first=False)

        # Add the target column
        features[target_col] = agg_df[target_col]

        # Train an isolation forest model
        model = IsolationForest(contamination=0.1, random_state=42)

        # Apply weights based on counts
        weights = agg_df[count_col].values / agg_df[count_col].sum()

        # Fit the model and get anomaly scores
        scores = model.fit_predict(features)

        # Convert scores to a 0-1 range where higher means more anomalous
        # Isolation Forest returns -1 for outliers and 1 for inliers
        normalized_scores = np.where(scores == -1, 1, 0)

        # Add the scores to a new DataFrame
        result_df = agg_df[group_cols + [target_col, count_col]].copy()
        result_df["anomaly_score"] = normalized_scores

        return result_df.sort_values("anomaly_score", ascending=False)

    else:
        # For z_score and iqr methods, use distribution_outliers function
        outliers_df = distribution_outliers(
            agg_df,
            target_col,
            group_cols,
            method=method,
            threshold=0,
            count_col=count_col,
        )

        # Rename column for consistency
        outliers_df = outliers_df.rename(columns={"outlier_score": "anomaly_score"})

        return outliers_df.sort_values("anomaly_score", ascending=False)


def information_gain(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
) -> Dict[str, float]:
    """
    Calculate information gain, which measures how much a variable reduces entropy
    in the target variable (common in decision trees).

    Parameters:
    -----------
    df : DataFrame
        Aggregated DataFrame with grouping variables, target metric, and counts
    target_col : str
        Column name containing the target metric (e.g., mean values)
    group_cols : str or list of str
        Column name(s) for the grouping variables
    count_col : str
        Column name with the count per group

    Returns:
    --------
    dict
        Dictionary with grouping variables as keys and information gain as values
    """

    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # We need to discretize the target for entropy calculation if it's continuous
    n_bins = min(10, len(agg_df) // 5)  # Rule of thumb: at least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(
        agg_df[target_col], q=n_bins, duplicates="drop", labels=False
    )

    # Calculate entropy of the target before splitting
    counts = agg_df.groupby("target_binned")[count_col].sum()
    total_count = counts.sum()

    if total_count == 0:
        return {col: 0.0 for col in group_cols}

    probs = counts / total_count
    entropy_before = stats.entropy(probs)

    # Calculate information gain for each variable
    result = {}

    for col in group_cols:
        # Calculate weighted entropy after splitting by this variable
        weighted_entropy = 0

        for val, group in agg_df.groupby(col):
            group_count = group[count_col].sum()
            weight = group_count / total_count

            # Calculate entropy for this group
            group_counts = group.groupby("target_binned")[count_col].sum()
            group_probs = group_counts / group_count if group_count > 0 else 0

            if isinstance(group_probs, pd.Series) and len(group_probs) > 0:
                group_entropy = stats.entropy(group_probs)
            else:
                group_entropy = 0

            weighted_entropy += weight * group_entropy

        # Information gain = entropy before - weighted entropy after
        info_gain = entropy_before - weighted_entropy
        result[col] = info_gain

    return result
