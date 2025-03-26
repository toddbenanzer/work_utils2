"""
Contribution-based interestingness metrics.

This module contains functions that measure how much each grouping variable
contributes to the variation in the target metric.
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ensure_aggregated, normalize_weights, weighted_variance


def variance_explained(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
) -> Dict[str, float]:
    """
    Calculate the proportion of variance explained by each grouping variable
    and their combinations.

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
        Dictionary with grouping variables and combinations as keys and
        proportion of variance explained as values
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # Calculate weights (proportions) from counts
    weights = normalize_weights(agg_df, count_col)

    # Calculate total weighted variance of the target
    total_variance = weighted_variance(agg_df[target_col].values, weights.values)

    if total_variance == 0 or np.isnan(total_variance):
        # No variance to explain
        return {col: 0.0 for col in group_cols}

    # Calculate variance explained for each grouping variable and combinations
    result = {}

    # Generate all combinations of grouping variables
    from itertools import combinations

    all_combinations = []
    for i in range(1, len(group_cols) + 1):
        all_combinations.extend(combinations(group_cols, i))

    for combo in all_combinations:
        combo_cols = list(combo)
        combo_name = " + ".join(combo_cols)

        # Calculate between-group variance
        group_means = agg_df.groupby(combo_cols)[target_col].apply(
            lambda x: weighted_mean(x.values, agg_df.loc[x.index, count_col].values)
        )

        # The variance explained is the weighted variance of these group means
        combo_weights = agg_df.groupby(combo_cols)[count_col].sum()
        combo_weights = combo_weights / combo_weights.sum()

        between_variance = weighted_variance(group_means.values, combo_weights.values)
        var_explained = between_variance / total_variance

        result[combo_name] = min(1.0, max(0.0, var_explained))  # Bound between 0 and 1

    return result


def feature_importance(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
    include_interactions: bool = True,
) -> pd.DataFrame:
    """
    Calculate the importance of each grouping variable in explaining
    the target metric variation. This is similar to feature importance
    in machine learning models.

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
    include_interactions : bool
        Whether to include interaction terms between variables

    Returns:
    --------
    DataFrame
        DataFrame with variables and their importance scores
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Get variance explained for each variable and combination
    var_explained = variance_explained(df, target_col, group_cols, count_col)

    # For feature importance, we can either:
    # 1. Just use the individual variable variance explained
    # 2. Calculate marginal importance by comparing full model with one-variable-removed models

    result = []

    # Add individual variable importance
    for col in group_cols:
        result.append(
            {
                "variable": col,
                "type": "main_effect",
                "importance": var_explained.get(col, 0.0),
            }
        )

    # Add interactions if requested
    if include_interactions and len(group_cols) > 1:
        from itertools import combinations

        for i in range(2, len(group_cols) + 1):
            for combo in combinations(group_cols, i):
                combo_name = " + ".join(combo)

                # Calculate importance as additional variance explained beyond individual variables
                individual_vars = sum(var_explained.get(col, 0.0) for col in combo)
                interaction_importance = (
                    var_explained.get(combo_name, 0.0) - individual_vars
                )
                interaction_importance = max(
                    0.0, interaction_importance
                )  # Ensure non-negative

                result.append(
                    {
                        "variable": combo_name,
                        "type": "interaction",
                        "importance": interaction_importance,
                    }
                )

    return pd.DataFrame(result).sort_values("importance", ascending=False)


def mutual_information(
    df: pd.DataFrame,
    target_col: str,
    group_cols: Union[str, List[str]],
    count_col: str = "count",
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Calculate the mutual information between each grouping variable
    and the target metric, which measures the amount of information
    obtained about one variable by observing the other.

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
    normalize : bool
        Whether to normalize the mutual information score (0-1 scale)

    Returns:
    --------
    dict
        Dictionary with grouping variables as keys and mutual information as values
    """
    # Ensure we have a list of columns
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    # Make a copy to work with
    agg_df = ensure_aggregated(df, target_col, count_col)

    # For mutual information with a continuous target, we need to discretize it
    # Let's use quantiles to bin the target
    n_bins = min(10, len(agg_df) // 5)  # Rule of thumb: at least 5 samples per bin
    n_bins = max(2, n_bins)  # At least 2 bins

    agg_df["target_binned"] = pd.qcut(agg_df[target_col], q=n_bins, duplicates="drop")

    # Calculate mutual information for each variable
    result = {}

    for col in group_cols:
        # We need to convert counts to actual observations for mutual information
        # Create a contingency table
        contingency = pd.crosstab(
            agg_df[col],
            agg_df["target_binned"],
            values=agg_df[count_col],
            aggfunc="sum",
        ).fillna(0)

        mi_score = stats.mutual_info_score(None, None, contingency=contingency.values)

        if normalize and mi_score > 0:
            # Normalize by entropy of the target
            target_entropy = stats.entropy(
                contingency.sum(axis=0) / contingency.sum().sum()
            )
            if target_entropy > 0:
                mi_score = mi_score / target_entropy

        result[col] = mi_score

    return result
