"""
Example usage of the interestingness toolkit.

This script demonstrates how to use the interestingness package
to analyze patterns in a sample dataset.
"""

import os
import sys

import numpy as np
import pandas as pd

# Add the parent directory to the path so we can import the interestingness package
# This is only needed for the example - when installed, the package would be available normally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions from the interestingness package
from interestingness.contribution import (
    feature_importance,
    mutual_information,
    variance_explained,
)
from interestingness.distribution import (
    distribution_outliers,
    entropy_score,
    kl_divergence,
)
from interestingness.effect_size import (
    cohens_d,
    percent_difference,
    standardized_difference,
)


def create_sample_data():
    """
    Create a sample aggregated dataset for demonstration purposes.

    Returns:
    --------
    DataFrame
        A pandas DataFrame with synthetic data for analysis
    """
    # Define the dimensions and values
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    channels = ["Online", "Branch", "Phone"]
    wealth_tiers = ["Low", "Medium", "High"]

    # Create all combinations of dimensions
    combinations = []
    for age in age_groups:
        for channel in channels:
            for wealth in wealth_tiers:
                combinations.append(
                    {"age_group": age, "channel": channel, "wealth_tier": wealth}
                )

    # Convert to DataFrame
    df = pd.DataFrame(combinations)

    # Add synthetic data
    # Here we'll create patterns where:
    # - Age strongly affects balance (older = higher balance)
    # - Wealth tier affects balance (higher tier = higher balance)
    # - Channel has a smaller effect, with Branch having slightly higher balances
    # - There's an interaction between age and channel (older people use Branch more)

    # Add count (sample size) - varies by segment
    np.random.seed(42)  # For reproducibility

    # Base counts
    df["count"] = np.random.randint(50, 200, len(df))

    # Age effect on count: more middle-aged customers
    age_effect = {"18-24": 0.7, "25-34": 1.2, "35-44": 1.5, "45-54": 1.3, "55+": 0.9}
    df["count"] = df.apply(
        lambda x: int(x["count"] * age_effect[x["age_group"]]), axis=1
    )

    # Channel effect on count: more online customers
    channel_effect = {"Online": 1.4, "Branch": 0.8, "Phone": 0.6}
    df["count"] = df.apply(
        lambda x: int(x["count"] * channel_effect[x["channel"]]), axis=1
    )

    # Create mean balances
    # Base balance
    df["balance_mean"] = 5000

    # Age effect on balance: older customers have higher balances
    age_balance_effect = {
        "18-24": 0.5,
        "25-34": 0.8,
        "35-44": 1.2,
        "45-54": 1.5,
        "55+": 2.0,
    }
    df["balance_mean"] = df.apply(
        lambda x: x["balance_mean"] * age_balance_effect[x["age_group"]], axis=1
    )

    # Wealth tier effect on balance
    wealth_balance_effect = {"Low": 0.6, "Medium": 1.0, "High": 2.5}
    df["balance_mean"] = df.apply(
        lambda x: x["balance_mean"] * wealth_balance_effect[x["wealth_tier"]], axis=1
    )

    # Channel effect on balance: Branch has higher balances
    channel_balance_effect = {"Online": 0.9, "Branch": 1.2, "Phone": 1.0}
    df["balance_mean"] = df.apply(
        lambda x: x["balance_mean"] * channel_balance_effect[x["channel"]], axis=1
    )

    # Interaction effect: older people in Branch have even higher balances
    df["balance_mean"] = df.apply(
        lambda x: (
            x["balance_mean"] * 1.3
            if x["age_group"] in ["45-54", "55+"] and x["channel"] == "Branch"
            else x["balance_mean"]
        ),
        axis=1,
    )

    # Add some random variation
    df["balance_mean"] = df["balance_mean"] * np.random.normal(1, 0.1, len(df))

    # Add standard deviation (approximately 30% of the mean)
    df["balance_std"] = df["balance_mean"] * 0.3

    # Add a digital adoption percent (another metric to analyze)
    # Base adoption rate
    df["digital_adoption"] = 0.7

    # Age effect on adoption: younger customers have higher adoption
    age_adoption_effect = {
        "18-24": 1.3,
        "25-34": 1.2,
        "35-44": 1.0,
        "45-54": 0.8,
        "55+": 0.6,
    }
    df["digital_adoption"] = df.apply(
        lambda x: min(
            0.95, x["digital_adoption"] * age_adoption_effect[x["age_group"]]
        ),
        axis=1,
    )

    # Channel effect on adoption: Online customers have higher adoption
    channel_adoption_effect = {"Online": 1.2, "Branch": 0.8, "Phone": 0.9}
    df["digital_adoption"] = df.apply(
        lambda x: min(
            0.95, x["digital_adoption"] * channel_adoption_effect[x["channel"]]
        ),
        axis=1,
    )

    # Add some random variation
    df["digital_adoption"] = df.apply(
        lambda x: min(
            0.95, max(0.1, x["digital_adoption"] * np.random.normal(1, 0.05))
        ),
        axis=1,
    )

    # Round and format for readability
    df["balance_mean"] = df["balance_mean"].round(2)
    df["balance_std"] = df["balance_std"].round(2)
    df["digital_adoption"] = df["digital_adoption"].round(4)

    return df


def main():
    """
    Main function to demonstrate interestingness metrics.
    """
    print("Creating sample aggregated data...")
    df = create_sample_data()

    print("\nSample data preview:")
    print(df.head())

    print("\n" + "=" * 80)
    print("CONTRIBUTION-BASED METRICS")
    print("=" * 80)

    # Variance explained analysis
    print("\n1. Variance Explained for Balance Mean:")
    group_cols = ["age_group", "channel", "wealth_tier"]
    var_explained = variance_explained(
        df, "balance_mean", group_cols, count_col="count"
    )

    # Sort by variance explained and print
    var_explained = {
        k: v for k, v in sorted(var_explained.items(), key=lambda x: x[1], reverse=True)
    }
    for group, score in var_explained.items():
        print(f"  {group}: {score:.4f} ({score*100:.1f}%)")

    # Feature importance
    print("\n2. Feature Importance for Balance Mean:")
    importance = feature_importance(df, "balance_mean", group_cols, count_col="count")
    print(importance)

    # Mutual information
    print("\n3. Mutual Information for Balance Mean:")
    mi_scores = mutual_information(df, "balance_mean", group_cols, count_col="count")
    for group, score in sorted(mi_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {score:.4f}")

    print("\n" + "=" * 80)
    print("DISTRIBUTION-BASED METRICS")
    print("=" * 80)

    # Entropy scores
    print("\n4. Entropy Scores for Age Groups:")
    entropy = entropy_score(df, "balance_mean", "age_group", count_col="count")
    for group, score in sorted(entropy.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {score:.4f}")

    # KL Divergence
    print("\n5. KL Divergence (compared to overall distribution):")
    kl_div = kl_divergence(df, "balance_mean", "age_group", count_col="count")
    for group, score in sorted(kl_div.items(), key=lambda x: x[1], reverse=True):
        print(f"  {group}: {score:.4f}")

    # Distribution outliers
    print("\n6. Distribution Outliers (z-score method):")
    outliers = distribution_outliers(
        df, "balance_mean", ["age_group", "channel"], method="z_score", threshold=1.5
    )
    print(outliers[["age_group", "channel", "mean", "outlier_score", "direction"]])

    print("\n" + "=" * 80)
    print("EFFECT SIZE METRICS")
    print("=" * 80)

    # Cohen's d
    print("\n7. Cohen's d for Age Groups (compared to overall):")
    for age in df["age_group"].unique():
        d = cohens_d(
            df,
            "balance_mean",
            "age_group",
            age,
            count_col="count",
            std_col="balance_std",
        )
        interpretation = (
            "very large"
            if abs(d) > 1.2
            else "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        )
        print(f"  {age} vs others: d = {d:.2f} ({interpretation})")

    # Standardized differences
    print("\n8. Largest Standardized Differences:")
    std_diff = standardized_difference(
        df,
        "balance_mean",
        ["age_group", "wealth_tier"],
        count_col="count",
        std_col="balance_std",
    )
    print(std_diff[["group_var", "value1", "value2", "mean_diff", "std_diff"]].head(5))

    # Percent differences
    print("\n9. Percent Differences from Average:")
    pct_diff = percent_difference(
        df, "balance_mean", ["channel", "wealth_tier"], count_col="count"
    )
    print(
        pct_diff[["channel", "wealth_tier", "mean", "reference", "percent_diff"]].head(
            5
        )
    )

    # Digital adoption analysis
    print("\n" + "=" * 80)
    print("DIGITAL ADOPTION ANALYSIS")
    print("=" * 80)

    print("\n10. Feature Importance for Digital Adoption:")
    adoption_importance = feature_importance(
        df, "digital_adoption", group_cols, count_col="count"
    )
    print(adoption_importance)

    print("\n11. Largest Percent Differences in Digital Adoption:")
    adoption_pct_diff = percent_difference(
        df, "digital_adoption", ["age_group"], count_col="count"
    )
    print(adoption_pct_diff[["age_group", "mean", "percent_diff"]])


if __name__ == "__main__":
    main()
