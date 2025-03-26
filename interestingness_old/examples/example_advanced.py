"""
Advanced usage examples for the interesting-ness toolkit.

This script demonstrates the more sophisticated metrics
available in the advanced module.
"""

import os
import sys

# Add the parent directory to the path so we can import the interesting-ness package
# This is only needed for the example - when installed, the package would be available normally
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the basic_usage module to get the sample data function
from basic_usage import create_sample_data

# Import advanced functions from the interesting-ness package
from interestingness_old.advanced import (
    anomaly_score,
    concentration_ratio,
    cramers_v,
    gini_impurity,
    information_gain,
    theil_index,
)


def main():
    """
    Main function to demonstrate advanced interestingness metrics.
    """
    print("Creating sample aggregated data...")
    df = create_sample_data()

    print("\nSample data preview:")
    print(df.head())

    print("\n" + "=" * 80)
    print("ADVANCED METRICS")
    print("=" * 80)

    # Gini Impurity
    print("\n1. Gini Impurity for Age Groups:")
    gini = gini_impurity(df, "balance_mean", "age_group", count_col="count")
    for group, score in sorted(gini.items(), key=lambda x: x[1]):
        print(f"  {group}: {score:.4f}")

    # Cramér's V
    print("\n2. Cramér's V (Association Strength):")
    for col in ["age_group", "channel", "wealth_tier"]:
        v = cramers_v(df, "balance_mean", col, count_col="count")
        print(f"  Association between balance and {col}: {v:.4f}")

        # Interpretation
        if v < 0.1:
            strength = "negligible"
        elif v < 0.2:
            strength = "weak"
        elif v < 0.4:
            strength = "moderate"
        elif v < 0.6:
            strength = "relatively strong"
        else:
            strength = "very strong"
        print(f"  Interpretation: {strength} association")

    # Theil Index
    print("\n3. Theil Index (Inequality Measure):")
    theil = theil_index(df, "balance_mean", "age_group", count_col="count")
    print(f"  Total inequality: {theil['total']:.4f}")
    print(
        f"  Within-group component: {theil['within']:.4f} "
        + f"({100 * theil['within'] / theil['total']:.1f}% of total)"
    )
    print(
        f"  Between-group component: {theil['between']:.4f} "
        + f"({100 * theil['between'] / theil['total']:.1f}% of total)"
    )

    # Concentration Ratio
    print("\n4. Concentration Ratio:")
    # Combine age group and wealth tier for a more interesting analysis
    df["segment"] = df["age_group"] + "_" + df["wealth_tier"]

    # Calculate concentration across segments
    cr = concentration_ratio(df, "balance_mean", "segment", top_n=3, count_col="count")
    print(
        f"  Top 3 segments concentration (CR3): {cr['CR3']:.4f} "
        + f"({100 * cr['CR3']:.1f}% of total balance)"
    )
    print(f"  Herfindahl index: {cr['herfindahl']:.4f}")

    # Interpret Herfindahl index
    if cr["herfindahl"] < 0.01:
        market_type = "highly competitive"
    elif cr["herfindahl"] < 0.15:
        market_type = "unconcentrated"
    elif cr["herfindahl"] < 0.25:
        market_type = "moderately concentrated"
    else:
        market_type = "highly concentrated"
    print(f"  Interpretation: {market_type} distribution")

    # Anomaly Score
    print("\n5. Anomaly Scores (Combined Method):")
    anomalies = anomaly_score(
        df,
        "balance_mean",
        ["age_group", "channel"],
        method="combined",
        count_col="count",
    )
    print(
        anomalies[["age_group", "channel", "mean", "anomaly_score", "direction"]].head(
            5
        )
    )

    # Try isolation forest if scikit-learn is available
    try:
        import sklearn

        print("\n6. Anomaly Scores (Isolation Forest Method):")
        iso_anomalies = anomaly_score(
            df,
            "balance_mean",
            ["age_group", "channel"],
            method="isolation_forest",
            count_col="count",
        )
        print(
            iso_anomalies[
                ["age_group", "channel", "balance_mean", "anomaly_score"]
            ].head(5)
        )
    except ImportError:
        print("\n6. Isolation Forest method not available (requires scikit-learn)")

    # Information Gain
    print("\n7. Information Gain (Feature Importance for Categorical Data):")
    info_gain = information_gain(
        df, "balance_mean", ["age_group", "channel", "wealth_tier"], count_col="count"
    )

    # Sort by information gain and print
    for col, gain in sorted(info_gain.items(), key=lambda x: x[1], reverse=True):
        print(f"  {col}: {gain:.4f}")

    # Digital adoption analysis with advanced metrics
    print("\n" + "=" * 80)
    print("ADVANCED DIGITAL ADOPTION ANALYSIS")
    print("=" * 80)

    # Cramér's V for digital adoption
    print("\n8. Cramér's V for Digital Adoption:")
    for col in ["age_group", "channel", "wealth_tier"]:
        v = cramers_v(df, "digital_adoption", col, count_col="count")
        print(f"  Association between digital adoption and {col}: {v:.4f}")

    # Information gain for digital adoption
    print("\n9. Information Gain for Digital Adoption:")
    adoption_info_gain = information_gain(
        df,
        "digital_adoption",
        ["age_group", "channel", "wealth_tier"],
        count_col="count",
    )

    for col, gain in sorted(
        adoption_info_gain.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {col}: {gain:.4f}")


if __name__ == "__main__":
    main()
