# Example usage
dimensions_to_analyze = [
    "age_bucket",
    "income_range",
    "region",
    "tenure",
    "credit_score_range",
]

# For analyzing average balance differences
balance_importance = analyze_offer_differences(
    df=account_data,
    target_variable="average_balance",
    dimensions=dimensions_to_analyze,
    weights="statistical",  # Weighting by statistical significance
)

# For analyzing credit card adoption rate differences
cc_importance = analyze_offer_differences(
    df=account_data,
    target_variable="has_credit_card",  # Assuming 1=has card, 0=doesn't have card
    dimensions=dimensions_to_analyze,
    weights="count",  # Weighting by number of accounts
)

# Display results
print("Balance difference explanatory factors:")
print(balance_importance[["dimension", "importance_score", "avg_effect_size"]])

print("\nCredit card adoption difference explanatory factors:")
print(cc_importance[["dimension", "importance_score", "avg_effect_size"]])


# Example usage with your pre-aggregated data
# Assume you already have a DataFrame like this:
# agg_df with columns: 'age_bucket', 'offer_status', 'avg_balance', 'count'

# For analyzing one dimension (e.g., age_bucket)
age_importance = calculate_importance(
    agg_df=age_aggregated_df,  # Your pre-aggregated data for age bucket
    dimension_col="age_bucket",
    offer_status_col="offer_status",
    metric_col="avg_balance",  # or 'adoption_rate'
    count_col="count",
    weights="statistical",
)

print(f"Age bucket importance: {age_importance['importance_score']:.4f}")
print(f"Average effect size: {age_importance['avg_effect_size']:.4f}")
print(f"Median p-value: {age_importance['median_p_value']:.4f}")

# To rank multiple dimensions, you would run this for each dimension
# and compare the importance scores
