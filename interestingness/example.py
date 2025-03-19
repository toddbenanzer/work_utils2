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
