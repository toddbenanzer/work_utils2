# Interestingness

A Python toolkit for identifying interesting patterns in pandas GroupBy aggregation results.

## Overview

The Interestingness package provides a collection of metrics and functions to help analysts discover and quantify meaningful patterns, relationships, and unusual segments in aggregated data. It is particularly useful when analyzing contingency tables or the results of pandas `groupby().agg()` operations.

The package includes three categories of metrics:

1. **Contribution-based metrics**: Identify which grouping variables most strongly influence your target metrics
2. **Distribution-based metrics**: Find unusual patterns and segments with different distributions
3. **Effect size metrics**: Measure standardized differences between segments

## Installation

```bash
pip install interestingness
```

Or clone this repository and install in development mode:

```bash
git clone https://github.com/username/interestingness.git
cd interestingness
pip install -e .
```

## Quick Start

```python
import pandas as pd
from interestingness.contribution import variance_explained, feature_importance
from interestingness.distribution import distribution_outliers
from interestingness.effect_size import percent_difference

# Load aggregated data
# This example assumes data has been aggregated with pd.groupby().agg()
df = pd.read_csv('aggregated_data.csv')

# Find which variables best explain balance differences
var_exp = variance_explained(df, 'balance_mean', ['age_group', 'channel', 'wealth_tier'], count_col='count')
print("Variance explained:", var_exp)

# Identify segments with unusual values
outliers = distribution_outliers(df, 'balance_mean', ['age_group', 'channel'], threshold=2.0)
print("Outlier segments:", outliers)

# Calculate percent differences between segments
pct_diff = percent_difference(df, 'balance_mean', ['channel'], count_col='count')
print("Percent differences:", pct_diff)
```

## Key Functions

### Contribution Metrics

- `variance_explained`: Calculate the proportion of variance in a target metric explained by each grouping variable
- `feature_importance`: Rank variables by how strongly they influence the target metric
- `mutual_information`: Measure the statistical dependence between variables

### Distribution Metrics

- `entropy_score`: Calculate the entropy (variability) within each group
- `kl_divergence`: Measure how different one group's distribution is from others
- `distribution_outliers`: Identify segments with statistically unusual values

### Effect Size Metrics

- `cohens_d`: Calculate standardized effect size between two specific groups
- `standardized_difference`: Compute pairwise standardized differences between groups
- `percent_difference`: Calculate percentage differences from a reference value

## Use Cases

- Finding drivers of key metrics like account balances, adoption rates, or conversions
- Discovering which customer segments differ most from the overall population
- Quantifying the impact of different variables on business metrics
- Identifying unusual patterns or anomalies within segments

## Interestingness Metrics Guide

This guide helps you understand when to use each metric in the interestingness toolkit and how to interpret the results.

### When to Use Each Type of Metric

#### Contribution-Based Metrics

**Use these when you want to know:**
- Which segmentation variables are most strongly driving differences in your target metrics
- How much of the variation in your data is explained by specific variables
- Which combinations of variables have the strongest associations with your target metrics

**Key Functions:**
- `variance_explained`: Shows how much of the overall variance in your target metric is explained by each grouping variable or combination of variables
- `feature_importance`: Provides a ranked list of which variables (and their interactions) most strongly affect your target metric
- `mutual_information`: Measures the statistical dependence between variables, detecting both linear and non-linear relationships

**Interpretation Example:**
If `variance_explained` shows that `age_group` explains 45% of the variance in account balances, but `age_group + wealth_tier` explains 85%, this suggests wealth tier is an important driver that interacts with age.

#### Distribution-Based Metrics

**Use these when you want to know:**
- Which segments have unusual patterns or distributions
- How different a specific segment's distribution is from others
- Which groups contain outliers or surprising values

**Key Functions:**
- `entropy_score`: Measures the uniformity or variability within groups (lower entropy = more uniformity)
- `kl_divergence`: Quantifies how one distribution differs from another, useful for finding segments that behave differently
- `distribution_outliers`: Identifies segments with unusually high or low values based on statistical criteria

**Interpretation Example:**
If the `kl_divergence` for online customers is much higher than for branch customers, this indicates that online customers have a distinctly different pattern in their balances or adoption rates compared to the overall customer base.

#### Effect Size Metrics

**Use these when you want to know:**
- The magnitude of differences between specific segments
- How practically significant (not just statistically significant) differences are
- Which pairwise comparisons show the largest effects

**Key Functions:**
- `cohens_d`: Calculates standardized effect size between two groups (small: 0.2, medium: 0.5, large: 0.8)
- `standardized_difference`: Shows all pairwise differences between groups in standardized units
- `percent_difference`: Calculates percentage differences from a reference value

**Interpretation Example:**
A `cohens_d` of 1.2 between young and old customer balances indicates a very large effect size - the difference is not only statistically significant but practically meaningful, with old customers having balances more than one standard deviation higher than young customers.

#### Common Analysis Workflows

##### Finding What Drives a Key Metric

1. **Start with contribution metrics**:
   ```python
   # Which variables explain most of the variance in balances?
   var_exp = variance_explained(df, 'balance_mean', ['age', 'channel', 'wealth_tier'])
   feature_imp = feature_importance(df, 'balance_mean', ['age', 'channel', 'wealth_tier'])
   ```

2. **Examine the largest differences**:
   ```python
   # Get standardized differences between groups
   std_diffs = standardized_difference(df, 'balance_mean', ['age', 'channel'])
   ```

3. **Look for unusual segments**:
   ```python
   # Find segments that deviate from the overall pattern
   outliers = distribution_outliers(df, 'balance_mean', ['age', 'channel'], threshold=2.0)
   ```

#### Comparing Two Specific Segments

1. **Calculate direct effect size**:
   ```python
   # Compare online vs. branch customers
   effect = cohens_d(df, 'balance_mean', 'channel', 'Online', 'Branch')
   ```

2. **Get percent difference**:
   ```python
   # How much higher/lower are online customer balances?
   pct_diff = percent_difference(df, 'balance_mean', 'channel', reference_value='Branch')
   ```

3. **Check distribution difference**:
   ```python
   # How different is the distribution of online customers vs. branch?
   kl = kl_divergence(df, 'balance_mean', 'channel', reference_group='Branch')
   ```

#### Discovering Unexpected Patterns

1. **Start with outlier detection**:
   ```python
   # Find segments with unusual values
   outliers = distribution_outliers(df, 'balance_mean', ['age', 'channel', 'wealth_tier'])
   ```

2. **Check entropy across segments**:
   ```python
   # Which segments have more variable values?
   entropy = entropy_score(df, 'balance_mean', 'wealth_tier')
   ```

3. **Examine interactions**:
   ```python
   # Are there interesting interaction effects?
   importance = feature_importance(df, 'balance_mean', ['age', 'channel', 'wealth_tier'], 
                                 include_interactions=True)
   ```

### Interpretation Guidelines

#### For Contribution Metrics

- **Variance Explained**: Values range from 0 to 1 (or 0% to 100%)
  - 0-0.1 (0-10%): Variable explains very little variance
  - 0.1-0.3 (10-30%): Variable explains moderate variance
  - 0.3-0.5 (30-50%): Variable explains substantial variance
  - >0.5 (>50%): Variable is a major driver of the target metric

- **Feature Importance**: Focus on relative rankings
  - Main effects: Direct impact of individual variables
  - Interactions: Additional impact from variable combinations beyond their individual effects
  - Higher importance = more influential in determining the target metric

- **Mutual Information**: Values are non-negative, with higher values indicating stronger relationships
  - When normalized (0-1): 0 means independence, 1 means perfect dependence

#### For Distribution Metrics

- **Entropy Score**: Typically ranges from 0 to 1 when normalized
  - 0: Complete uniformity (all values identical)
  - Higher values: More variability in the group
  - Compare entropy between groups to find segments with more/less consistency

- **KL Divergence**: Non-negative values, with higher values indicating greater distribution differences
  - 0: Distributions are identical
  - 0.1-0.5: Modest differences
  - >0.5: Substantial differences
  - >2.0: Extreme differences

- **Distribution Outliers**: Interpretation depends on the method used
  - Z-score: Values >2 are typically considered outliers (>2 standard deviations)
  - IQR: Values >1.5 indicate potential outliers (>1.5 × interquartile range)
  - Check the 'direction' column to see if the outlier is above or below expectations

#### For Effect Size Metrics

- **Cohen's d**: Standardized difference between two groups
  - 0.2: Small effect
  - 0.5: Medium effect
  - 0.8: Large effect
  - >1.2: Very large effect

- **Standardized Difference**: Similar interpretation to Cohen's d
  - Look for the largest differences to find the most noteworthy comparisons
  - The sign indicates direction (positive means first group > second group)

- **Percent Difference**: Easily interpretable in business terms
  - The magnitude indicates practical significance
  - For example, a 50% higher balance is more impactful than a 5% difference

### Advanced Metrics

The interestingness package also includes advanced metrics for more sophisticated analysis:

#### Gini Impurity

**Use when you want to know:**
- How "pure" or homogeneous a group is with respect to the target metric
- Which segments have the most mixed values and which are most uniform

**Interpretation:**
- Range: 0 to 0.5 (for binary outcomes)
- 0: Completely pure (all values in the group are identical)
- Higher values: More impurity/diversity within the group

#### Cramér's V

**Use when you want to know:**
- The strength of association between categorical variables
- Similar to correlation but for categorical data

**Interpretation:**
- Range: 0 to 1
- 0

3. **Consider practical significance**:
   - Statistical measures should be paired with business context
   - A large effect size might be important even if it affects a small segment
   - A small effect size might be important if it affects a large segment

4. **Look for interactions**:
   - Many interesting insights come from how variables interact
   - For example, a channel might perform differently across age groups
   - The feature_importance function with include_interactions=True helps identify these

5. **Validate findings**:
   - Cross-check insights across multiple metrics
   - Consider testing on different time periods or datasets
   - Be cautious about drawing conclusions from very small segments


## License

MIT License