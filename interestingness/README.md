# Interestingness Metrics

A Python package for evaluating how "interesting" or meaningful groupings are in your pre-aggregated data. This helps you identify which segmentation variables create the most significant or insightful distinctions in your metrics of interest.

## Overview

When exploring data, you often want to know which grouping variables (like product category, region, customer segment, etc.) create the most meaningful differences in your metrics. This package provides various statistical measures to quantify this "interestingness" and helps you rank different grouping approaches.

This package is designed to work with **pre-aggregated data**, where you've already calculated metrics like means, counts, and variances for each group.

## Features

- **Basic Interestingness Metrics**
  - Group variance and coefficient of variation
  - ANOVA F-statistics and effect sizes
  - Maximum deviation and range ratios
  - Gini coefficient

- **Advanced Interestingness Metrics**
  - Entropy reduction / information gain
  - Discriminative power
  - Group separation index
  - Outlier group detection
  - Non-parametric tests (Kruskal-Wallis approximation)

- **Utility Functions**
  - Ranking groups by various metrics
  - Visualizing group differences
  - Comprehensive interestingness reports
  - Multi-column group handling

## Installation

```bash
# Not yet available on PyPI
# For now, just copy the package files to your project
```

## Basic Usage

```python
import pandas as pd
from interestingness_metrics import coefficient_of_variation, anova_f_statistic
from interestingness_utils import interestingness_report

# Load your pre-aggregated data
# Each row represents a group with columns for mean, count, variance, etc.
df = pd.read_csv('my_aggregated_data.csv')

# Calculate a simple interestingness metric
cv = coefficient_of_variation(df, mean_col='avg_value', weight_col='count')
print(f"Coefficient of variation across groups: {cv:.2f}")

# Calculate ANOVA F-statistic
anova_result = anova_f_statistic(
    df, mean_col='avg_value', count_col='count', var_col='variance'
)
print(f"ANOVA F-statistic: {anova_result['f_statistic']:.2f}")
print(f"p-value: {anova_result['p_value']:.5f}")

# Generate a comprehensive report
report = interestingness_report(
    df, 
    mean_col='avg_value',
    count_col='count',
    group_col='segment',
    var_col='variance'
)
```

## Example Use Cases

1. **Product Analytics**: Which product categorization scheme (by type, price tier, customer segment, etc.) best explains differences in conversion rates?

2. **Marketing Segmentation**: Which customer segment shows the most significant differences in average order value?

3. **A/B Test Analysis**: Beyond just comparing test vs. control, which subgroups show the greatest differentiation in response to the test?

4. **Geographic Analysis**: Which geographic grouping level (country, region, city) shows the most meaningful patterns in user engagement?

## Requirements

- Python 3.6+
- NumPy
- pandas
- SciPy
- Matplotlib (for visualization functions)

## Contributing

This package is designed to be extended with additional interestingness metrics. Feel free to suggest or implement additional ways to evaluate groupings.

## License

MIT