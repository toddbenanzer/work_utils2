# Random DataFrame Generator

A lightweight Python module for generating random pandas DataFrames with various column types and distributions. Perfect for creating demo data, testing, and prototyping data analysis workflows.

## Features

- Generate DataFrames with customizable number of rows
- Create columns with different data types:
  - Unique keys / IDs
  - Boolean values
  - Integer values
  - Float values
  - Dates
  - Categorical values with custom distributions
  - Random text
- Control distributions of numeric data:
  - Uniform
  - Normal (Gaussian)
  - Log-normal
- Apply constraints like min/max values
- Add null values with specified frequency
- Set random seed for reproducibility

## Installation

1. Clone this repository or copy the files to your project directory
2. Make sure you have the required dependencies:
   ```
   pip install pandas numpy
   ```

## Usage

### Basic Example

```python
from random_dataframe import create_dataframe
from utils import create_id_column, create_float_column, create_category_column

# Define column specifications
specs = {
    'id': create_id_column(start=1001),
    'value': create_float_column(
        distribution='normal',
        mean=100,
        std=15,
        min_val=0
    ),
    'category': create_category_column(
        values=['Low', 'Medium', 'High'],
        probabilities=[0.6, 0.3, 0.1]
    ),
}

# Generate DataFrame
df = create_dataframe(specs, n_rows=1000, random_seed=42)
print(df.head())
```

### Column Type Reference

#### Unique Key
```python
create_id_column(start=1, nulls_pct=0)
```

#### Boolean
```python
create_boolean_column(prob_true=0.5, nulls_pct=0)
```

#### Integer
```python
create_integer_column(
    distribution='uniform',  # Options: 'uniform', 'normal', 'lognormal'
    min_val=0,               # Optional minimum constraint
    max_val=100,             # Optional maximum constraint
    mean=0,                  # For normal/lognormal distributions
    std=1,                   # Standard deviation / sigma
    nulls_pct=0              # Percentage of null values
)
```

#### Float
```python
create_float_column(
    distribution='uniform',  # Options: 'uniform', 'normal', 'lognormal'
    min_val=0,               # Optional minimum constraint
    max_val=100,             # Optional maximum constraint
    mean=0,                  # For normal/lognormal distributions
    std=1,                   # Standard deviation / sigma
    nulls_pct=0              # Percentage of null values
)
```

#### Date
```python
create_date_column(
    start_date='2020-01-01',
    end_date='2023-12-31',
    nulls_pct=0
)
```

#### Category
```python
create_category_column(
    values=['A', 'B', 'C'],                # List of possible values
    probabilities=[0.7, 0.2, 0.1],         # Optional probability weights
    nulls_pct=0                            # Percentage of null values
)
```

#### Text
```python
create_text_column(
    min_length=5,
    max_length=30,
    nulls_pct=0
)
```

## Advanced Example

Creating a sales dataset with multiple column types:

```python
from random_dataframe import create_dataframe
from utils import (
    create_id_column,
    create_date_column,
    create_integer_column,
    create_float_column,
    create_category_column,
    create_boolean_column
)

# Create a dataset simulating sales data
specs = {
    'transaction_id': create_id_column(start=10000),
    'date': create_date_column(
        start_date='2022-01-01',
        end_date='2022-12-31'
    ),
    'customer_id': create_integer_column(
        distribution='uniform',
        min_val=1, 
        max_val=1000
    ),
    'product_category': create_category_column(
        values=['Electronics', 'Clothing', 'Food', 'Books', 'Home'],
        probabilities=[0.3, 0.25, 0.2, 0.15, 0.1]
    ),
    'quantity': create_integer_column(
        distribution='lognormal',
        mean=1,
        std=0.5,
        min_val=1,
        max_val=10
    ),
    'price': create_float_column(
        distribution='lognormal',
        mean=3,
        std=1,
        min_val=10,
        max_val=500
    ),
    'is_discounted': create_boolean_column(prob_true=0.3)
}

# Create the dataset
sales_df = create_dataframe(specs, n_rows=10000, random_seed=42)

# Calculate total revenue
sales_df['revenue'] = sales_df['quantity'] * sales_df['price']

# Analyze the data
monthly_sales = sales_df.groupby(sales_df['date'].dt.strftime('%Y-%m'))['revenue'].sum()
print(monthly_sales)
```

## Extending the Module

The module is designed to be simple and extensible. To add new column types or distributions:

1. Add a new generation function in `random_dataframe.py`
2. Create a corresponding helper function in `utils.py` 
3. Update the imports and `__all__` list in `__init__.py`

## See More Examples

Check the `examples.py` file for more usage examples and demonstrations.