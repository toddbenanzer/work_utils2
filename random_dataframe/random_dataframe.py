"""
Module for generating random pandas DataFrames with various column types and distributions.
Useful for creating demonstration data, testing, and prototyping.
"""

import string
from typing import Dict, Optional

import numpy as np
import pandas as pd


def create_dataframe(
    specs: Dict, n_rows: int = 1000, random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a random DataFrame based on provided column specifications.

    Args:
        specs: Dictionary where keys are column names and values are specification dictionaries
        n_rows: Number of rows to generate
        random_seed: Random seed for reproducibility

    Returns:
        A pandas DataFrame with random data

    Example:
        specs = {
            'id': {'type': 'unique_key', 'start': 1000},
            'value': {'type': 'float', 'distribution': 'normal', 'mean': 100, 'std': 15, 'min': 0},
            'category': {'type': 'category', 'values': ['A', 'B', 'C'], 'probabilities': [0.7, 0.2, 0.1]},
        }
        df = create_dataframe(specs, n_rows=500)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    result = {}

    for col_name, col_spec in specs.items():
        col_type = col_spec["type"]
        nulls_pct = col_spec.get("nulls_pct", 0)

        # Generate column data based on type
        if col_type == "unique_key":
            start = col_spec.get("start", 1)
            result[col_name] = np.arange(start, start + n_rows)

        elif col_type == "boolean":
            prob_true = col_spec.get("prob_true", 0.5)
            result[col_name] = np.random.random(n_rows) < prob_true

        elif col_type == "integer":
            result[col_name] = _generate_numeric_column(
                n_rows,
                distribution=col_spec.get("distribution", "uniform"),
                params=col_spec,
                as_int=True,
            )

        elif col_type == "float":
            result[col_name] = _generate_numeric_column(
                n_rows,
                distribution=col_spec.get("distribution", "uniform"),
                params=col_spec,
                as_int=False,
            )

        elif col_type == "date":
            start_date = col_spec.get("start_date", "2020-01-01")
            end_date = col_spec.get("end_date", "2023-12-31")

            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            start_ts = start_date.timestamp()
            end_ts = end_date.timestamp()

            # Generate random timestamps between start and end
            random_timestamps = np.random.uniform(start_ts, end_ts, n_rows)
            result[col_name] = pd.to_datetime(random_timestamps, unit="s").strftime(
                "%Y-%m-%d"
            )

        elif col_type == "category":
            values = col_spec["values"]
            probabilities = col_spec.get("probabilities")

            if probabilities is None:
                # Equal probability for each value
                probabilities = [1 / len(values)] * len(values)

            result[col_name] = np.random.choice(values, size=n_rows, p=probabilities)

        elif col_type == "text":
            min_length = col_spec.get("min_length", 5)
            max_length = col_spec.get("max_length", 30)

            # Generate random strings
            result[col_name] = [
                "".join(
                    np.random.choice(
                        list(string.ascii_letters + string.digits),
                        np.random.randint(min_length, max_length + 1),
                    )
                )
                for _ in range(n_rows)
            ]

        # Apply nulls if specified
        if nulls_pct > 0:
            null_mask = np.random.random(n_rows) < (nulls_pct / 100)
            result[col_name] = pd.Series(result[col_name])
            result[col_name][null_mask] = None

    return pd.DataFrame(result)


def _generate_numeric_column(
    n_rows: int,
    distribution: str = "uniform",
    params: Dict = None,
    as_int: bool = False,
) -> np.ndarray:
    """
    Generate numeric column with specified distribution.

    Args:
        n_rows: Number of rows
        distribution: Distribution type ('uniform', 'normal', 'lognormal')
        params: Parameters for the distribution
        as_int: Whether to convert to integers

    Returns:
        NumPy array with generated values
    """
    params = params or {}
    values = None

    if distribution == "uniform":
        min_val = params.get("min", 0)
        max_val = params.get("max", 100)
        values = np.random.uniform(min_val, max_val, n_rows)

    elif distribution == "normal":
        mean = params.get("mean", 0)
        std = params.get("std", 1)
        values = np.random.normal(mean, std, n_rows)

    elif distribution == "lognormal":
        mean = params.get("mean", 0)
        sigma = params.get("sigma", 1)
        values = np.random.lognormal(mean, sigma, n_rows)

    # Apply min/max limits if specified
    if "min" in params:
        values = np.maximum(values, params["min"])
    if "max" in params:
        values = np.minimum(values, params["max"])

    if as_int:
        values = values.astype(int)

    return values
