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
    _default_null_pct = 0
    _default_distribution = "uniform"
    _default_id_start = 1
    _default_boolean_probability = 0.5
    _default_start_date = "2020-01-01"
    _default_end_date = "2023-12-31"
    _default_min_text_length = 5
    _default_max_text_length = 30

    for col_name, col_spec in specs.items():
        col_type = col_spec["type"]
        nulls_pct = col_spec.get("nulls_pct", _default_null_pct)

        # Generate column data based on type
        if col_type == "unique_key":
            start = col_spec.get("start", _default_id_start)
            result[col_name] = np.arange(start, start + n_rows)

        elif col_type == "boolean":
            prob_true = col_spec.get("prob_true", _default_boolean_probability)
            result[col_name] = np.random.random(n_rows) < prob_true

        elif col_type == "integer":
            result[col_name] = _generate_numeric_column(
                n_rows,
                distribution=col_spec.get("distribution", _default_distribution),
                params=col_spec,
                as_int=True,
            )

        elif col_type == "float":
            result[col_name] = _generate_numeric_column(
                n_rows,
                distribution=col_spec.get("distribution", _default_distribution),
                params=col_spec,
                as_int=False,
            )

        elif col_type == "date":
            start_date = col_spec.get("start_date", _default_start_date)
            end_date = col_spec.get("end_date", _default_end_date)
            eow_eom = col_spec.get("eow_eom", "day")

            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            start_ts = start_date.timestamp()
            end_ts = end_date.timestamp()

            # Generate random timestamps between start and end
            random_timestamps = np.random.uniform(start_ts, end_ts, n_rows)
            col_vals = pd.to_datetime(random_timestamps, unit="s")
            if eow_eom == "eow":
                col_vals = col_vals.map(next_friday)
            elif eow_eom == "eom":
                col_vals = col_vals.map(last_day_of_month)
            else:
                pass
            result[col_name] = col_vals.strftime("%Y-%m-%d")

        elif col_type == "category":
            values = col_spec["values"]
            probabilities = col_spec.get("probabilities")
            if probabilities is None:
                # Equal probability for each value
                probabilities = [1 / len(values)] * len(values)

            result[col_name] = np.random.choice(values, size=n_rows, p=probabilities)

        elif col_type == "text":
            min_length = col_spec.get("min_length", _default_min_text_length)
            max_length = col_spec.get("max_length", _default_max_text_length)

            # Generate random strings
            result[col_name] = np.array(
                [
                    "".join(
                        np.random.choice(
                            list(string.ascii_letters + string.digits),
                            np.random.randint(min_length, max_length + 1),
                        )
                    )
                    for _ in range(n_rows)
                ]
            )

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

    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    # Apply min/max limits if specified
    if "min" in params:
        values = np.maximum(values, params["min"])
    if "max" in params:
        values = np.minimum(values, params["max"])

    if as_int:
        values = values.astype(int)

    return values


def next_friday(dt):
    # If it's already Friday (weekday=4), return the same date
    # Otherwise, get the next Friday
    if dt.weekday() == 4:  # Friday is weekday 4
        return dt
    else:
        days_until_friday = (4 - dt.weekday()) % 7
        return dt + pd.DateOffset(days=days_until_friday)


def last_day_of_month(dt):
    next_month = dt.replace(day=28) + pd.DateOffset(days=4)  # Goes to next month
    return next_month - pd.DateOffset(
        days=next_month.day
    )  # Subtract the day to get last day of current month
