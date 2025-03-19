"""
Utility functions for random DataFrame generation.
Provides helpers to create common column specifications and manage data generation.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def create_id_column(start: int = 1, nulls_pct: float = 0) -> Dict:
    """
    Create specification for a unique ID column.

    Args:
        start: Starting value for the sequence
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    return {"type": "unique_key", "start": start, "nulls_pct": nulls_pct}


def create_boolean_column(prob_true: float = 0.5, nulls_pct: float = 0) -> Dict:
    """
    Create specification for a boolean column.

    Args:
        prob_true: Probability of True values (0-1)
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    return {"type": "boolean", "prob_true": prob_true, "nulls_pct": nulls_pct}


def create_integer_column(
    distribution: str = "uniform",
    min_val: Optional[int] = None,
    max_val: Optional[int] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    nulls_pct: float = 0,
) -> Dict:
    """
    Create specification for an integer column.

    Args:
        distribution: 'uniform', 'normal', or 'lognormal'
        min_val: Minimum value constraint
        max_val: Maximum value constraint
        mean: Mean for normal or lognormal distributions
        std: Standard deviation for normal distribution
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    spec = {"type": "integer", "distribution": distribution, "nulls_pct": nulls_pct}

    if distribution == "uniform":
        spec["min"] = 0 if min_val is None else min_val
        spec["max"] = 100 if max_val is None else max_val
    elif distribution == "normal":
        spec["mean"] = 0 if mean is None else mean
        spec["std"] = 1 if std is None else std
    elif distribution == "lognormal":
        spec["mean"] = 0 if mean is None else mean
        spec["sigma"] = 1 if std is None else std

    # Add constraints if provided
    if min_val is not None and distribution != "uniform":
        spec["min"] = min_val
    if max_val is not None and distribution != "uniform":
        spec["max"] = max_val

    return spec


def create_float_column(
    distribution: str = "uniform",
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None,
    nulls_pct: float = 0,
) -> Dict:
    """
    Create specification for a float column.

    Args:
        distribution: 'uniform', 'normal', or 'lognormal'
        min_val: Minimum value constraint
        max_val: Maximum value constraint
        mean: Mean for normal or lognormal distributions
        std: Standard deviation for normal distribution or sigma for lognormal
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    # Reuse the integer column creation with float type
    spec = create_integer_column(
        distribution=distribution,
        min_val=min_val,
        max_val=max_val,
        mean=mean,
        std=std,
        nulls_pct=nulls_pct,
    )
    spec["type"] = "float"
    return spec


def create_date_column(
    start_date: Union[str, pd.Timestamp] = "2020-01-01",
    end_date: Union[str, pd.Timestamp] = "2023-12-31",
    nulls_pct: float = 0,
) -> Dict:
    """
    Create specification for a date column.

    Args:
        start_date: Lower bound for date range
        end_date: Upper bound for date range
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    return {
        "type": "date",
        "start_date": start_date,
        "end_date": end_date,
        "nulls_pct": nulls_pct,
    }


def create_category_column(
    values: List, probabilities: Optional[List[float]] = None, nulls_pct: float = 0
) -> Dict:
    """
    Create specification for a categorical column.

    Args:
        values: List of possible values
        probabilities: Probability weights for each value (must sum to 1)
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    spec = {"type": "category", "values": values, "nulls_pct": nulls_pct}

    if probabilities is not None:
        # Ensure probabilities sum to 1
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        spec["probabilities"] = list(probabilities)

    return spec


def create_text_column(
    min_length: int = 5, max_length: int = 30, nulls_pct: float = 0
) -> Dict:
    """
    Create specification for a random text column.

    Args:
        min_length: Minimum text length
        max_length: Maximum text length
        nulls_pct: Percentage of null values (0-100)

    Returns:
        Column specification dictionary
    """
    return {
        "type": "text",
        "min_length": min_length,
        "max_length": max_length,
        "nulls_pct": nulls_pct,
    }
