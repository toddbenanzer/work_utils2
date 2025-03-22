"""
Functions for filtering and defining population subsets from DataFrames.
This module provides utilities to apply filter functions to pandas DataFrames
to create population samples for analysis.
"""

from typing import Any, Callable, Dict, List, Union

import pandas as pd


def filter_population(
    df: pd.DataFrame, filter_func: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """
    Apply a filter function to a DataFrame to create a population subset.

    Args:
        df: Input DataFrame representing the entire population
        filter_func: Function that takes a DataFrame and returns a filtered DataFrame

    Returns:
        Filtered DataFrame representing the population subset
    """
    if not callable(filter_func):
        raise TypeError("filter_func must be callable")

    result = filter_func(df)

    # Check if the result is a valid DataFrame
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Filter function must return a pandas DataFrame")

    # Log the filter operation
    filter_rate = (len(result) / len(df)) * 100 if len(df) > 0 else 0
    print(
        f"Filter applied: {filter_func.__name__} - Retained {filter_rate:.1f}% of records"
    )

    return result


def combine_filters(filter_funcs: List[Callable], operator: str = "and") -> Callable:
    """
    Combine multiple filter functions using the specified logical operator.

    Args:
        filter_funcs: List of filter functions to combine
        operator: Logical operator to use ('and' or 'or')

    Returns:
        Combined filter function
    """
    if not filter_funcs:
        raise ValueError("At least one filter function must be provided")

    if operator.lower() not in ("and", "or"):
        raise ValueError("Operator must be 'and' or 'or'")

    def combined_filter(df: pd.DataFrame) -> pd.DataFrame:
        if operator.lower() == "and":
            result = df.copy()
            for filter_func in filter_funcs:
                result = filter_func(result)
            return result
        else:  # 'or' operator
            mask = pd.Series(False, index=df.index)
            for filter_func in filter_funcs:
                filtered = filter_func(df)
                mask = mask | df.index.isin(filtered.index)
            return df[mask]

    # Set a meaningful name for the combined filter
    filter_names = [getattr(f, "__name__", "unnamed_filter") for f in filter_funcs]
    combined_filter.__name__ = f"{operator.lower()}_".join(filter_names)

    return combined_filter


def create_filter_from_config(config: Dict[str, Any]) -> Callable:
    """
    Create a filter function from a configuration dictionary.

    Args:
        config: Dictionary containing filter configuration
            Must include 'type' and may include additional parameters

    Returns:
        Filter function that implements the specified filter
    """
    if "type" not in config:
        raise ValueError("Filter config must include 'type'")

    filter_type = config["type"]

    if filter_type == "threshold":
        # Example: {"type": "threshold", "column": "balance", "operator": "<", "value": 5000}
        return create_threshold_filter(
            config["column"], config["operator"], config["value"]
        )
    elif filter_type == "categorical":
        # Example: {"type": "categorical", "column": "account_type", "values": ["checking", "savings"]}
        return create_categorical_filter(
            config["column"], config["values"], config.get("include", True)
        )
    elif filter_type == "date_range":
        # Example: {"type": "date_range", "column": "open_date", "start": "2022-01-01", "end": "2022-12-31"}
        return create_date_range_filter(
            config["column"], config.get("start"), config.get("end")
        )
    elif filter_type == "custom":
        # For custom filter logic, expect a Python function in the config
        if "function" not in config:
            raise ValueError("Custom filter must include 'function'")
        return config["function"]
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def create_threshold_filter(
    column: str, operator: str, value: Union[int, float]
) -> Callable:
    """
    Create a filter function that applies a threshold comparison.

    Args:
        column: Column name to filter on
        operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
        value: Threshold value for comparison

    Returns:
        Filter function
    """
    op_map = {
        ">": lambda x, y: x > y,
        "<": lambda x, y: x < y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
        "==": lambda x, y: x == y,
        "!=": lambda x, y: x != y,
    }

    if operator not in op_map:
        raise ValueError(
            f"Invalid operator: {operator}. Must be one of {list(op_map.keys())}"
        )

    def threshold_filter(df: pd.DataFrame) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        return df[op_map[operator](df[column], value)]

    threshold_filter.__name__ = f"{column}_{operator}_{value}"
    return threshold_filter


def create_categorical_filter(
    column: str, values: List[Any], include: bool = True
) -> Callable:
    """
    Create a filter function that filters based on categorical values.

    Args:
        column: Column name to filter on
        values: List of values to include or exclude
        include: If True, keep rows with values in the list; if False, exclude them

    Returns:
        Filter function
    """

    def categorical_filter(df: pd.DataFrame) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        if include:
            return df[df[column].isin(values)]
        else:
            return df[~df[column].isin(values)]

    action = "include" if include else "exclude"
    categorical_filter.__name__ = (
        f"{column}_{action}_{','.join(str(v) for v in values)}"
    )
    return categorical_filter


def create_date_range_filter(column: str, start_date=None, end_date=None) -> Callable:
    """
    Create a filter function that filters based on a date range.

    Args:
        column: Date column name to filter on
        start_date: Start date (inclusive), or None for no lower bound
        end_date: End date (inclusive), or None for no upper bound

    Returns:
        Filter function
    """

    def date_range_filter(df: pd.DataFrame) -> pd.DataFrame:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")

        # Ensure column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            try:
                date_col = pd.to_datetime(df[column])
            except:
                raise ValueError(
                    f"Column '{column}' could not be converted to datetime"
                )
        else:
            date_col = df[column]

        result = df.copy()

        if start_date is not None:
            start = pd.to_datetime(start_date)
            result = result[date_col >= start]

        if end_date is not None:
            end = pd.to_datetime(end_date)
            result = result[date_col <= end]

        return result

    date_range_filter.__name__ = (
        f"{column}_from_{start_date or 'any'}_to_{end_date or 'any'}"
    )
    return date_range_filter
