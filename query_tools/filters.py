"""
Filter tools for working with pandas DataFrames.

Provides a flexible way to create and combine filters for DataFrame selection using a
dictionary-based interface. Filters are created as boolean masks that can be applied
to DataFrames as needed.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

# ===== Base filter functions =====


def create_equals_filter(df: pd.DataFrame, column: str, value: Any) -> pd.Series:
    """
    Create a filter where column equals the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] == value


def create_not_equals_filter(df: pd.DataFrame, column: str, value: Any) -> pd.Series:
    """
    Create a filter where column does not equal the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] != value


def create_in_filter(df: pd.DataFrame, column: str, values: List[Any]) -> pd.Series:
    """
    Create a filter where column value is in the specified list.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        values: List of values to check against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column].isin(values)


def create_not_in_filter(df: pd.DataFrame, column: str, values: List[Any]) -> pd.Series:
    """
    Create a filter where column value is not in the specified list.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        values: List of values to check against

    Returns:
        Boolean Series that can be used as a mask
    """
    return ~df[column].isin(values)


def create_greater_than_filter(
    df: pd.DataFrame, column: str, value: Union[int, float, pd.Timestamp]
) -> pd.Series:
    """
    Create a filter where column is greater than the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] > value


def create_greater_equals_filter(
    df: pd.DataFrame, column: str, value: Union[int, float, pd.Timestamp]
) -> pd.Series:
    """
    Create a filter where column is greater than or equal to the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] >= value


def create_less_than_filter(
    df: pd.DataFrame, column: str, value: Union[int, float, pd.Timestamp]
) -> pd.Series:
    """
    Create a filter where column is less than the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] < value


def create_less_equals_filter(
    df: pd.DataFrame, column: str, value: Union[int, float, pd.Timestamp]
) -> pd.Series:
    """
    Create a filter where column is less than or equal to the specified value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        value: Value to compare against

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column] <= value


def create_between_filter(
    df: pd.DataFrame,
    column: str,
    min_value: Union[int, float, pd.Timestamp],
    max_value: Union[int, float, pd.Timestamp],
    inclusive: str = "both",
) -> pd.Series:
    """
    Create a filter where column is between min_value and max_value.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        min_value: Minimum value for range
        max_value: Maximum value for range
        inclusive: Include bounds: 'both' (default), 'neither', 'left', 'right'

    Returns:
        Boolean Series that can be used as a mask
    """
    if inclusive == "both":
        return (df[column] >= min_value) & (df[column] <= max_value)
    elif inclusive == "left":
        return (df[column] >= min_value) & (df[column] < max_value)
    elif inclusive == "right":
        return (df[column] > min_value) & (df[column] <= max_value)
    elif inclusive == "neither":
        return (df[column] > min_value) & (df[column] < max_value)
    else:
        raise ValueError(
            f"Invalid 'inclusive' value: {inclusive}. Must be 'both', 'neither', 'left', or 'right'"
        )


def create_null_filter(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Create a filter for rows where column value is null.

    Args:
        df: Input DataFrame
        column: Column name to filter on

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column].isna()


def create_not_null_filter(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Create a filter for rows where column value is not null.

    Args:
        df: Input DataFrame
        column: Column name to filter on

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column].notna()


def create_contains_filter(
    df: pd.DataFrame, column: str, substring: str, case: bool = True
) -> pd.Series:
    """
    Create a filter for rows where column value contains the substring.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        substring: String to search for
        case: If True, case-sensitive. If False, case-insensitive.

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column].str.contains(substring, case=case, na=False)


def create_startswith_filter(
    df: pd.DataFrame, column: str, prefix: str, case: bool = True
) -> pd.Series:
    """
    Create a filter for rows where column value starts with the prefix.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        prefix: String to check at beginning
        case: If True, case-sensitive. If False, case-insensitive.

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to string to handle non-string columns
    if case:
        return df[column].astype(str).str.startswith(prefix, na=False)
    else:
        return (
            df[column].astype(str).str.lower().str.startswith(prefix.lower(), na=False)
        )


def create_endswith_filter(
    df: pd.DataFrame, column: str, suffix: str, case: bool = True
) -> pd.Series:
    """
    Create a filter for rows where column value ends with the suffix.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        suffix: String to check at end
        case: If True, case-sensitive. If False, case-insensitive.

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to string to handle non-string columns
    if case:
        return df[column].astype(str).str.endswith(suffix, na=False)
    else:
        return df[column].astype(str).str.lower().str.endswith(suffix.lower(), na=False)


def create_regex_filter(
    df: pd.DataFrame, column: str, pattern: str, case: bool = True
) -> pd.Series:
    """
    Create a filter for rows where column value matches the regex pattern.

    Args:
        df: Input DataFrame
        column: Column name to filter on
        pattern: Regular expression pattern
        case: If True, case-sensitive. If False, case-insensitive.

    Returns:
        Boolean Series that can be used as a mask
    """
    return df[column].str.match(pattern, case=case, na=False)


# ===== Date-specific filters =====


def create_date_year_filter(df: pd.DataFrame, column: str, year: int) -> pd.Series:
    """
    Create a filter for rows where date column has the specified year.

    Args:
        df: Input DataFrame
        column: Date column name
        year: Year to filter by

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to datetime if not already
    date_col = pd.to_datetime(df[column])
    return date_col.dt.year == year


def create_date_month_filter(df: pd.DataFrame, column: str, month: int) -> pd.Series:
    """
    Create a filter for rows where date column has the specified month.

    Args:
        df: Input DataFrame
        column: Date column name
        month: Month to filter by (1-12)

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to datetime if not already
    date_col = pd.to_datetime(df[column])
    return date_col.dt.month == month


def create_date_day_filter(df: pd.DataFrame, column: str, day: int) -> pd.Series:
    """
    Create a filter for rows where date column has the specified day.

    Args:
        df: Input DataFrame
        column: Date column name
        day: Day to filter by (1-31)

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to datetime if not already
    date_col = pd.to_datetime(df[column])
    return date_col.dt.day == day


def create_date_quarter_filter(
    df: pd.DataFrame, column: str, quarter: int
) -> pd.Series:
    """
    Create a filter for rows where date column is in the specified quarter.

    Args:
        df: Input DataFrame
        column: Date column name
        quarter: Quarter to filter by (1-4)

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to datetime if not already
    date_col = pd.to_datetime(df[column])
    return date_col.dt.quarter == quarter


def create_date_weekday_filter(
    df: pd.DataFrame, column: str, weekday: int
) -> pd.Series:
    """
    Create a filter for rows where date column is on the specified weekday.

    Args:
        df: Input DataFrame
        column: Date column name
        weekday: Weekday to filter by (0=Monday, 6=Sunday)

    Returns:
        Boolean Series that can be used as a mask
    """
    # Convert to datetime if not already
    date_col = pd.to_datetime(df[column])
    return date_col.dt.weekday == weekday


def combine_filters_and(df: pd.DataFrame, filters: List[Dict]) -> pd.Series:
    """
    Combine multiple filters with AND logic.

    Args:
        df: Input DataFrame
        filters: List of filter specifications

    Returns:
        Combined boolean mask
    """
    if not filters:
        return pd.Series(True, index=df.index)

    result = apply_filter(df, filters[0])
    for filter_spec in filters[1:]:
        result = result & apply_filter(df, filter_spec)

    return result


def combine_filters_or(df: pd.DataFrame, filters: List[Dict]) -> pd.Series:
    """
    Combine multiple filters with OR logic.

    Args:
        df: Input DataFrame
        filters: List of filter specifications

    Returns:
        Combined boolean mask
    """
    if not filters:
        return pd.Series(False, index=df.index)

    result = apply_filter(df, filters[0])
    for filter_spec in filters[1:]:
        result = result | apply_filter(df, filter_spec)

    return result


def negate_filter(df: pd.DataFrame, filter_spec: Dict) -> pd.Series:
    """
    Negate a filter (NOT logic).

    Args:
        df: Input DataFrame
        filter_spec: Filter specification

    Returns:
        Negated boolean mask
    """
    return ~apply_filter(df, filter_spec)


def apply_filter(df: pd.DataFrame, filter_spec: Dict) -> pd.Series:
    """
    Apply a filter specification to a DataFrame.

    Args:
        df: Input DataFrame
        filter_spec: Dictionary defining the filter

    Returns:
        Boolean Series mask

    Example filter specifications:
        {"type": "equals", "column": "category", "value": "A"}
        {"type": "between", "column": "score", "min_value": 70, "max_value": 90}
        {"type": "in", "column": "product_category", "values": ["Books", "Electronics"]}
        {"type": "and", "filters": [filter1, filter2, ...]}
        {"type": "date_year", "column": "open_date", "year": 2024}
    """
    filter_type = filter_spec.get("type", "").lower()

    # Simple filters
    if filter_type == "equals":
        return create_equals_filter(df, filter_spec["column"], filter_spec["value"])
    elif filter_type == "not_equals":
        return create_not_equals_filter(df, filter_spec["column"], filter_spec["value"])
    elif filter_type == "in":
        return create_in_filter(df, filter_spec["column"], filter_spec["values"])
    elif filter_type == "not_in":
        return create_not_in_filter(df, filter_spec["column"], filter_spec["values"])
    elif filter_type == "greater_than":
        return create_greater_than_filter(
            df, filter_spec["column"], filter_spec["value"]
        )
    elif filter_type == "greater_equals":
        return create_greater_equals_filter(
            df, filter_spec["column"], filter_spec["value"]
        )
    elif filter_type == "less_than":
        return create_less_than_filter(df, filter_spec["column"], filter_spec["value"])
    elif filter_type == "less_equals":
        return create_less_equals_filter(
            df, filter_spec["column"], filter_spec["value"]
        )
    elif filter_type == "between":
        inclusive = filter_spec.get("inclusive", "both")
        return create_between_filter(
            df,
            filter_spec["column"],
            filter_spec["min_value"],
            filter_spec["max_value"],
            inclusive,
        )
    elif filter_type == "null":
        return create_null_filter(df, filter_spec["column"])
    elif filter_type == "not_null":
        return create_not_null_filter(df, filter_spec["column"])
    elif filter_type == "contains":
        case = filter_spec.get("case", True)
        return create_contains_filter(
            df, filter_spec["column"], filter_spec["substring"], case
        )
    elif filter_type == "startswith":
        case = filter_spec.get("case", True)
        return create_startswith_filter(
            df, filter_spec["column"], filter_spec["prefix"], case
        )
    elif filter_type == "endswith":
        case = filter_spec.get("case", True)
        return create_endswith_filter(
            df, filter_spec["column"], filter_spec["suffix"], case
        )
    elif filter_type == "regex":
        case = filter_spec.get("case", True)
        return create_regex_filter(
            df, filter_spec["column"], filter_spec["pattern"], case
        )

    # Date filters
    elif filter_type == "date_year":
        return create_date_year_filter(df, filter_spec["column"], filter_spec["year"])
    elif filter_type == "date_month":
        return create_date_month_filter(df, filter_spec["column"], filter_spec["month"])
    elif filter_type == "date_day":
        return create_date_day_filter(df, filter_spec["column"], filter_spec["day"])
    elif filter_type == "date_quarter":
        return create_date_quarter_filter(
            df, filter_spec["column"], filter_spec["quarter"]
        )
    elif filter_type == "date_weekday":
        return create_date_weekday_filter(
            df, filter_spec["column"], filter_spec["weekday"]
        )

    # Logical combinations
    elif filter_type == "and":
        return combine_filters_and(df, filter_spec["filters"])
    elif filter_type == "or":
        return combine_filters_or(df, filter_spec["filters"])
    elif filter_type == "not":
        return negate_filter(df, filter_spec["filter"])
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """
    Apply filters to a DataFrame and return the filtered DataFrame.

    Args:
        df: Input DataFrame
        filters: Filter specification

    Returns:
        Filtered DataFrame
    """
    mask = apply_filter(df, filters)
    return df[mask]


def filter_count(df: pd.DataFrame, filters: Dict) -> int:
    """
    Count how many rows match the given filters.

    Args:
        df: Input DataFrame
        filters: Filter specification

    Returns:
        Count of matching rows
    """
    mask = apply_filter(df, filters)
    return mask.sum()


def get_filter_stats(
    df: pd.DataFrame, filters: Dict, numeric_columns: Optional[List[str]] = None
) -> Dict:
    """
    Get basic statistics for filtered data.

    Args:
        df: Input DataFrame
        filters: Filter specification
        numeric_columns: Optional list of numeric columns to calculate stats for.
                        If None, will use all numeric columns.

    Returns:
        Dictionary of statistics
    """
    # Apply the filter
    mask = apply_filter(df, filters)
    filtered_df = df[mask]

    # Determine which columns to calculate stats for
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    # Calculate statistics
    stats = {
        "count": len(filtered_df),
        "percentage": (len(filtered_df) / len(df)) * 100 if len(df) > 0 else 0,
        "columns": {},
    }

    for col in numeric_columns:
        if col in filtered_df.columns:
            col_stats = filtered_df[col].describe().to_dict()
            # Convert numpy types to Python native types for better serialization
            stats["columns"][col] = {
                k: float(v) if isinstance(v, np.number) else v
                for k, v in col_stats.items()
            }

    return stats


def save_filter_definition(filter_spec: Dict, filename: str) -> None:
    """
    Save a filter definition to a JSON file.

    Args:
        filter_spec: Filter specification
        filename: Output JSON filename
    """
    import json

    with open(filename, "w") as f:
        json.dump(filter_spec, f, indent=2)


def load_filter_definition(filename: str) -> Dict:
    """
    Load a filter definition from a JSON file.

    Args:
        filename: Input JSON filename

    Returns:
        Filter specification dictionary
    """
    import json

    with open(filename, "r") as f:
        return json.load(f)
