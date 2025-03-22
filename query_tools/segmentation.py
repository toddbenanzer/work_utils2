"""
Functions for creating and managing segments within data for analysis.
This module provides utilities to define segments that can be used for grouping and analysis.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def create_segment(
    df: pd.DataFrame, segment_definition: Dict[str, Any]
) -> pd.DataFrame:
    """
    Create a segment column in a DataFrame based on the provided definition.

    Args:
        df: Input DataFrame
        segment_definition: Dictionary defining the segment
            Must include 'type' and appropriate parameters for that type

    Returns:
        DataFrame with added segment column
    """
    if "type" not in segment_definition:
        raise ValueError("Segment definition must include 'type'")

    if "name" not in segment_definition:
        raise ValueError("Segment definition must include 'name' for the output column")

    segment_type = segment_definition["type"]
    segment_name = segment_definition["name"]

    result = df.copy()

    if segment_type == "direct":
        # Simply use an existing column as a segment
        # Example: {"type": "direct", "name": "state_segment", "source_column": "state"}
        if "source_column" not in segment_definition:
            raise ValueError("Direct segment must specify 'source_column'")

        source_column = segment_definition["source_column"]
        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found in DataFrame")

        result[segment_name] = df[source_column]

    elif segment_type == "mapping":
        # Map values from source column to segment values
        # Example: {"type": "mapping", "name": "account_tier", "source_column": "account_type",
        #           "mapping": {"checking": "basic", "premium_checking": "premium"}}
        if (
            "source_column" not in segment_definition
            or "mapping" not in segment_definition
        ):
            raise ValueError(
                "Mapping segment must specify 'source_column' and 'mapping'"
            )

        source_column = segment_definition["source_column"]
        mapping = segment_definition["mapping"]
        default = segment_definition.get("default", "other")

        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found in DataFrame")

        # Apply mapping with default for unmapped values
        result[segment_name] = df[source_column].map(mapping).fillna(default)

    elif segment_type == "bins":
        # Create segments from continuous data by binning
        # Example: {"type": "bins", "name": "balance_tier", "source_column": "balance",
        #           "bins": [0, 1000, 5000, 10000], "labels": ["low", "medium", "high"]}
        if not all(k in segment_definition for k in ["source_column", "bins"]):
            raise ValueError("Bins segment must specify 'source_column' and 'bins'")

        source_column = segment_definition["source_column"]
        bins = segment_definition["bins"]
        labels = segment_definition.get("labels", None)

        if source_column not in df.columns:
            raise ValueError(f"Source column '{source_column}' not found in DataFrame")

        # Create the binned segment
        result[segment_name] = pd.cut(
            df[source_column], bins=bins, labels=labels, include_lowest=True
        )

    elif segment_type == "expression":
        # Create segments using a pandas query expression or lambda
        # Example: {"type": "expression", "name": "high_value",
        #           "expression": "balance > 10000 and has_investments == True"}
        if "expression" not in segment_definition:
            raise ValueError("Expression segment must specify 'expression'")

        expression = segment_definition["expression"]

        if callable(expression):
            # It's a function that takes a DataFrame and returns a boolean Series
            result[segment_name] = expression(df)
        else:
            # It's a string query expression
            try:
                mask = df.eval(expression)
                result[segment_name] = np.where(mask, "Yes", "No")
            except Exception as e:
                raise ValueError(f"Error evaluating expression: {str(e)}")

    elif segment_type == "custom":
        # Apply a custom segmentation function
        # Example: {"type": "custom", "name": "custom_segment", "function": my_segment_function}
        if "function" not in segment_definition:
            raise ValueError("Custom segment must specify 'function'")

        custom_func = segment_definition["function"]
        if not callable(custom_func):
            raise TypeError("Custom segment 'function' must be callable")

        try:
            result[segment_name] = custom_func(df)
        except Exception as e:
            raise ValueError(f"Error applying custom segmentation function: {str(e)}")

    else:
        raise ValueError(f"Unknown segment type: {segment_type}")

    return result


def apply_segments(
    df: pd.DataFrame, segment_definitions: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Apply multiple segment definitions to a DataFrame.

    Args:
        df: Input DataFrame
        segment_definitions: List of segment definition dictionaries

    Returns:
        DataFrame with all segments added
    """
    result = df.copy()

    for segment_def in segment_definitions:
        result = create_segment(result, segment_def)

    return result


def balance_tiering(
    df: pd.DataFrame,
    balance_col: str,
    tiers: List[float],
    labels: Optional[List[str]] = None,
    output_col: str = "balance_tier",
) -> pd.DataFrame:
    """
    Create balance tier segments based on account balances.
    Convenience function for a common segmentation need.

    Args:
        df: Input DataFrame
        balance_col: Column containing balance values
        tiers: List of tier boundaries (including min and max)
        labels: Optional labels for the tiers (len should be one less than tiers)
        output_col: Name for the output tier column

    Returns:
        DataFrame with balance tier column added
    """
    segment_def = {
        "type": "bins",
        "name": output_col,
        "source_column": balance_col,
        "bins": tiers,
        "labels": labels,
    }

    return create_segment(df, segment_def)


def create_date_segments(
    df: pd.DataFrame, date_col: str, period: str = "M", output_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Create time-based segments from a date column.

    Args:
        df: Input DataFrame
        date_col: Column containing date values
        period: Time period for grouping ('D' for day, 'W' for week, 'M' for month, 'Q' for quarter, 'Y' for year)
        output_col: Name for the output column (defaults to {date_col}_{period})

    Returns:
        DataFrame with date segment column added
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")

    result = df.copy()

    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
        try:
            result[date_col] = pd.to_datetime(result[date_col])
        except:
            raise ValueError(f"Column '{date_col}' could not be converted to datetime")

    # Determine output column name if not provided
    if output_col is None:
        output_col = f"{date_col}_{period}"

    # Create the appropriate date segment based on period
    if period == "D":
        result[output_col] = result[date_col].dt.date
    elif period == "W":
        result[output_col] = result[date_col].dt.to_period("W").dt.start_time.dt.date
    elif period == "M":
        result[output_col] = result[date_col].dt.to_period("M").astype(str)
    elif period == "Q":
        result[output_col] = result[date_col].dt.to_period("Q").astype(str)
    elif period == "Y":
        result[output_col] = result[date_col].dt.year
    else:
        raise ValueError(
            f"Unsupported period: {period}. Use 'D', 'W', 'M', 'Q', or 'Y'"
        )

    return result


def combine_segments(
    df: pd.DataFrame, segment_cols: List[str], output_col: str, separator: str = "_"
) -> pd.DataFrame:
    """
    Combine multiple segment columns into a single composite segment.

    Args:
        df: Input DataFrame
        segment_cols: List of segment columns to combine
        output_col: Name for the output combined segment column
        separator: String to use between segment values

    Returns:
        DataFrame with combined segment column added
    """
    # Verify all segment columns exist
    missing_cols = [col for col in segment_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Segment columns not found in DataFrame: {missing_cols}")

    result = df.copy()

    # Convert all segment columns to strings and combine
    segments = [result[col].astype(str) for col in segment_cols]
    result[output_col] = segments[0]

    for i in range(1, len(segments)):
        result[output_col] = result[output_col] + separator + segments[i]

    return result
