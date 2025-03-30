"""
Utility functions for PowerPoint data visualization.
"""

import json
import os
from typing import Any, Dict, Optional

import pandas as pd


def df_to_serializable(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a DataFrame to a serializable dictionary.

    Args:
        df: DataFrame to convert

    Returns:
        Dictionary representation of DataFrame
    """
    result = {
        "columns": list(df.columns),
        "index": list(df.index),
        "data": df.values.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        result["columns"] = [
            list(col) if isinstance(col, tuple) else col for col in df.columns
        ]
        result["is_multi_columns"] = True
    else:
        result["is_multi_columns"] = False

    # Handle multi-index index
    if isinstance(df.index, pd.MultiIndex):
        result["index"] = [
            list(idx) if isinstance(idx, tuple) else idx for idx in df.index
        ]
        result["is_multi_index"] = True
    else:
        result["is_multi_index"] = False

    return result


def serializable_to_df(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert a serializable dictionary back to a DataFrame.

    Args:
        data: Dictionary representation of DataFrame

    Returns:
        Reconstructed DataFrame
    """
    # Handle multi-index columns
    if data.get("is_multi_columns", False):
        columns = pd.MultiIndex.from_tuples(
            [tuple(col) if isinstance(col, list) else (col,) for col in data["columns"]]
        )
    else:
        columns = data["columns"]

    # Handle multi-index index
    if data.get("is_multi_index", False):
        index = pd.MultiIndex.from_tuples(
            [tuple(idx) if isinstance(idx, list) else (idx,) for idx in data["index"]]
        )
    else:
        index = data["index"]

    # Create DataFrame
    df = pd.DataFrame(data["data"], columns=columns, index=index)

    # Convert dtypes if possible
    for col, dtype in data.get("dtypes", {}).items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError):
                pass  # Skip if conversion fails

    return df


def save_slide_template(
    slide_config: Dict[str, Any], name: str, path: Optional[str] = None
) -> str:
    """
    Save a slide template configuration to a JSON file.

    Args:
        slide_config: Slide configuration dictionary
        name: Template name
        path: Directory to save template, or None for current directory

    Returns:
        Path to saved template file
    """
    # Set template name in config
    slide_config["template_name"] = name

    # Determine save path
    if path is None:
        path = os.getcwd()

    # Create directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)

    # Create filename
    filename = f"{name.lower().replace(' ', '_')}_template.json"
    filepath = os.path.join(path, filename)

    # Save to file
    with open(filepath, "w") as f:
        json.dump(slide_config, f, indent=2)

    return filepath


def load_slide_template(filepath: str) -> Dict[str, Any]:
    """
    Load a slide template configuration from a JSON file.

    Args:
        filepath: Path to template file

    Returns:
        Slide configuration dictionary
    """
    with open(filepath, "r") as f:
        return json.load(f)


def detect_pivotable_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze a DataFrame and suggest a pivot configuration.

    Args:
        df: DataFrame to analyze

    Returns:
        Suggested pivot specification
    """
    # If DataFrame has Multi-Index, it's likely already pivoted
    if isinstance(df.index, pd.MultiIndex) or isinstance(df.columns, pd.MultiIndex):
        return None

    # Check number of columns - if too few or too many, pivoting might not help
    if len(df.columns) <= 3 or len(df.columns) > 15:
        return None

    # Check column types to identify potential index and value columns
    categorical_columns = []
    numeric_columns = []

    for col in df.columns:
        if df[col].dtype in (
            "object",
            "category",
        ) or pd.api.types.is_datetime64_any_dtype(df[col]):
            categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    # Need at least one categorical and one numeric column for a meaningful pivot
    if not categorical_columns or not numeric_columns:
        return None

    # Select first 2 categorical columns as potential index/columns
    potential_index = categorical_columns[0] if categorical_columns else None
    potential_columns = categorical_columns[1] if len(categorical_columns) > 1 else None

    # If we don't have enough categorical columns, return None
    if not potential_index or not potential_columns:
        return None

    # Build a pivot spec
    pivot_spec = {
        "index": potential_index,
        "columns": potential_columns,
        "values": numeric_columns,
    }

    return pivot_spec


def auto_chart_type(df: pd.DataFrame) -> str:
    """
    Automatically determine the best chart type for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Suggested chart type
    """
    # Number of rows and columns
    n_rows = len(df)
    n_cols = (
        len(df.columns) - 1 if isinstance(df.index, pd.MultiIndex) else len(df.columns)
    )

    # If few rows with one or two series, use column chart
    if n_rows <= 7 and n_cols <= 2:
        return "column"

    # If many rows with one or two series, use line chart
    elif n_rows > 7 and n_cols <= 2:
        return "line"

    # If few rows with many series, use bar chart
    elif n_rows <= 7 and n_cols > 2:
        return "bar"

    # If time series (datetime index), use line chart
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        return "line"

    # If categorical index with numeric columns, use column chart
    elif df.index.dtype in ("object", "category"):
        return "column"

    # Default to column chart
    else:
        return "column"
