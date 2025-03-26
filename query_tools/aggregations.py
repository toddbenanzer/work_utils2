"""
Aggregation tools for working with pandas DataFrames.

Provides a flexible way to perform groupby operations and aggregations using a
dictionary-based interface. Includes support for binning numeric columns with
human-readable labels.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ===== Binning functions =====


def create_bins(
    df: pd.DataFrame,
    column: str,
    method: str = "equal_width",
    n_bins: int = 5,
    bin_edges: Optional[List[float]] = None,
    bin_labels: Optional[List[str]] = None,
    include_missing: bool = True,
    format_template: Optional[str] = None,
    sortable: bool = False,
) -> pd.Series:
    """
    Create bins from a numeric column.

    Args:
        df: Input DataFrame
        column: Column name to bin
        method: Binning method ('equal_width', 'equal_freq', or 'custom')
        n_bins: Number of bins to create (used for equal_width and equal_freq)
        bin_edges: Custom bin edges for 'custom' method (including min and max values)
        bin_labels: Optional custom labels for bins
        include_missing: Whether to include a separate category for missing values
        format_template: Optional template for bin labels
        sortable: Optional flag to add a numeric prefix for easy sorting

    Returns:
        Series with bin labels
    """
    # Handle missing values
    if include_missing:
        # Create a copy of the column with nulls preserved for later handling
        series = df[column].copy()
        data_for_binning = df[column].dropna()
    else:
        series = df[column].copy()
        data_for_binning = df[column].fillna(
            df[column].min() - 1
        )  # Put nulls in lowest bin

    # Determine bin edges based on method
    if method == "custom" and bin_edges is not None:
        # Ensure custom bin edges are sorted and unique
        edges = sorted(set(bin_edges))

        # Ensure we cover the full range by adding -inf and inf if needed
        if edges[0] > float("-inf"):
            edges.insert(0, float("-inf"))
        if edges[-1] < float("inf"):
            edges.append(float("inf"))

    elif method == "equal_width":
        min_val = data_for_binning.min()
        max_val = data_for_binning.max()
        if min_val == max_val:  # Handle case where all values are the same
            edges = [float("-inf"), min_val, float("inf")]
        else:
            # Create equal width bins between min and max
            inner_edges = np.linspace(min_val, max_val, n_bins + 1)
            # Add outer bounds to ensure all values are covered
            edges = [float("-inf")] + list(inner_edges[1:-1]) + [float("inf")]

    elif method == "equal_freq":
        if len(data_for_binning) == 0:
            # Handle empty data case
            edges = [float("-inf"), float("inf")]
        else:
            # Create quantile-based bins
            quantiles = np.linspace(0, 1, n_bins + 1)
            inner_edges = data_for_binning.quantile(
                quantiles
            ).unique()  # Use unique to avoid duplicates
            # Add outer bounds
            edges = [float("-inf")] + list(inner_edges[1:-1]) + [float("inf")]

    else:
        raise ValueError(f"Unknown binning method: {method}")

    # Convert edges to bin intervals
    bin_intervals = list(zip(edges[:-1], edges[1:]))

    # Generate default human-readable labels if not provided
    if bin_labels is None:
        bin_labels = _format_bin_labels(bin_intervals, format_template, sortable)

    # Ensure we have enough labels
    if len(bin_labels) != len(bin_intervals):
        raise ValueError(
            f"Number of bin labels ({len(bin_labels)}) doesn't match number of bins ({len(bin_intervals)})"
        )

    # Create a dictionary to map intervals to labels
    interval_to_label = dict(zip(bin_intervals, bin_labels))

    # Apply binning using pd.cut
    binned = pd.cut(
        data_for_binning, bins=edges, labels=bin_labels, include_lowest=True
    )

    # Map the binned values back to the original series
    result = pd.Series(index=series.index, dtype="object")
    result.loc[~series.isna()] = binned.values

    # Handle missing values
    if include_missing:
        result.loc[series.isna()] = "Missing"

    return result


def _format_bin_labels(
    bin_intervals: List[Tuple[float, float]],
    template: Optional[str] = None,
    sortable: bool = False,
) -> List[str]:
    """
    Format bin intervals into human-readable labels.

    Args:
        bin_intervals: List of (lower, upper) bound tuples
        template: Optional format template with {lower} and {upper} placeholders

    Returns:
        List of formatted bin labels
    """
    labels = []

    for i, (lower, upper) in enumerate(bin_intervals):
        if template is not None:
            # Use provided template
            label = template.format(
                lower=lower if lower != float("-inf") else "Min",
                upper=upper if upper != float("inf") else "Max",
            )
        elif lower == float("-inf"):
            label = f"Less Than {upper:.0f}" if upper != float("inf") else "All Values"
        elif upper == float("inf"):
            label = f"{lower:.0f} or More"
        else:
            label = f"{lower:.0f} to {upper:.0f}"

        # Add sorting prefix if requested
        if sortable:
            # Format with leading zeros to ensure proper sorting with more than 9 bins
            prefix = f"{i+1:02d}:  " if len(bin_intervals) > 9 else f"{i+1}:  "
            label = prefix + label

        labels.append(label)

    return labels


# ===== Aggregation functions =====
_agg_functions = {
    "count": "count",
    "nunique": "nunique",
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "min",
    "max": "max",
    "std": "std",
    "var": "var",
    "geo_mean": lambda x: np.exp(np.mean(np.log(x))),
    "range": lambda x: x.max() - x.min(),
    "q1": lambda x: x.quantile(0.25),
    "q3": lambda x: x.quantile(0.75),
    "iqr": lambda x: x.quantile(0.75) - x.quantile(0.25),
    "mode": lambda x: x.mode().iloc[0] if not x.empty else np.nan,
}


def apply_aggregation(df: pd.DataFrame, aggregation_spec: Dict) -> pd.DataFrame:
    """
    Apply groupby and aggregation operations to a DataFrame.

    Args:
        df: Input DataFrame
        aggregation_spec: Dictionary defining the aggregation

    Returns:
        DataFrame with grouped and aggregated data

    Example aggregation specifications:
        {
            "groupby": ["category", "region"],
            "aggregations": [
                {"column": "sales", "function": "sum", "name": "Total Sales"},
                {"column": "quantity", "function": "mean", "name": "Avg Quantity"}
            ]
        }

        {
            "groupby": [
                {"column": "price", "bins": {"method": "equal_width", "n_bins": 5}},
                "product_category"
            ],
            "aggregations": [
                {"column": "quantity", "function": "sum"},
                {"column": "customer_id", "function": "nunique", "name": "Customer Count"}
            ]
        }
    """
    # Extract groupby columns and prepare any binning
    groupby_cols = []
    temp_columns = []  # Track temporary columns we create for binning

    for group_item in aggregation_spec.get("groupby", []):
        if isinstance(group_item, str):
            # Simple column name
            groupby_cols.append(group_item)
        elif isinstance(group_item, dict):
            # Column with binning
            column = group_item["column"]
            bin_spec = group_item.get("bins", {})

            # Create a binned version of the column
            bin_column_name = bin_spec.get("name", f"{column}_binned")
            temp_columns.append(bin_column_name)

            # Apply binning using the appropriate method
            binning_method = bin_spec.get("method", "equal_freq")

            if binning_method == "custom":
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method="custom",
                    bin_edges=bin_spec.get("bin_edges"),
                    bin_labels=bin_spec.get("bin_labels"),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            elif binning_method in ["equal_width", "equal_freq"]:
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method=binning_method,
                    n_bins=bin_spec.get("n_bins", 5),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            else:
                raise ValueError(f"Unknown binning method: {binning_method}")

            # Use the binned column in groupby
            groupby_cols.append(bin_column_name)
        else:
            raise ValueError(f"Invalid groupby specification: {group_item}")

    # If no groupby columns specified, return the original DataFrame
    if not groupby_cols:
        return df.copy()

    # Prepare aggregations
    aggs = {}
    for agg_spec in aggregation_spec.get("aggregations", []):
        column = agg_spec["column"]
        function = agg_spec["function"]
        output_name = agg_spec.get("name", f"{function}_{column}")

        aggs[output_name] = pd.NamedAgg(column=column, aggfunc=_agg_functions[function])

    # Apply groupby and aggregations
    if aggs:
        result = df.groupby(groupby_cols, dropna=False).agg(**aggs).reset_index()
    else:
        # If no aggregations are specified, just do a count
        result = df.groupby(groupby_cols, dropna=False).size().reset_index(name="count")

    # Clean up temporary columns
    for col in temp_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return result


def create_pivot_table(df: pd.DataFrame, pivot_spec: Dict) -> pd.DataFrame:
    """
    Create a pivot table from a DataFrame.

    Args:
        df: Input DataFrame
        pivot_spec: Dictionary defining the pivot operation

    Returns:
        Pivot table as a DataFrame

    Example pivot specifications:
        {
            "index": "category",
            "columns": "region",
            "values": "sales",
            "aggfunc": "sum",
            "fill_value": 0,
            "margins": True
        }

        {
            "index": [
                {"column": "price", "bins": {"method": "equal_width", "n_bins": 5}}
            ],
            "columns": "product_category",
            "values": ["quantity", "sales"],
            "aggfunc": {"quantity": "sum", "sales": "mean"},
            "fill_value": 0
        }
    """
    # Process index columns (with potential binning)
    index_cols = []
    temp_columns = []  # Track temporary columns we create for binning

    # Handle index columns (can be a string, list, or dict)
    index_spec = pivot_spec.get("index", [])
    if isinstance(index_spec, str):
        index_spec = [index_spec]

    for idx_item in index_spec:
        if isinstance(idx_item, str):
            # Simple column name
            index_cols.append(idx_item)
        elif isinstance(idx_item, dict):
            # Column with binning
            column = idx_item["column"]
            bin_spec = idx_item.get("bins", {})

            # Create a binned version of the column
            bin_column_name = f"{column}_binned"
            temp_columns.append(bin_column_name)

            # Apply binning based on method
            binning_method = bin_spec.get("method", "equal_width")

            if binning_method == "custom":
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method="custom",
                    bin_edges=bin_spec.get("bin_edges"),
                    bin_labels=bin_spec.get("bin_labels"),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            elif binning_method in ["equal_width", "equal_freq"]:
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method=binning_method,
                    n_bins=bin_spec.get("n_bins", 5),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            else:
                raise ValueError(f"Unknown binning method: {binning_method}")

            # Use the binned column in index
            index_cols.append(bin_column_name)
        else:
            raise ValueError(f"Invalid index specification: {idx_item}")

    # Handle columns (with potential binning)
    columns_spec = pivot_spec.get("columns")
    columns_col = None

    if columns_spec:
        if isinstance(columns_spec, str):
            # Simple column name
            columns_col = columns_spec
        elif isinstance(columns_spec, dict):
            # Column with binning
            column = columns_spec["column"]
            bin_spec = columns_spec.get("bins", {})

            # Create a binned version of the column
            bin_column_name = f"{column}_binned_col"
            temp_columns.append(bin_column_name)

            # Apply binning based on method
            binning_method = bin_spec.get("method", "equal_width")

            if binning_method == "custom":
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method="custom",
                    bin_edges=bin_spec.get("bin_edges"),
                    bin_labels=bin_spec.get("bin_labels"),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            elif binning_method in ["equal_width", "equal_freq"]:
                df[bin_column_name] = create_bins(
                    df,
                    column,
                    method=binning_method,
                    n_bins=bin_spec.get("n_bins", 5),
                    include_missing=bin_spec.get("include_missing", True),
                    format_template=bin_spec.get("format_template"),
                    sortable=bin_spec.get("sortable", False),
                )
            else:
                raise ValueError(f"Unknown binning method: {binning_method}")

            # Use the binned column for pivot columns
            columns_col = bin_column_name
        else:
            raise ValueError(f"Invalid columns specification: {columns_spec}")

    # Extract values and aggregation function
    values = pivot_spec.get("values")
    aggfunc = pivot_spec.get("aggfunc", "mean")
    fill_value = pivot_spec.get("fill_value", None)
    margins = pivot_spec.get("margins", False)
    margins_name = pivot_spec.get("margins_name", "All")

    # Create pivot table
    result = pd.pivot_table(
        df,
        index=index_cols if index_cols else None,
        columns=columns_col,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        margins_name=margins_name,
        dropna=False,
    )

    # Reset index for flat table if requested
    if pivot_spec.get("reset_index", True):
        result = result.reset_index()

    # Clean up temporary columns
    for col in temp_columns:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return result


# ===== Helper functions for summary statistics =====


def summarize_by_group(df: pd.DataFrame, summary_spec: Dict) -> pd.DataFrame:
    """
    Generate summary statistics by group.

    Args:
        df: Input DataFrame
        summary_spec: Dictionary defining the summary operation

    Returns:
        DataFrame with summary statistics

    Example summary specifications:
        {
            "groupby": "category",
            "columns": ["sales", "quantity"],
            "statistics": ["count", "mean", "min", "max", "sum"]
        }

        {
            "groupby": [
                {"column": "price", "bins": {"method": "equal_width", "n_bins": 5}}
            ],
            "columns": "sales",
            "statistics": [
                {"name": "Count", "function": "count"},
                {"name": "Average", "function": "mean"},
                {"name": "Total", "function": "sum"}
            ]
        }
    """
    # Extract groupby columns (with potential binning)
    groupby_spec = summary_spec.get("groupby", [])
    if isinstance(groupby_spec, str):
        groupby_spec = [groupby_spec]

    # Reuse the aggregation function to prepare groupby columns with binning
    groupby_agg_spec = {"groupby": groupby_spec}
    groupby_result = apply_aggregation(df, groupby_agg_spec)

    # Get the actual groupby column names after binning
    groupby_cols = [col for col in groupby_result.columns if col != "count"]

    # Extract columns to summarize
    columns_to_summarize = summary_spec.get("columns", [])
    if isinstance(columns_to_summarize, str):
        columns_to_summarize = [columns_to_summarize]

    # Extract statistics to compute
    statistics = summary_spec.get("statistics", ["count", "mean", "min", "max"])
    if isinstance(statistics, str):
        statistics = [statistics]

    # Create a list of aggregation specifications
    aggregations = []
    for column in columns_to_summarize:
        if isinstance(statistics[0], str):
            # Simple list of statistic names
            for stat in statistics:
                aggregations.append(
                    {"column": column, "function": stat, "name": f"{column}_{stat}"}
                )
        else:
            # List of dictionaries with custom names
            for stat_spec in statistics:
                aggregations.append(
                    {
                        "column": column,
                        "function": stat_spec["function"],
                        "name": f"{column}_{stat_spec.get('name', stat_spec['function'])}",
                    }
                )

    # Perform the aggregation
    agg_spec = {"groupby": groupby_spec, "aggregations": aggregations}

    return apply_aggregation(df, agg_spec)


# ===== Dictionary-based aggregation application =====


def process_aggregation(df: pd.DataFrame, agg_config: Dict) -> pd.DataFrame:
    """
    Process an aggregation configuration and apply it to a DataFrame.

    Args:
        df: Input DataFrame
        agg_config: Dictionary with aggregation configuration

    Returns:
        Processed DataFrame

    Example configuration:
        {
            "type": "aggregation",
            "spec": {
                "groupby": ["category", "region"],
                "aggregations": [
                    {"column": "sales", "function": "sum", "name": "Total Sales"}
                ]
            }
        }

        {
            "type": "pivot",
            "spec": {
                "index": "category",
                "columns": "region",
                "values": "sales",
                "aggfunc": "sum"
            }
        }

        {
            "type": "summary",
            "spec": {
                "groupby": "category",
                "columns": ["sales", "quantity"],
                "statistics": ["count", "mean", "sum"]
            }
        }
    """
    agg_type = agg_config.get("type", "aggregation").lower()
    spec = agg_config.get("spec", {})

    if agg_type == "aggregation":
        return apply_aggregation(df, spec)
    elif agg_type == "pivot":
        return create_pivot_table(df, spec)
    elif agg_type == "summary":
        return summarize_by_group(df, spec)
    else:
        raise ValueError(f"Unknown aggregation type: {agg_type}")
