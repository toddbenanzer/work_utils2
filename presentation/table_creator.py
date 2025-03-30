"""
Utilities for creating and formatting tables in PowerPoint slides.
"""

from typing import Any, Dict, NamedTuple, Optional, Tuple, Union

import pandas as pd
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.table import Table, _Cell
from pptx.util import Inches, Pt


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., '#4472C4')

    Returns:
        Tuple of (R, G, B) values
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


class TableParams(NamedTuple):
    """Structure to hold calculated table parameters."""

    total_rows: int
    total_cols: int
    index_col_count: int
    data_col_count: int
    header_rows_needed: int
    data_rows_needed: int
    is_multi_index: bool
    is_multi_column: bool


def _calculate_table_parameters(data: pd.DataFrame) -> TableParams:
    """Calculates necessary parameters based on the DataFrame structure."""
    is_multi_index = isinstance(data.index, pd.MultiIndex)
    is_multi_column = isinstance(data.columns, pd.MultiIndex)

    index_col_count = len(data.index.names) if is_multi_index else 1
    data_col_count = len(data.columns)
    total_cols = index_col_count + data_col_count

    header_rows_needed = data.columns.nlevels if is_multi_column else 1
    data_rows_needed = len(data)
    total_rows = header_rows_needed + data_rows_needed

    return TableParams(
        total_rows=total_rows,
        total_cols=total_cols,
        index_col_count=index_col_count,
        data_col_count=data_col_count,
        header_rows_needed=header_rows_needed,
        data_rows_needed=data_rows_needed,
        is_multi_index=is_multi_index,
        is_multi_column=is_multi_column,
    )


def _determine_table_dimensions(
    width: Optional[Union[float, Inches]],
    height: Optional[Union[float, Inches]],
    style_settings: Dict[str, Any],
) -> Tuple[Inches, Inches]:
    """Determines the final width and height for the table shape."""
    actual_width = width
    actual_height = height

    if actual_width is None:
        default_width_inches = style_settings.get("width", 8.0)
        actual_width = Inches(default_width_inches)
        if actual_height is None:
            actual_height = Inches(0.1)  # Auto-height trick
    elif actual_height is None:
        actual_height = Inches(0.1)  # Auto-height trick

    # Ensure Inches type
    if isinstance(actual_width, (int, float)):
        actual_width = Inches(actual_width)
    if isinstance(actual_height, (int, float)):
        actual_height = Inches(actual_height)

    return actual_width, actual_height


def _apply_column_widths(
    table: Table, total_cols: int, style_settings: Dict[str, Any]
) -> None:
    """Applies column widths from style_settings if provided."""
    col_widths = style_settings.get("column_widths")
    if col_widths:
        for i, width_val in enumerate(col_widths):
            if i < total_cols:
                try:
                    # Ensure width_val is treated as inches if it's numeric
                    width_inches = (
                        Inches(width_val)
                        if isinstance(width_val, (int, float))
                        else width_val
                    )
                    table.columns[i].width = width_inches
                except Exception as e:
                    print(f"Warning: Could not set width for column {i}: {e}")


def _fill_headers(table: Table, data: pd.DataFrame, params: TableParams) -> None:
    """Fills the header rows of the table with index and column names."""
    # Fill Index Names (Row 0)
    if params.is_multi_index:
        for i, name in enumerate(data.index.names):
            if name is not None and i < params.index_col_count:
                table.cell(0, i).text = str(name)
    elif data.index.name is not None:
        table.cell(0, 0).text = str(data.index.name)

    # Fill Column Names
    for col_idx, col_value in enumerate(data.columns):
        target_col = col_idx + params.index_col_count
        if target_col >= params.total_cols:
            continue  # Safety break

        if params.is_multi_column:
            # col_value is a tuple (level0_val, level1_val, ...)
            for level, value in enumerate(col_value):
                if level < params.header_rows_needed:
                    table.cell(level, target_col).text = str(value)
        else:
            # col_value is a single name, goes in the first header row
            if 0 < params.header_rows_needed:  # Ensure header row exists
                table.cell(0, target_col).text = str(col_value)


def _fill_data(table: Table, data: pd.DataFrame, params: TableParams) -> None:
    """Fills the data cells of the table."""
    first_data_row_index = params.header_rows_needed

    if params.data_rows_needed <= 0:
        return  # No data rows to fill

    for row_offset, (index_val, data_row) in enumerate(data.iterrows()):
        current_row_index = first_data_row_index + row_offset
        if current_row_index >= params.total_rows:
            break  # Safety break

        # --- Fill index columns for the current data row ---
        if params.is_multi_index:
            if isinstance(index_val, tuple):
                for level, idx_val in enumerate(index_val):
                    if level < params.index_col_count:
                        table.cell(current_row_index, level).text = str(idx_val)
            else:  # Defensive check
                if 0 < params.index_col_count:
                    table.cell(current_row_index, 0).text = str(index_val)
        else:
            table.cell(current_row_index, 0).text = str(index_val)

        # --- Fill data columns for the current data row ---
        col_offset = 0
        # Need to iterate through the *Series* `data_row` correctly
        for value in data_row:  # Iterating directly over the Series gives values
            current_col_index = params.index_col_count + col_offset
            if current_col_index < params.total_cols:
                cell: _Cell = table.cell(current_row_index, current_col_index)
                cell.text = str(value) if pd.notna(value) else ""
            col_offset += 1


def create_table(
    slide,  # Assuming PptxSlide type hint is available
    data: pd.DataFrame,
    left: Union[float, Inches],
    top: Union[float, Inches],
    width: Optional[Union[float, Inches]] = None,
    height: Optional[Union[float, Inches]] = None,
    style_settings: Optional[Dict[str, Any]] = None,
) -> Optional[Table]:  # Return Optional[Table] to handle empty case gracefully
    """
    Create a table on a slide from DataFrame data (Refactored with Helpers).

    Handles single and multi-level indexes and columns.

    Args:
        slide: PowerPoint slide object to add the table to.
        data: DataFrame containing the table data.
        left: Left position of the table (can be float or Inches).
        top: Top position of the table (can be float or Inches).
        width: Optional width of the table. If None, uses default or calculates.
        height: Optional height of the table. If None, attempts auto-height.
        style_settings: Optional dictionary with style settings, including
                        'width' (default width in inches) and 'column_widths'
                        (list of widths in inches).

    Returns:
        PowerPoint Table object added to the slide, or None if data is empty.
    """
    if style_settings is None:
        style_settings = {}

    # 1. Calculate Core Parameters
    params = _calculate_table_parameters(data)

    # Handle empty DataFrame case
    if (
        params.total_rows <= params.header_rows_needed
        or params.total_cols <= params.index_col_count
    ):
        # Check if *only* headers exist or *only* index exists.
        # A table with just headers or just an index column might still be valid depending on requirements.
        # Let's refine the empty check: truly empty data rows *and* no columns beyond index?
        is_effectively_empty = (
            params.data_rows_needed <= 0 and params.data_col_count <= 0
        )

        if is_effectively_empty:
            print("Warning: DataFrame has no data content to display.")
            # Optionally create a placeholder or return None
            # shapes = slide.shapes
            # placeholder_width, placeholder_height = _determine_table_dimensions(width, height, style_settings)
            # if isinstance(left, (int, float)): left = Inches(left)
            # if isinstance(top, (int, float)): top = Inches(top)
            # table_shape = shapes.add_table(1, 1, left, top, placeholder_width, Inches(0.2))
            # table_shape.table.cell(0,0).text = "[No Data]"
            # return table_shape.table
            return None  # Returning None might be cleaner

    # 2. Determine Final Dimensions for the Shape
    actual_width, actual_height = _determine_table_dimensions(
        width, height, style_settings
    )

    # Ensure left/top are Inches
    if isinstance(left, (int, float)):
        left = Inches(left)
    if isinstance(top, (int, float)):
        top = Inches(top)

    # 3. Create Table Shape
    shapes = slide.shapes
    try:
        table_shape: GraphicFrame = shapes.add_table(
            params.total_rows, params.total_cols, left, top, actual_width, actual_height
        )
        table: Table = table_shape.table
    except Exception as e:
        print(f"Error creating table shape: {e}")
        print(f"Attempted rows: {params.total_rows}, cols: {params.total_cols}")
        return None  # Cannot proceed if shape creation fails

    # 4. Apply Column Widths
    _apply_column_widths(table, params.total_cols, style_settings)

    # 5. Fill Header Cells
    _fill_headers(table, data, params)

    # 6. Fill Data Cells
    _fill_data(table, data, params)

    return table


def format_table(
    table: Table, data: pd.DataFrame, style_settings: Optional[Dict[str, Any]] = None
) -> None:
    """
    Format a PowerPoint table with styling.

    Args:
        table: PowerPoint Table object to format
        data: DataFrame with the table data
        style_settings: Table style settings from config
    """
    if not style_settings:
        return

    # Get styling options
    has_header = style_settings.get("has_header", True)
    first_row = style_settings.get("first_row", True)
    banded_rows = style_settings.get("banded_rows", True)
    banded_columns = style_settings.get("banded_columns", False)

    # Get font settings
    font_name = style_settings.get("font_name", "Calibri")
    header_font_size = style_settings.get("header_font_size", 14)
    header_font_bold = style_settings.get("header_font_bold", True)
    body_font_size = style_settings.get("body_font_size", 12)
    body_font_bold = style_settings.get("body_font_bold", False)

    # Get color settings
    header_fill_color = style_settings.get("header_fill_color")
    header_font_color = style_settings.get("header_font_color")
    body_fill_color = style_settings.get("body_fill_color")
    body_font_color = style_settings.get("body_font_color")
    banded_fill_color = style_settings.get("banded_fill_color")

    # If the table has a header row
    if has_header and first_row:
        # Format header row
        for cell in table.rows[0].cells:
            # Set font properties
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.name = font_name
                    run.font.size = Pt(header_font_size)
                    run.font.bold = header_font_bold

                    # Set font color if specified
                    if header_font_color:
                        rgb = hex_to_rgb(header_font_color)
                        run.font.color.rgb = RGBColor(*rgb)

            # Set cell background color if specified
            if header_fill_color:
                rgb = hex_to_rgb(header_fill_color)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(*rgb)

            # Vertical alignment
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Format the data rows
    for row_idx, row in enumerate(table.rows[1:], start=1):
        for cell_idx, cell in enumerate(row.cells):
            # Set font properties
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
                for run in paragraph.runs:
                    run.font.name = font_name
                    run.font.size = Pt(body_font_size)
                    run.font.bold = body_font_bold

                    # Set font color if specified
                    if body_font_color:
                        rgb = hex_to_rgb(body_font_color)
                        run.font.color.rgb = RGBColor(*rgb)

            # Set cell background color for banded rows
            if banded_rows and row_idx % 2 == 0 and banded_fill_color:
                rgb = hex_to_rgb(banded_fill_color)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(*rgb)
            elif body_fill_color:
                rgb = hex_to_rgb(body_fill_color)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(*rgb)

            # Set cell background color for banded columns
            if banded_columns and cell_idx % 2 == 0 and banded_fill_color:
                rgb = hex_to_rgb(banded_fill_color)
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(*rgb)

            # Vertical alignment
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE

    # Set first column formatting if it's an index
    for row in table.rows[1:]:  # Skip header row
        cell = row.cells[0]
        # Make index cells bold
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.bold = True


def format_hierarchical_table(
    table: Table, data: pd.DataFrame, style_settings: Optional[Dict[str, Any]] = None
) -> None:
    """
    Format a table with hierarchical index/columns.

    Args:
        table: PowerPoint Table object to format
        data: DataFrame with hierarchical index/columns
        style_settings: Table style settings from config
    """
    # Format the main table first
    format_table(table, data, style_settings)

    # Special formatting for hierarchical columns/indices
    if isinstance(data.index, pd.MultiIndex) or isinstance(data.columns, pd.MultiIndex):
        # Get the number of index levels
        index_levels = (
            len(data.index.names) if isinstance(data.index, pd.MultiIndex) else 1
        )

        # Format the index columns (left columns)
        for row_idx in range(1, len(table.rows)):
            for col_idx in range(index_levels):
                cell = table.cell(row_idx, col_idx)

                # Make index cells italic
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.italic = True

                # Set background color for index cells (lighter shade)
                header_fill_color = (
                    style_settings.get("header_fill_color") if style_settings else None
                )
                if header_fill_color:
                    # Use a lighter shade for index cells
                    rgb = hex_to_rgb(header_fill_color)
                    # Make it lighter by mixing with white
                    lighter_rgb = tuple(int(0.7 * c + 0.3 * 255) for c in rgb)
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = RGBColor(*lighter_rgb)
