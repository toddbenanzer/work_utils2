"""
Utilities for creating and formatting tables in PowerPoint slides.
"""

from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.slide import Slide as PptxSlide
from pptx.table import Table
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


def create_table(
    slide: PptxSlide,
    data: pd.DataFrame,
    left: Union[float, Inches],
    top: Union[float, Inches],
    width: Optional[Union[float, Inches]] = None,
    height: Optional[Union[float, Inches]] = None,
    style_settings: Optional[Dict[str, Any]] = None,
) -> Table:
    """
    Create a table on a slide from DataFrame data.

    Args:
        slide: PowerPoint slide to add table to
        data: DataFrame with table data
        left: Left position of table
        top: Top position of table
        width: Width of table (None for auto)
        height: Height of table (None for auto)
        style_settings: Table style settings from config

    Returns:
        PowerPoint Table object
    """
    # Get number of rows and columns
    n_rows = len(data) + 1  # +1 for header row
    n_cols = len(data.columns)
    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, height).table

    # Apply column widths if provided
    if style_settings and style_settings.get("column_widths"):
        for i, width_val in enumerate(style_settings["column_widths"]):
            if i < n_cols:
                table.columns[i].width = Inches(width_val)

    # Fill table with data
    # First, handle column headers
    if isinstance(data.columns, pd.MultiIndex):
        # Handle multi-level column headers
        col_offset = 0

        # Handle index names first (if any)
        if isinstance(data.index, pd.MultiIndex):
            for level, name in enumerate(data.index.names):
                if name:  # Only fill if the name exists
                    table.cell(0, level).text = str(name)
            col_offset = len(data.index.names)
        else:
            # Single index case
            index_name = data.index.name
            if index_name:
                table.cell(0, 0).text = str(index_name)
            col_offset = 1

        # Now handle the column headers
        for col_idx, col_values in enumerate(data.columns):
            if isinstance(col_values, tuple):
                # Multi-level column
                for level, value in enumerate(col_values):
                    table.cell(level, col_idx + col_offset).text = str(value)
            else:
                # Single level column
                table.cell(0, col_idx + col_offset).text = str(col_values)
    else:
        # Single-level column headers
        col_offset = 0

        # Handle index name (if any)
        if isinstance(data.index, pd.MultiIndex):
            for level, name in enumerate(data.index.names):
                if name:  # Only fill if the name exists
                    table.cell(0, level).text = str(name)
            col_offset = len(data.index.names)
        else:
            # Single index case
            index_name = data.index.name
            if index_name:
                table.cell(0, 0).text = str(index_name)
            col_offset = 1

        # Fill column headers
        for col_idx, col_name in enumerate(data.columns):
            table.cell(0, col_idx + col_offset).text = str(col_name)

    # Now fill the data cells
    if isinstance(data.index, pd.MultiIndex):
        # Handle multi-level indexes
        for row_idx, (idx, row) in enumerate(data.iterrows(), start=1):
            # Fill index columns
            if isinstance(idx, tuple):
                for level, idx_val in enumerate(idx):
                    table.cell(row_idx, level).text = str(idx_val)
            else:
                table.cell(row_idx, 0).text = str(idx)

            # Fill data columns
            for col_idx, value in enumerate(row):
                cell_text = str(value) if pd.notna(value) else ""
                table.cell(row_idx, col_idx + col_offset).text = cell_text
    else:
        # Single-level index
        for row_idx, (idx, row) in enumerate(data.iterrows(), start=1):
            # Fill index column
            table.cell(row_idx, 0).text = str(idx)

            # Fill data columns
            for col_idx, value in enumerate(row):
                cell_text = str(value) if pd.notna(value) else ""
                table.cell(row_idx, col_idx + 1).text = cell_text

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
