"""
Utilities for adding styled pandas DataFrames as tables to PowerPoint slides.
Supports row, column, and cell-specific styling, merged cells, and data formatting.
"""

import copy  # For deep copying style dictionaries
import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pptx.dml.color import RGBColor
from pptx.enum.text import (  # MSO_ANCHOR is used for vertical alignment
    MSO_ANCHOR,
    PP_ALIGN,
)
from pptx.slide import Slide
from pptx.table import Table, _Cell
from pptx.text.text import TextFrame
from pptx.util import Inches, Pt

# --- Logging Setup ---
# Consider making logger configurable if this were a larger library
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Configuration & Helper Functions ---


def deep_merge_dicts(
    default_dict: Dict[str, Any], custom_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries, with values from custom_dict taking precedence.
    Handles nested dictionaries like 'borders'. Preserves the original dictionaries.

    Parameters:
    -----------
    default_dict : dict
        The base dictionary containing default values.
    custom_dict : dict
        The dictionary with custom values that will override defaults.

    Returns:
    --------
    dict
        A new dictionary containing the merged result.
    """
    if not isinstance(default_dict, dict):
        # If default isn't a dict, custom dict (if it's a dict) replaces it entirely,
        # otherwise return a deep copy of custom to avoid modifying it.
        return (
            copy.deepcopy(custom_dict) if isinstance(custom_dict, dict) else custom_dict
        )
    if not isinstance(custom_dict, dict):
        # If custom isn't a dict, it overrides entirely (even if default was dict)
        return custom_dict  # No need to copy primitive types

    # Start with a deep copy of the default to avoid modifying it
    result = copy.deepcopy(default_dict)

    for key, value in custom_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # For non-dict values or keys not originally in result,
            # simply use a deep copy of the custom value
            result[key] = copy.deepcopy(value)
    return result


def get_default_table_style() -> Dict[str, Any]:
    """Returns a default style configuration dictionary for tables."""
    # Structure remains the same as before
    return {
        "position": {"left": 1.0, "top": 1.0},
        "dimensions": {"width": 0, "height": 0},
        "row_heights": 0.4,
        "col_widths": 1.2,
        "number_formats": {},  # Renamed formatter, but key name kept for consistency
        "style_definitions": {
            "header": {
                "font_family": "Arial",
                "font_size": 12,
                "bold": True,
                "italic": False,
                "fill_color": (0, 112, 192),
                "font_color": (255, 255, 255),
                "h_align": "center",
                "v_align": "middle",
                "borders": {"bottom": {"width": 1.5, "color": (0, 0, 0)}},
            },
            "standard": {
                "font_family": "Arial",
                "font_size": 11,
                "bold": False,
                "italic": False,
                "fill_color": (255, 255, 255),
                "font_color": (0, 0, 0),
                "h_align": "left",
                "v_align": "middle",
                "borders": {"bottom": {"width": 0.75, "color": (192, 192, 192)}},
            },
            # ... other styles like subtotal, total, numeric_col etc. ...
            "subtotal": {
                "font_family": "Arial",
                "font_size": 11,
                "bold": True,
                "italic": False,
                "fill_color": (240, 240, 240),
                "font_color": (0, 0, 0),
                "h_align": "left",
                "v_align": "middle",
                "borders": {
                    "top": {"width": 1.0, "color": (0, 0, 0)},
                    "bottom": {"width": 1.0, "color": (0, 0, 0)},
                },
            },
            "total": {
                "font_family": "Arial",
                "font_size": 12,
                "bold": True,
                "italic": False,
                "fill_color": (217, 217, 217),
                "font_color": (0, 0, 0),
                "h_align": "left",
                "v_align": "middle",
                "borders": {
                    "top": {"width": 2.0, "color": (0, 0, 0)},
                    "bottom": {"width": 2.0, "color": (0, 0, 0)},
                },
            },
            "numeric_col": {"h_align": "right", "font_family": "Calibri"},
            "currency_col": {"h_align": "right", "font_family": "Calibri"},
            "category_col": {"bold": True, "h_align": "center"},
            "highlight_red": {"font_color": (200, 0, 0), "bold": True},
        },
        "row_styles": {0: "header"},
        "default_row_style": "standard",
        "column_styles": {},
        "cell_styles": {},
        "indent_cells": [],
        "indent_amount": 0.2,
        "merged_cells": [],
    }


def create_table_style(custom_style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Creates a complete table style by merging custom settings with defaults.
    Uses deep merge to handle nested dictionaries correctly.

    Parameters:
    -----------
    custom_style : dict, optional
        Custom style overrides to apply on top of the default style.

    Returns:
    --------
    dict
        A complete style settings dictionary.
    """
    base_style = get_default_table_style()
    if custom_style is None:
        return base_style
    # Ensure deep merge is used for proper nesting handling
    return deep_merge_dicts(base_style, custom_style)


# --- Data Formatting Module ---
class DataFormatter:
    """Handles formatting of data values (initially numbers, extendable)."""

    @staticmethod
    def format_value(value: Any, format_spec: Dict[str, Any]) -> str:
        """
        Formats a single value according to the specified format.
        Currently handles numeric types, returns others as strings.

        Parameters:
        -----------
        value : Any
            The value to format.
        format_spec : dict
            Dictionary with formatting specifications (type, decimal_places, etc.).

        Returns:
        --------
        str
            The formatted string representation of the value.
        """
        if pd.isna(value) or value is None:
            return ""

        format_type = format_spec.get("type", "auto")  # Default to auto-detect/string

        # --- Numeric Formatting Logic ---
        is_numeric = False
        numeric_value = None
        try:
            # More robust check for numeric types, handling strings with commas
            if isinstance(value, (int, float, np.number)):
                numeric_value = float(value)
                is_numeric = True
            elif isinstance(value, str):
                cleaned_value = value.replace(",", "")
                if (
                    cleaned_value.replace(".", "", 1).replace("-", "", 1).isdigit()
                ):  # Check if potentially numeric
                    numeric_value = float(cleaned_value)
                    is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False  # Failed conversion

        if is_numeric and format_type in [
            "number",
            "currency",
            "percentage",
            "thousands",
            "millions",
        ]:
            # Apply existing numeric formatting
            decimal_places = format_spec.get("decimal_places", 2)
            show_commas = format_spec.get("show_commas", True)
            symbol = format_spec.get("symbol", "$")
            value_to_format = numeric_value
            suffix = ""
            prefix = ""

            if format_type == "percentage":
                value_to_format *= 100
                suffix = "%"
            elif format_type == "currency":
                prefix = symbol
            elif format_type == "thousands":
                value_to_format /= 1000
                suffix = "K"
            elif format_type == "millions":
                value_to_format /= 1000000
                suffix = "M"

            fmt_str = (
                f"{{:,.{decimal_places}f}}"
                if show_commas
                else f"{{:.{decimal_places}f}}"
            )
            formatted = fmt_str.format(value_to_format)

            # Remove trailing zeros and decimal point unless currency
            if "." in formatted and format_type != "currency":
                formatted = formatted.rstrip("0").rstrip(".")

            if format_type == "currency":
                # Handle negative currency format ($ -value vs -$ value) - standard is -$value
                if numeric_value < 0:
                    return f"-{prefix}{formatted.lstrip('-')}"
                else:
                    return f"{prefix}{formatted}"
            return f"{formatted}{suffix}"

        # --- Add Date/Bool/Other Formatting Here (Future Enhancement) ---
        # Example placeholder:
        # if format_type == 'date' and isinstance(value, (datetime.date, datetime.datetime)):
        #     date_format_str = format_spec.get('format', '%Y-%m-%d')
        #     return value.strftime(date_format_str)

        # Default: return as string if no specific format applied
        return str(value)

    @staticmethod
    def apply_formats_to_dataframe(
        dataframe: pd.DataFrame, format_rules: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Applies formatting to specified columns in a DataFrame.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The DataFrame to format.
        format_rules : dict
            Dictionary mapping column names to format specifications. Renamed from number_formats.

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with formatted values as strings.
        """
        if not format_rules:
            # Return copy with all columns as string if no rules
            return dataframe.astype(str)

        # Use deep copy to avoid modifying original df during processing
        df_formatted = dataframe.copy(deep=True)

        for col_name, format_spec in format_rules.items():
            if col_name in df_formatted.columns:
                try:
                    df_formatted[col_name] = df_formatted[col_name].apply(
                        lambda x: DataFormatter.format_value(x, format_spec)
                    )
                except Exception as e:
                    logger.error(
                        f"Error formatting column '{col_name}': {e}", exc_info=True
                    )
                    # Optionally convert to string on error, or keep original error value
                    df_formatted[col_name] = df_formatted[col_name].astype(str)
            else:
                logger.warning(
                    f"Column '{col_name}' specified in format_rules not found in DataFrame."
                )

        # Ensure any columns *not* formatted are also strings for consistency
        for col in df_formatted.columns:
            if col not in format_rules:
                df_formatted[col] = df_formatted[col].astype(str)

        return df_formatted


# --- PowerPoint Table Formatter Class ---


class PowerPointTableFormatter:
    """
    Creates and styles a PowerPoint table from a pandas DataFrame.
    Supports row, column, and cell-specific styling, merged cells, data formatting.
    """

    # Alignment constants mapping
    _H_ALIGN_MAP = {
        "left": PP_ALIGN.LEFT,
        "center": PP_ALIGN.CENTER,
        "right": PP_ALIGN.RIGHT,
        "justify": PP_ALIGN.JUSTIFY,
    }
    _V_ALIGN_MAP = {
        "top": MSO_ANCHOR.TOP,
        "middle": MSO_ANCHOR.MIDDLE,
        "bottom": MSO_ANCHOR.BOTTOM,
    }

    def __init__(self, slide: Slide, dataframe: pd.DataFrame, style: Dict[str, Any]):
        """
        Initializes the formatter.

        Parameters:
        -----------
        slide : pptx.slide.Slide
            The PowerPoint slide to add the table to.
        dataframe : pd.DataFrame
            The data to populate the table. Must contain data convertible to string.
            Headers are derived from dataframe columns. Data starts from row 1.
        style : dict
            A style configuration dictionary (use create_table_style()).
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if not isinstance(style, dict):
            raise TypeError("Input 'style' must be a dictionary.")

        self.slide = slide
        # Keep a copy of the original for reference if needed, but operate on formatted data
        self.original_dataframe = dataframe.copy()
        self.style = style  # Assumed to be validated/merged by create_table_style
        self.n_rows = dataframe.shape[0] + 1  # +1 for header row
        self.n_cols = dataframe.shape[1]
        self.table: Optional[Table] = None  # Will hold the pptx Table object

        # Extract frequently used style parts for cleaner access
        self.style_definitions = self.style.get("style_definitions", {})
        self.row_style_map = self.style.get("row_styles", {0: "header"})
        self.default_row_style_key = self.style.get("default_row_style", "standard")
        self.column_style_map = self.style.get("column_styles", {})
        self.cell_style_map = self.style.get("cell_styles", {})
        self.merged_cells_list = self.style.get("merged_cells", [])
        self.indent_cells_list = self.style.get("indent_cells", [])
        self.indent_amount_val = self.style.get("indent_amount", 0.2)

        # Prepare data immediately - converts df to strings with formatting
        self.formatted_dataframe = self._prepare_data()

    def _prepare_data(self) -> pd.DataFrame:
        """Applies data formatting based on style configuration."""
        # Key name 'number_formats' kept in style dict for backward compatibility/clarity for now
        format_rules = self.style.get("number_formats", {})
        return DataFormatter.apply_formats_to_dataframe(
            self.original_dataframe, format_rules
        )

    def _create_table(self) -> Table:
        """Creates the table shape on the slide with calculated dimensions."""
        pos = self.style.get("position", {"left": 1.0, "top": 1.0})
        dim_override = self.style.get("dimensions", {"width": 0, "height": 0})
        col_widths = self.style.get("col_widths", 1.2)  # Inches or list
        row_heights = self.style.get("row_heights", 0.4)  # Inches or list

        # --- Determine Width ---
        if dim_override.get("width", 0) > 0:
            final_width = Inches(dim_override["width"])
        elif isinstance(col_widths, list):
            final_width = Inches(sum(w for w in col_widths if w > 0))
        elif col_widths > 0:
            final_width = Inches(col_widths * self.n_cols)
        else:  # Fallback if widths are zero or not specified correctly
            final_width = Inches(1.2 * self.n_cols)  # Default fallback width

        # --- Determine Height ---
        if dim_override.get("height", 0) > 0:
            final_height = Inches(dim_override["height"])
        elif isinstance(row_heights, list):
            final_height = Inches(sum(h for h in row_heights if h > 0))
        elif row_heights > 0:
            final_height = Inches(row_heights * self.n_rows)
        else:  # Fallback
            final_height = Inches(0.4 * self.n_rows)  # Default fallback height

        logger.info(
            f"Creating table shape at ({pos.get('left', 1.0)}, {pos.get('top', 1.0)}) "
            f"with size ({final_width/Inches(1):.2f}in, {final_height/Inches(1):.2f}in)"
        )

        table_shape = self.slide.shapes.add_table(
            self.n_rows,
            self.n_cols,
            Inches(pos.get("left", 1.0)),
            Inches(pos.get("top", 1.0)),
            final_width,
            final_height,
        )
        logger.info("Table shape created.")
        return table_shape.table

    def _apply_merges(self):
        """Applies cell merges as defined in the style configuration."""
        if self.table is None:
            logger.error("Cannot apply merges, table object is None.")
            return
        if not self.merged_cells_list:
            # logger.info("No cell merges specified.") # Reduce noise
            return

        logger.info(f"Applying {len(self.merged_cells_list)} cell merge(s)...")
        processed_cells = set()  # Track top-left cells involved in a merge

        for r1, c1, r2, c2 in self.merged_cells_list:
            # Basic coordinate validation
            if not (
                0 <= r1 < self.n_rows
                and 0 <= c1 < self.n_cols
                and r1 <= r2 < self.n_rows
                and c1 <= c2 < self.n_cols
            ):
                logger.warning(
                    f"Invalid merge range skipped: ({r1},{c1}) to ({r2},{c2}). "
                    f"Table size is ({self.n_rows}x{self.n_cols})."
                )
                continue

            # Check if start cell is already part of another merge initiated earlier
            if (r1, c1) in processed_cells:
                logger.warning(
                    f"Skipping merge for ({r1},{c1}) to ({r2},{c2}) as start cell "
                    f"({r1},{c1}) was already the start of a previous merge."
                )
                continue

            try:
                cell1 = self.table.cell(r1, c1)
                # Check if the target cell is already spanned (part of another merge)
                # This check helps prevent errors if merge ranges overlap implicitly
                if cell1.is_spanned:
                    logger.warning(
                        f"Skipping merge for ({r1},{c1}) to ({r2},{c2}) as start cell "
                        f"({r1},{c1}) is already part of a merged span (likely defined earlier)."
                    )
                    continue

                # Proceed with merge if start cell is not spanned
                cell2 = self.table.cell(r2, c2)
                logger.debug(f"Merging cells from ({r1},{c1}) to ({r2},{c2}).")
                cell1.merge(cell2)
                processed_cells.add((r1, c1))  # Mark this start cell as processed

            except Exception as e:
                # Catch potential errors during merge operation itself
                logger.error(
                    f"Failed to merge cells ({r1},{c1}) to ({r2},{c2}): {e}",
                    exc_info=True,
                )

        logger.info("Cell merges applied.")

    def _populate_table(self):
        """Fills the table with headers and formatted data, respecting merged cells."""
        if self.table is None:
            raise ValueError("Table object is None, cannot populate.")
        logger.info("Populating table with headers and data...")

        # Add column headers (using original dataframe for headers)
        for c_idx, col_name in enumerate(self.original_dataframe.columns):
            try:
                cell = self.table.cell(0, c_idx)
                # Only write text to the 'master' cell (top-left) of a merged range
                if not cell.is_spanned:
                    cell.text = str(col_name)
            except IndexError:
                logger.error(
                    f"IndexError populating header at column {c_idx}. Table columns: {self.n_cols}"
                )

        # Add formatted data rows (using formatted_dataframe)
        for r_idx_data, row_data in enumerate(
            self.formatted_dataframe.itertuples(index=False)
        ):
            r_idx_table = r_idx_data + 1  # Table row index (+1 for header)
            for c_idx, value in enumerate(row_data):
                try:
                    cell = self.table.cell(r_idx_table, c_idx)
                    # Only write text to the 'master' cell (top-left) of a merged range
                    if not cell.is_spanned:
                        cell.text = str(value)  # Ensure string conversion
                except IndexError:
                    logger.error(
                        f"IndexError populating data at cell ({r_idx_table},{c_idx}). Table size: ({self.n_rows}x{self.n_cols})"
                    )

        logger.info("Table populated.")

    def _apply_dimensions(self):
        """Applies row heights and column widths if specified in the style."""
        if self.table is None:
            raise ValueError("Table object is None, cannot apply dimensions.")

        row_heights = self.style.get("row_heights", None)
        col_widths = self.style.get("col_widths", None)
        apply_dims = False  # Flag to log only if action is taken

        # Apply row heights
        if row_heights is not None:
            apply_dims = True
            height_list = (
                [row_heights] * self.n_rows
                if isinstance(row_heights, (int, float))
                else row_heights
            )
            if len(height_list) > 0:  # Ensure list is not empty
                for i in range(self.n_rows):
                    try:
                        height_val = height_list[
                            i % len(height_list)
                        ]  # Cycle if needed
                        if height_val > 0:
                            self.table.rows[i].height = Inches(height_val)
                    except IndexError:
                        pass  # Should not happen with modulo

        # Apply column widths
        if col_widths is not None:
            apply_dims = True
            width_list = (
                [col_widths] * self.n_cols
                if isinstance(col_widths, (int, float))
                else col_widths
            )
            if len(width_list) > 0:
                for i in range(self.n_cols):
                    try:
                        width_val = width_list[i % len(width_list)]  # Cycle if needed
                        if width_val > 0:
                            self.table.columns[i].width = Inches(width_val)
                    except IndexError:
                        pass

        if apply_dims:
            logger.info("Applied specified table dimensions (rows/columns).")
        # else: logger.info("No specific row/column dimensions to apply.") # Reduce noise

    def _apply_font_style(self, text_frame: TextFrame, style: Dict[str, Any]):
        """Applies font settings from style dict to all runs in all paragraphs."""
        font_settings = {
            "name": style.get("font_family"),
            "size": Pt(style["font_size"]) if "font_size" in style else None,
            "bold": style.get("bold"),
            "italic": style.get("italic"),
            "color_rgb": (
                RGBColor(*style["font_color"]) if "font_color" in style else None
            ),
        }
        # Filter out None values
        active_font_settings = {k: v for k, v in font_settings.items() if v is not None}

        if not active_font_settings:
            return  # Nothing to apply

        for paragraph in text_frame.paragraphs:
            # Ensure at least one run exists to apply font to
            if not paragraph.runs:
                paragraph.add_run()

            for run in paragraph.runs:
                font = run.font
                if "name" in active_font_settings:
                    font.name = active_font_settings["name"]
                if "size" in active_font_settings:
                    font.size = active_font_settings["size"]
                if "bold" in active_font_settings:
                    font.bold = active_font_settings["bold"]
                if "italic" in active_font_settings:
                    font.italic = active_font_settings["italic"]
                if "color_rgb" in active_font_settings:
                    font.color.rgb = active_font_settings["color_rgb"]

    def _apply_cell_style(self, cell: _Cell, cell_style: Dict[str, Any]):
        """Applies comprehensive styling (fill, align, font, borders) to a single cell."""
        # Do not style spanned cells directly; their appearance is controlled by the master cell.
        if cell.is_spanned:
            return

        # --- Cell Fill ---
        if "fill_color" in cell_style:
            r, g, b = cell_style["fill_color"]
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(r, g, b)
        else:
            # If no fill specified in the *final* merged style for this cell, make it transparent.
            cell.fill.background()  # Explicitly set background fill (no color)

        # --- Vertical Alignment ---
        if "v_align" in cell_style and cell_style["v_align"] in self._V_ALIGN_MAP:
            cell.vertical_anchor = self._V_ALIGN_MAP[cell_style["v_align"]]

        # --- Text Frame / Paragraph / Font Formatting ---
        text_frame = cell.text_frame
        # Ensure word wrap is sensible (usually default is fine)
        text_frame.word_wrap = cell_style.get("word_wrap", True)
        # Apply margins if specified in style (optional)
        # text_frame.margin_left = Inches(cell_style.get('margin_left', 0.1)) ... etc.

        # Apply paragraph settings (like alignment)
        if not text_frame.paragraphs:
            text_frame.add_paragraph()  # Ensure paragraph exists
        for paragraph in text_frame.paragraphs:
            if "h_align" in cell_style and cell_style["h_align"] in self._H_ALIGN_MAP:
                paragraph.alignment = self._H_ALIGN_MAP[cell_style["h_align"]]
            # Apply other paragraph settings here if needed (space before/after, etc.)

        # Apply font settings using the helper method
        self._apply_font_style(text_frame, cell_style)

        # --- Border Formatting ---
        # Applies borders based *only* on the final calculated style for this cell.
        self._apply_border_formatting(cell, cell_style.get("borders", {}))

    def _apply_border_formatting(
        self, cell: _Cell, border_style: Dict[str, Dict[str, Any]]
    ):
        """Applies border styles defined in border_style dict to a cell."""
        # Again, skip spanned cells; borders are controlled by the master cell.
        if cell.is_spanned:
            return

        def apply_side(cell_border, side_style: Dict[str, Any]):
            """Helper to apply border properties if they exist in the style."""
            line_props = {}
            # Color must be RGB tuple
            if (
                "color" in side_style
                and isinstance(side_style["color"], tuple)
                and len(side_style["color"]) == 3
            ):
                line_props["color"] = RGBColor(*side_style["color"])
            # Width must be numeric
            if "width" in side_style and isinstance(side_style["width"], (int, float)):
                line_props["width"] = Pt(side_style["width"])
            # Dash style could be added here (e.g., 'dash_style': MSO_LINE_DASH_STYLE.DASH)
            if "dash_style" in side_style:
                line_props["dash_style"] = side_style[
                    "dash_style"
                ]  # Assumes correct MSO constant passed

            if line_props:
                try:
                    cell_border.update(line_props)
                except Exception as e:
                    logger.error(f"Error applying border properties {line_props}: {e}")
            # else: Consider if we need to explicitly "clear" the border if side_style is empty?
            # Setting width to 0 might work. E.g., cell_border.width = Pt(0)

        # Process each border side if defined in the style
        if "top" in border_style:
            apply_side(cell.border_top, border_style["top"])
        if "right" in border_style:
            apply_side(cell.border_right, border_style["right"])
        if "bottom" in border_style:
            apply_side(cell.border_bottom, border_style["bottom"])
        if "left" in border_style:
            apply_side(cell.border_left, border_style["left"])

    def _apply_table_styling(self):
        """
        Applies styling using a hierarchical approach:
        1. Base Row Style
        2. Column Style Override (merged onto row style)
        3. Cell Specific Override (merged onto the result)
        Handles warnings for missing style keys.
        """
        if self.table is None:
            raise ValueError("Table object is None, cannot apply styles.")
        logger.info("Applying table styling (row, column, cell precedence)...")

        # Get default row style definition once, ensure it's a dict
        default_row_style = self.style_definitions.get(self.default_row_style_key, {})
        if not isinstance(default_row_style, dict):
            logger.error(
                f"Default row style key '{self.default_row_style_key}' does not map to a dictionary in style_definitions. Using empty default."
            )
            default_row_style = {}

        # Cache looked-up style definitions to avoid repeated lookups/warnings
        resolved_style_defs = {self.default_row_style_key: default_row_style}

        for r_idx in range(self.n_rows):
            # --- 1. Determine Base Row Style ---
            row_style_key = self.row_style_map.get(r_idx, self.default_row_style_key)
            if row_style_key not in resolved_style_defs:
                resolved_style_defs[row_style_key] = self.style_definitions.get(
                    row_style_key, {}
                )
                if not resolved_style_defs[
                    row_style_key
                ]:  # Log warning only once per missing key
                    logger.warning(
                        f"Row style key '{row_style_key}' for row {r_idx} not found in style_definitions. Using default."
                    )
                    resolved_style_defs[row_style_key] = (
                        default_row_style  # Fallback to default
                    )

            base_row_style = resolved_style_defs[row_style_key]

            for c_idx in range(self.n_cols):
                cell = self.table.cell(r_idx, c_idx)
                # Skip styling spanned cells directly
                if cell.is_spanned:
                    continue

                # Start with a deep copy of the base row style for this cell
                # Use deepcopy to prevent modifications bleeding between cells
                current_cell_style = copy.deepcopy(base_row_style)

                # --- 2. Apply Column Style Override ---
                col_style_key = self.column_style_map.get(c_idx)
                if col_style_key:
                    if (
                        col_style_key not in resolved_style_defs
                    ):  # Check cache/definitions
                        resolved_style_defs[col_style_key] = self.style_definitions.get(
                            col_style_key, {}
                        )
                        if not resolved_style_defs[col_style_key]:
                            logger.warning(
                                f"Column style key '{col_style_key}' for column {c_idx} not found in style_definitions."
                            )

                    col_style_def = resolved_style_defs[col_style_key]
                    if col_style_def:  # Apply only if found
                        # logger.debug(f"Merging col style '{col_style_key}' onto base for cell ({r_idx},{c_idx})")
                        current_cell_style = deep_merge_dicts(
                            current_cell_style, col_style_def
                        )

                # --- 3. Apply Cell Specific Override ---
                # Cell overrides are direct style dictionaries, not keys
                cell_override_style = self.cell_style_map.get((r_idx, c_idx))
                if cell_override_style:
                    if isinstance(cell_override_style, dict):
                        # logger.debug(f"Merging cell override style for cell ({r_idx},{c_idx})")
                        current_cell_style = deep_merge_dicts(
                            current_cell_style, cell_override_style
                        )
                    else:
                        logger.warning(
                            f"Cell style override for ({r_idx},{c_idx}) is not a dictionary, skipping."
                        )

                # --- 4. Apply the final merged style to the cell ---
                try:
                    self._apply_cell_style(cell, current_cell_style)
                except Exception as e:
                    logger.error(
                        f"Failed to apply final style to cell ({r_idx},{c_idx}): {e}",
                        exc_info=True,
                    )

        logger.info("Table styling applied.")

    def _apply_indentation(self):
        """Applies text indentation to specified cells, respecting merged cells."""
        if self.table is None:
            raise ValueError("Table object is None, cannot apply indentation.")
        if not self.indent_cells_list or self.indent_amount_val <= 0:
            # logger.info("No indentation specified or amount is zero.") # Reduce noise
            return

        logger.info(
            f"Applying indentation ({self.indent_amount_val:.2f}in) to {len(self.indent_cells_list)} specified cells..."
        )
        indent_val = Inches(self.indent_amount_val)

        for row_idx, col_idx in self.indent_cells_list:
            # Validate indices
            if not (0 <= row_idx < self.n_rows and 0 <= col_idx < self.n_cols):
                logger.warning(
                    f"Invalid cell index for indentation skipped: ({row_idx}, {col_idx})"
                )
                continue

            try:
                cell = self.table.cell(row_idx, col_idx)
                # Only apply indentation to the 'master' cell (top-left) of a merged range
                if not cell.is_spanned:
                    text_frame = cell.text_frame
                    if not text_frame.paragraphs:
                        text_frame.add_paragraph()  # Ensure paragraph exists
                    for paragraph in text_frame.paragraphs:
                        # Set indent on paragraph format
                        pf = paragraph.paragraph_format
                        # Ensure level 0 for standard indent (might interact with bullet points later)
                        if paragraph.level != 0:
                            paragraph.level = 0
                        pf.left_indent = indent_val
                # else: logger.debug(f"Skipping indentation for spanned cell ({row_idx},{col_idx})") # Optional debug
            except Exception as e:
                logger.error(
                    f"Failed to apply indent to cell ({row_idx},{col_idx}): {e}",
                    exc_info=True,
                )

        logger.info("Indentation applied.")

    def add_to_slide(self) -> Table:
        """
        Executes the full process to create, populate, merge, and style the table.

        Returns:
        --------
        pptx.table.Table
            The created and formatted PowerPoint table object.

        Raises:
        ------
        Exception
             If any fatal error occurs during the process.
        """
        logger.info(
            f"Starting process to add DataFrame ({self.n_rows-1} rows x {self.n_cols} cols) as table."
        )
        try:
            self.table = self._create_table()
            self._apply_merges()  # Merge cells first
            self._populate_table()  # Then populate data
            self._apply_dimensions()  # Then set dimensions
            self._apply_table_styling()  # Then apply styles
            self._apply_indentation()  # Finally apply indentation
            logger.info("Table successfully added and formatted on slide.")
            return self.table
        except Exception as e:
            logger.error(
                f"Fatal error occurred while adding table to slide: {e}", exc_info=True
            )
            # Depending on desired behavior, you might cleanup here (e.g., remove partially created shape)
            raise  # Re-raise the exception to signal failure


# --- Main Public Function ---


def add_dataframe_as_table(
    slide: Slide, dataframe: pd.DataFrame, style_config: Optional[Dict[str, Any]] = None
) -> Optional[Table]:  # Return optional Table in case of failure
    """
    High-level function to add a pandas DataFrame as a styled table to a PowerPoint slide.

    Parameters:
    -----------
    slide : pptx.slide.Slide
        The PowerPoint slide object.
    dataframe : pd.DataFrame
        The data to display. Column names become headers.
    style_config : dict, optional
        A dictionary with custom style settings. If None, default styles are used.
        Use `create_table_style()` to generate or merge styles safely.

    Returns:
    --------
    pptx.table.Table or None
        The created PowerPoint table object, or None if a fatal error occurred.
    """
    try:
        # Ensure we have a valid, merged style configuration
        final_style = create_table_style(custom_style=style_config)

        # Instantiate the formatter and run the process
        formatter = PowerPointTableFormatter(slide, dataframe, final_style)
        table = formatter.add_to_slide()
        return table
    except TypeError as te:  # Catch type errors from init
        logger.error(
            f"Type error during table formatter initialization: {te}", exc_info=True
        )
        return None
    except Exception as e:
        # Errors during add_to_slide are already logged, just return None
        logger.error(f"Table creation failed. See previous logs. Error: {e}")
        return None


# # --- Example Usage ---
# # (Example usage block remains the same as the previous version,
# #  demonstrating the configuration options)
# if __name__ == "__main__":

#     # Sample DataFrame with repeated categories suitable for merging
#     data = {
#         "Region": [
#             "North",
#             "North",
#             "North",
#             "South",
#             "South",
#             "West",
#             "West",
#             "Summary",
#         ],
#         "Product": [
#             "A - Extended Name",
#             "B",
#             "C",
#             "A",
#             "B",
#             "B",
#             "C",
#             "Total",
#         ],  # Added long name
#         "Sales": [12000, 8500, 21000, 15500, 9200, 18000, 11500, 95700],
#         "Quota": [10000, 9000, 20000, 16000, 9000, 17500, 12000, 93500],
#         "Variance": [0.20, -0.055, 0.05, -0.03125, 0.0222, 0.02857, -0.04167, np.nan],
#     }
#     df = pd.DataFrame(data)

#     prs = Presentation()
#     slide_layout = prs.slide_layouts[5]
#     slide = prs.slides.add_slide(slide_layout)
#     title = slide.shapes.title
#     title.text = "Sales Report - Refactored & Styled Table"

#     # --- Define Custom Style Overrides ---
#     custom_settings = {
#         "position": {"left": 0.5, "top": 1.2},
#         "col_widths": [
#             1.0,
#             1.5,
#             1.2,
#             1.2,
#             1.2,
#         ],  # Adjusted width for longer product name
#         "row_heights": 0.3,
#         "number_formats": {  # Use the new formatter name internally
#             "Sales": {"type": "currency", "decimal_places": 0, "symbol": "$"},
#             "Quota": {"type": "currency", "decimal_places": 0, "symbol": "$"},
#             "Variance": {"type": "percentage", "decimal_places": 1},
#         },
#         "style_definitions": {  # Only overrides needed
#             "header": {
#                 "fill_color": (79, 129, 189),
#                 "font_size": 10,
#                 "borders": {"bottom": {"width": 2.0}},
#             },
#             "standard": {
#                 "font_size": 9,
#                 "borders": {"bottom": {"width": 0.5, "color": (200, 200, 200)}},
#             },
#             "total": {
#                 "font_size": 10,
#                 "bold": True,
#                 "fill_color": (220, 220, 220),
#                 "borders": {"top": {"width": 1.5}, "bottom": {"width": 1.5}},
#             },
#             # highlight_red and category_col defined in defaults are sufficient
#         },
#         "row_styles": {0: "header", 8: "total"},
#         "column_styles": {
#             0: "category_col",
#             2: "currency_col",
#             3: "currency_col",
#             4: "numeric_col",
#         },
#         "cell_styles": {
#             (2, 4): {
#                 "font_color": (200, 0, 0),
#                 "bold": True,
#             },  # Highlight negative variance (direct style)
#             (4, 4): {"font_color": (200, 0, 0), "bold": True},
#             (7, 4): {"font_color": (200, 0, 0), "bold": True},
#             (1, 1): {
#                 "fill_color": (255, 255, 200)
#             },  # Highlight specific cell background
#         },
#         "merged_cells": [
#             (1, 0, 3, 0),
#             (4, 0, 5, 0),
#             (6, 0, 7, 0),  # Merge Region cells
#             (8, 0, 8, 1),  # Merge 'Summary' text in total row
#         ],
#         "indent_cells": [
#             (r, 1) for r in range(1, 8)
#         ],  # Indent all product names (rows 1-7, col 1)
#         "indent_amount": 0.15,
#     }

#     table_style = create_table_style(custom_style=custom_settings)

#     logger.info("--- Adding Table to Slide ---")
#     added_table = add_dataframe_as_table(slide, df, table_style)

#     if added_table:
#         logger.info("--- Table added successfully. ---")
#     else:
#         logger.error("--- Failed to add table to slide. Check logs above. ---")

#     output_filename = "refactored_table_final_example.pptx"
#     prs.save(output_filename)
#     print(f"\nPresentation saved as {output_filename}")
