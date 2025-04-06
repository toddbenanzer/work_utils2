# -*- coding: utf-8 -*-
"""
Utilities for adding pandas DataFrame charts to PowerPoint slides.
Supports customizable stacked bar/column charts and can be extended.
"""

import copy
import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from pptx.chart.chart import Chart
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import (
    XL_CHART_TYPE,
    XL_LABEL_POSITION,
    XL_LEGEND_POSITION,
    XL_TICK_MARK,
)
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.shapes.graphfrm import GraphicFrame  # For type hinting chart_shape
from pptx.slide import Slide
from pptx.text.text import Font
from pptx.util import Inches, Pt

# --- Configuration ---
# Consider moving logging configuration outside this module if used application-wide
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Constants ---

# Mapping for data label positions
LABEL_POSITION_MAP: Dict[str, XL_LABEL_POSITION] = {
    "inside_end": XL_LABEL_POSITION.INSIDE_END,
    "outside_end": XL_LABEL_POSITION.OUTSIDE_END,
    "center": XL_LABEL_POSITION.CENTER,
    "inside_base": XL_LABEL_POSITION.INSIDE_BASE,
    "best_fit": XL_LABEL_POSITION.BEST_FIT,
    # Add other positions if needed
}

# Mapping for legend positions
LEGEND_POSITION_MAP: Dict[str, XL_LEGEND_POSITION] = {
    "top": XL_LEGEND_POSITION.TOP,
    "bottom": XL_LEGEND_POSITION.BOTTOM,
    "left": XL_LEGEND_POSITION.LEFT,
    "right": XL_LEGEND_POSITION.RIGHT,
    # Use CORNER for the top-right position as TOP_RIGHT does not exist
    "top_right": XL_LEGEND_POSITION.CORNER,
    "corner": XL_LEGEND_POSITION.CORNER,
}

# Mapping for tick mark types
TICK_MARK_MAP: Dict[str, XL_TICK_MARK] = {
    "none": XL_TICK_MARK.NONE,
    "inside": XL_TICK_MARK.INSIDE,
    "outside": XL_TICK_MARK.OUTSIDE,
    "cross": XL_TICK_MARK.CROSS,
}

# Default colors for series (used if not specified in style)
# Consider defining these based on your presentation template's theme colors
DEFAULT_COLORS: Tuple[str, ...] = (
    "3C2F80",
    "2C1F10",
    "4C4C4C",
    "7030A0",
    "0070C0",
    "00B050",
    "FFC000",
    "FF0000",
    "A9A9A9",
    "5A5A5A",
)


# --- Helper Functions ---


def deep_merge_dicts(
    default_dict: Dict[str, Any], custom_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. Values from custom_dict take precedence.
    Creates deep copies to avoid modifying original dictionaries.
    """
    if not isinstance(default_dict, dict):
        return copy.deepcopy(custom_dict)
    if not isinstance(custom_dict, dict):
        return copy.deepcopy(custom_dict)

    result = copy.deepcopy(default_dict)

    for key, value in custom_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)  # Ensure deep copy of custom values too
    return result


def get_default_chart_style() -> Dict[str, Any]:
    """Returns a default style configuration dictionary for charts."""
    # Defined as a function to ensure a fresh copy each time
    return {
        "chart_type": "stacked_bar",  # Default to horizontal stacked bar
        "position": {"left": 1.0, "top": 1.5},
        "dimensions": {"width": 8.0, "height": 5.0},
        "title": {
            "text": None,  # No title by default
            "visible": True,  # Controls if space is allocated (even if text is None)
            "font": {
                "name": "Calibri",
                "size": 14,
                "bold": True,
                "italic": False,
                "color": (0, 0, 0),
            },
        },
        "data_labels": {
            "enabled": True,
            "position": "center",
            "number_format": "0",  # Default to integer display
            "font": {
                "name": "Calibri",
                "size": 9,
                "bold": False,
                "color": (255, 255, 255),
            },
        },
        "legend": {
            "enabled": True,
            "position": "bottom",
            "include_in_layout": True,  # Let PowerPoint manage layout by default
            "font": {"name": "Calibri", "size": 10},
        },
        "category_axis": {
            "visible": True,
            "tick_labels": {"visible": True, "font": {"name": "Calibri", "size": 10}},
            "major_gridlines": False,
            "minor_gridlines": False,
            "tick_marks": "outside",
            "line": {"visible": True},  # Control axis line visibility
        },
        "value_axis": {
            "visible": True,
            "major_gridlines": True,
            "minor_gridlines": False,
            "tick_marks": "outside",
            "tick_labels": {"visible": True, "font": {"name": "Calibri", "size": 10}},
            "number_format": "General",  # Use Excel's default number format
            "max_scale": None,  # Auto-scale by default
            "min_scale": None,  # Auto-scale by default
            "line": {"visible": True},  # Control axis line visibility
        },
        "plot_area": {
            "border": {"visible": False},  # Control plot area border visibility
            "fill": {"type": "none"},  # 'none', 'solid', 'gradient', etc.
        },
        "gap_width": 150,  # Percentage space between categories (for bar/column)
        "overlap": 100,  # Percentage overlap for stacked (100) or clustered (-ve to +ve)
        "colors": {},  # Dictionary to map series names to specific hex colors
    }


def create_chart_style(custom_style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Creates a complete chart style by merging custom settings with defaults.
    """
    base_style = get_default_chart_style()
    if custom_style is None:
        return base_style
    # Ensure custom_style is a dict before merging
    if not isinstance(custom_style, dict):
        logger.warning("custom_style is not a dictionary, ignoring it.")
        return base_style
    return deep_merge_dicts(base_style, custom_style)


# --- Chart Utility Functions ---


def _apply_font_settings(font_obj: Font, font_config: Dict[str, Any]) -> None:
    """Applies common font settings from a config dict to a Font object."""
    font_obj.name = font_config.get("name", font_obj.name)
    if "size" in font_config:
        try:
            font_obj.size = Pt(int(font_config["size"]))
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid font size value: {font_config['size']}. Must be numeric."
            )
    if "bold" in font_config:
        font_obj.bold = bool(font_config["bold"])
    if "italic" in font_config:
        font_obj.italic = bool(font_config["italic"])

    color_val = font_config.get("color")
    if isinstance(color_val, tuple) and len(color_val) == 3:
        try:
            font_obj.color.rgb = RGBColor(
                int(color_val[0]), int(color_val[1]), int(color_val[2])
            )
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid RGB color tuple: {color_val}. Values must be integers 0-255."
            )
    elif isinstance(color_val, str):  # Allow hex string
        try:
            font_obj.color.rgb = RGBColor.from_string(color_val)
        except ValueError:
            logger.warning(f"Invalid hex color string: {color_val}")


def _get_chart_type_enum(type_str: str) -> XL_CHART_TYPE:
    """Maps a string chart type to its python-pptx enum."""
    type_map = {
        "stacked_bar": XL_CHART_TYPE.BAR_STACKED,  # Horizontal
        "stacked_column": XL_CHART_TYPE.COLUMN_STACKED,  # Vertical
        "clustered_bar": XL_CHART_TYPE.BAR_CLUSTERED,
        "clustered_column": XL_CHART_TYPE.COLUMN_CLUSTERED,
        "line": XL_CHART_TYPE.LINE,
        "line_markers": XL_CHART_TYPE.LINE_MARKERS,
        # Add more mappings as needed (pie, area, xy_scatter etc.)
    }
    default_type = XL_CHART_TYPE.BAR_STACKED
    enum_type = type_map.get(type_str.lower())
    if enum_type is None:
        logger.warning(
            f"Unsupported chart type '{type_str}'. Defaulting to {default_type.name}."
        )
        return default_type
    return enum_type


# --- PowerPoint Chart Formatter Class ---


class PowerPointChartFormatter:
    """
    Creates and styles a PowerPoint chart from a pandas DataFrame based on a style config.
    """

    def __init__(self, slide: Slide, dataframe: pd.DataFrame, style: Dict[str, Any]):
        """
        Initializes the chart formatter.

        Parameters:
        -----------
        slide : pptx.slide.Slide
            The PowerPoint slide to add the chart to.
        dataframe : pd.DataFrame
            The data to visualize. Assumes the first column contains categories
            and subsequent columns contain series data.
        style : dict
            A complete style configuration dictionary (use create_chart_style()).
        """
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError("Input 'dataframe' must be a non-empty pandas DataFrame.")
        if not isinstance(style, dict):
            # Should not happen if create_chart_style is used, but defensive check.
            raise TypeError("Input 'style' must be a dictionary.")
        if dataframe.shape[1] < 2:
            raise ValueError(
                "DataFrame must have at least two columns (categories + 1 series)."
            )

        self.slide = slide
        # Ensure we work with a copy, especially if manipulating data later
        self.dataframe = dataframe.copy()
        self.style = style  # Assumes style is already merged with defaults
        self.chart: Optional[Chart] = None  # Initialize chart attribute

    def _create_chart_object(self) -> None:
        """Creates the chart shape on the slide and populates with data."""
        pos_cfg = self.style.get("position", {})
        dim_cfg = self.style.get("dimensions", {})
        chart_type_str = self.style.get(
            "chart_type", "stacked_bar"
        )  # Default if missing

        xl_chart_type = _get_chart_type_enum(chart_type_str)

        # --- Prepare Chart Data ---
        chart_data = CategoryChartData()
        # Assume first column is category labels
        categories_col = self.dataframe.columns[0]
        # Ensure categories are strings for pptx
        chart_data.categories = self.dataframe[categories_col].astype(str).tolist()

        # Subsequent columns are series
        data_cols = self.dataframe.columns[1:]
        for col in data_cols:
            # Ensure numeric data, replace errors/NaNs with 0 for chart stability
            # You might want different NaN handling (e.g., None for gaps in line charts)
            series_data = (
                pd.to_numeric(self.dataframe[col], errors="coerce").fillna(0).tolist()
            )
            chart_data.add_series(str(col), series_data)  # Series name must be string

        # --- Add Chart Shape to Slide ---
        try:
            # Use .get with defaults for position/dimensions for safety
            left = Inches(pos_cfg.get("left", 1.0))
            top = Inches(pos_cfg.get("top", 1.5))
            width = Inches(dim_cfg.get("width", 8.0))
            height = Inches(dim_cfg.get("height", 5.0))

            chart_shape: GraphicFrame = self.slide.shapes.add_chart(
                xl_chart_type, left, top, width, height, chart_data
            )
            self.chart = chart_shape.chart
            title_text = self.style.get("title", {}).get("text", "Untitled")
            logger.info(
                f"Chart '{title_text}' created successfully (Type: {xl_chart_type.name})."
            )

        except Exception as e:
            logger.error(f"Failed to add chart shape to slide: {e}", exc_info=True)
            self.chart = None  # Ensure chart is None on failure
            raise  # Re-raise exception after logging

    def _apply_chart_title(self) -> None:
        """Applies title settings based on the style configuration."""
        if not self.chart:
            return
        title_config = self.style.get("title", {})
        title_text = title_config.get("text")
        is_visible = title_config.get("visible", True)  # Check visibility flag

        self.chart.has_title = bool(is_visible)  # Set visibility based on flag

        if self.chart.has_title and title_text:
            # Only set text and format if visible and text is provided
            self.chart.chart_title.has_text_frame = True
            tf = self.chart.chart_title.text_frame
            tf.text = str(title_text)  # Ensure text is string
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT  # Adjust title box size

            font_config = title_config.get("font")
            if font_config and tf.paragraphs:
                # Title text is usually in the first paragraph, first run
                run = (
                    tf.paragraphs[0].runs[0]
                    if tf.paragraphs[0].runs
                    else tf.paragraphs[0].add_run()
                )
                _apply_font_settings(run.font, font_config)

    def _apply_axis_settings(self) -> None:
        """Applies settings for category and value axes, including plot area border."""
        if not self.chart:
            return

        # --- Category Axis ---
        cat_axis_config = self.style.get("category_axis", {})
        if hasattr(self.chart, "category_axis"):
            category_axis = self.chart.category_axis
            category_axis.visible = cat_axis_config.get("visible", True)

            if category_axis.visible:
                # Gridlines
                category_axis.has_major_gridlines = cat_axis_config.get(
                    "major_gridlines", False
                )
                category_axis.has_minor_gridlines = cat_axis_config.get(
                    "minor_gridlines", False
                )

                # Tick Marks
                tick_mark_str = cat_axis_config.get("tick_marks", "none").lower()
                category_axis.major_tick_mark = TICK_MARK_MAP.get(
                    tick_mark_str, XL_TICK_MARK.NONE
                )
                category_axis.minor_tick_mark = (
                    XL_TICK_MARK.NONE
                )  # Minor usually not needed for category

                # Tick Labels Font and Visibility
                tick_labels_cfg = cat_axis_config.get("tick_labels", {})
                if hasattr(category_axis, "tick_labels"):
                    category_axis.has_tick_labels = tick_labels_cfg.get("visible", True)
                    if category_axis.has_tick_labels:
                        font_config = tick_labels_cfg.get("font")
                        if font_config:
                            _apply_font_settings(
                                category_axis.tick_labels.font, font_config
                            )

                # Axis Line Visibility
                line_cfg = cat_axis_config.get("line", {})
                if hasattr(category_axis.format, "line"):
                    if not line_cfg.get("visible", True):
                        category_axis.format.line.fill.background()  # Make line invisible
                    # TODO: Add line color/style customization if needed

        # --- Value Axis ---
        val_axis_config = self.style.get("value_axis", {})
        if hasattr(self.chart, "value_axis"):
            value_axis = self.chart.value_axis
            value_axis.visible = val_axis_config.get("visible", True)

            if value_axis.visible:
                # Gridlines
                value_axis.has_major_gridlines = val_axis_config.get(
                    "major_gridlines", True
                )
                value_axis.has_minor_gridlines = val_axis_config.get(
                    "minor_gridlines", False
                )

                # Tick Marks
                tick_mark_str = val_axis_config.get("tick_marks", "none").lower()
                value_axis.major_tick_mark = TICK_MARK_MAP.get(
                    tick_mark_str, XL_TICK_MARK.NONE
                )
                # You might want minor ticks configurable:
                # value_axis.minor_tick_mark = TICK_MARK_MAP.get(tick_mark_str, XL_TICK_MARK.NONE)

                # Tick Labels Font, Format, Visibility
                tick_labels_cfg = val_axis_config.get("tick_labels", {})
                if hasattr(value_axis, "tick_labels"):
                    value_axis.has_tick_labels = tick_labels_cfg.get("visible", True)
                    if value_axis.has_tick_labels:
                        font_config = tick_labels_cfg.get("font")
                        if font_config:
                            _apply_font_settings(
                                value_axis.tick_labels.font, font_config
                            )

                        # Number format for axis labels
                        number_format = val_axis_config.get("number_format", "General")
                        value_axis.tick_labels.number_format = number_format
                        # Link format to source only if using Excel's default
                        value_axis.tick_labels.number_format_is_linked = (
                            number_format == "General"
                        )

                # Axis Scale Limits
                max_scale = val_axis_config.get("max_scale")
                min_scale = val_axis_config.get("min_scale")
                if max_scale is not None:
                    try:
                        value_axis.maximum_scale = float(max_scale)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid max_scale: {max_scale}")
                if min_scale is not None:
                    try:
                        value_axis.minimum_scale = float(min_scale)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid min_scale: {min_scale}")

                # Axis Line Visibility
                line_cfg = val_axis_config.get("line", {})
                if hasattr(value_axis.format, "line"):
                    if not line_cfg.get("visible", True):
                        value_axis.format.line.fill.background()  # Make line invisible
                    # TODO: Add line color/style customization if needed

        # --- Plot Area Formatting (Border and Fill) ---
        plot_cfg = self.style.get("plot_area", {})
        plot_border_cfg = plot_cfg.get("border", {})

        # Check if the chart object actually has a plot_area attribute
        if hasattr(self.chart, "plot_area"):
            plot_area = self.chart.plot_area  # Access plot_area directly from the chart

            # --- Plot Area Border ---
            # Check visibility setting from style config
            if not plot_border_cfg.get("visible", False):
                # Check if the plot area has line formatting capabilities
                if hasattr(plot_area.format, "line") and plot_area.format.line:
                    # Make border invisible by setting its fill to background
                    plot_area.format.line.fill.background()
                    logger.debug("Plot area border set to invisible.")
            else:
                # TODO: Add plot area border color/style customization if visible
                # Example:
                # if hasattr(plot_area.format, 'line') and plot_area.format.line:
                #    line = plot_area.format.line
                #    line.color.rgb = RGBColor.from_string(plot_border_cfg.get("color", "000000"))
                #    line.width = Pt(plot_border_cfg.get("width", 1))
                #    # Add dash style etc.
                logger.debug(
                    "Plot area border visibility is enabled (styling not implemented yet)."
                )

            # --- Plot Area Fill ---
            # TODO: Add plot area fill customization (solid, gradient, none) based on plot_cfg.get("fill")
            # fill_cfg = plot_cfg.get("fill", {})
            # fill_type = fill_cfg.get("type", "none")
            # if hasattr(plot_area.format, 'fill'):
            #    fill = plot_area.format.fill
            #    if fill_type == "none":
            #        fill.background()
            #    elif fill_type == "solid":
            #        fill.solid()
            #        fill.fore_color.rgb = RGBColor.from_string(fill_cfg.get("color", "FFFFFF"))
            #    # Add gradient etc.

        else:
            logger.warning(
                "Chart object does not have a 'plot_area' attribute. Cannot apply plot area formatting."
            )

    def _apply_legend_settings(self) -> None:
        """Applies legend settings (position, visibility, font)."""
        if not self.chart:
            return
        legend_config = self.style.get("legend", {})
        self.chart.has_legend = legend_config.get("enabled", True)

        if (
            self.chart.has_legend
            and hasattr(self.chart, "legend")
            and self.chart.legend
        ):
            legend = self.chart.legend
            position_str = legend_config.get("position", "bottom").lower()
            legend.position = LEGEND_POSITION_MAP.get(
                position_str, XL_LEGEND_POSITION.BOTTOM  # Default to bottom if invalid
            )
            # Control whether legend overlaps chart area
            legend.include_in_layout = legend_config.get("include_in_layout", True)

            # Apply font settings to legend text
            font_config = legend_config.get("font")
            if font_config and hasattr(legend, "font"):
                _apply_font_settings(legend.font, font_config)

    def _apply_data_labels(self) -> None:
        """Applies data label settings (visibility, position, font, number format)."""
        if not self.chart or not self.chart.plots:  # Check plots exist
            return
        data_label_config = self.style.get("data_labels", {})
        # Assuming styling applies to the first plot (common for bar/column/line)
        plot = self.chart.plots[0]

        # Check if plot object supports data labels
        if not hasattr(plot, "has_data_labels"):
            logger.warning(
                f"Chart type {self.chart.chart_type.name} might not support plot-level data labels."
            )
            return

        plot.has_data_labels = data_label_config.get("enabled", False)

        if plot.has_data_labels and hasattr(plot, "data_labels"):
            data_labels = plot.data_labels
            position_str = data_label_config.get("position", "center").lower()
            data_labels.position = LABEL_POSITION_MAP.get(
                position_str, XL_LABEL_POSITION.CENTER  # Default to center
            )

            # Apply number format if specified
            number_format = data_label_config.get("number_format", "General")
            data_labels.number_format = number_format
            # Link format to source only if using Excel's default
            data_labels.number_format_is_linked = number_format == "General"

            # Apply font settings
            font_config = data_label_config.get("font")
            if font_config and hasattr(data_labels, "font"):
                _apply_font_settings(data_labels.font, font_config)

    def _apply_series_settings(self) -> None:
        """Applies settings per series, primarily colors."""
        if not self.chart or not self.chart.series:
            return

        series_colors = self.style.get("colors", {})  # Map series name -> hex color

        for i, series in enumerate(self.chart.series):
            # Determine color: Use specific color if defined, else default palette
            # Use series.name which should match the DataFrame column header
            color_hex = series_colors.get(series.name)
            if not color_hex:
                # Fallback to default color palette based on series index
                color_index = i % len(DEFAULT_COLORS)
                color_hex = DEFAULT_COLORS[color_index]
                # Log fallback only if specific colors were expected but not found for this series
                # if series_colors: # Only log if custom colors dict is not empty
                #    logger.debug(f"No specific color found for series '{series.name}'. Using default color {color_hex} at index {color_index}.")

            # Apply fill color (works for bar, column, area, pie points)
            try:
                if hasattr(series.format, "fill"):
                    fill = series.format.fill
                    fill.solid()  # Set to solid color fill
                    fill.fore_color.rgb = RGBColor.from_string(color_hex)
                else:
                    logger.debug(
                        f"Series '{series.name}' format does not have 'fill' attribute."
                    )

            except ValueError:
                logger.warning(
                    f"Invalid hex color string '{color_hex}' for series '{series.name}'. Skipping color."
                )
            except AttributeError as ae:
                # Catch potential issues if accessing fill properties fails
                logger.warning(
                    f"Could not set fill for series '{series.name}'. Format may differ or be unavailable. Error: {ae}"
                )

            # Apply line color/style (for line charts)
            # TODO: Extend this based on chart type if needed
            # if self.chart.chart_type in [XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS, XL_CHART_TYPE.XY_SCATTER_LINES] :
            #      if hasattr(series.format, 'line'):
            #         line = series.format.line
            #         line.color.rgb = RGBColor.from_string(color_hex) # Or use a separate line color config
            #         # line.width = Pt(2.0) # Example: Set line width

    def _apply_bar_column_settings(self) -> None:
        """Applies settings specific to bar/column charts (gap width, overlap)."""
        if not self.chart or not self.chart.plots:
            return  # Check plots exist

        # These settings are typically on the plot object for bar/column types
        plot = self.chart.plots[0]
        chart_type = self.chart.chart_type

        # Check if chart type supports these properties
        if chart_type in (
            XL_CHART_TYPE.BAR_STACKED,
            XL_CHART_TYPE.COLUMN_STACKED,
            XL_CHART_TYPE.BAR_CLUSTERED,
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            # Add other relevant types like BAR_STACKED_100, etc. if needed
        ):
            # Gap Width
            if hasattr(plot, "gap_width"):
                gap_width = self.style.get("gap_width", 150)  # Default from style
                try:
                    plot.gap_width = int(gap_width)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid gap_width: {gap_width}")
            else:
                logger.debug(
                    f"Plot for chart type {chart_type.name} has no 'gap_width' attribute."
                )

            # Overlap
            if hasattr(plot, "overlap"):
                overlap = self.style.get(
                    "overlap", 100 if "STACKED" in chart_type.name else 0
                )  # Sensible default based on type
                try:
                    plot.overlap = int(overlap)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid overlap: {overlap}")
            else:
                logger.debug(
                    f"Plot for chart type {chart_type.name} has no 'overlap' attribute."
                )

    def add_to_slide(self) -> Optional[Chart]:
        """
        Executes the full process: creates the chart object and applies all styles.

        Returns:
        --------
        Optional[pptx.chart.chart.Chart]
            The created and styled PowerPoint chart object, or None if creation failed.
        """
        try:
            # 1. Create the chart shape and load data
            self._create_chart_object()
            if not self.chart:  # Check if chart creation failed inside the method
                logger.error("Chart object creation failed. Aborting styling.")
                return None

            # 2. Apply style settings - order can matter (e.g., axes/legend affect plot area)
            logger.debug("Applying chart styling...")
            self._apply_chart_title()
            self._apply_legend_settings()  # Legend size/position affects plot area
            self._apply_axis_settings()  # Axis visibility/labels affect plot area & contains plot border setting
            self._apply_series_settings()  # Apply colors/styles to series data points/bars/lines
            self._apply_data_labels()  # Apply labels after series exist and are colored
            self._apply_bar_column_settings()  # Apply gap/overlap specific to bar/column types

            logger.info("Chart styling applied successfully.")
            return self.chart

        except Exception as e:
            # Catch errors during styling phase as well
            logger.error(f"Error during chart styling: {e}", exc_info=True)
            # Optionally remove partially created chart shape? Requires storing chart_shape ref.
            # shapes = self.slide.shapes
            # if self.chart and self.chart.part in [sp.part for sp in shapes]: ... remove logic ...
            return None  # Indicate failure


# --- Main Public Function ---


def add_chart(
    slide: Slide, dataframe: pd.DataFrame, style_config: Optional[Dict[str, Any]] = None
) -> Optional[Chart]:
    """
    High-level function to add a pandas DataFrame as a styled chart to a PowerPoint slide.

    Parameters:
    -----------
    slide : pptx.slide.Slide
        The PowerPoint slide object.
    dataframe : pd.DataFrame
        The data to display. Assumes first column is categories, rest are series.
    style_config : dict, optional
        A dictionary with custom style settings. These will be merged with
        the default style settings.

    Returns:
    --------
    Optional[pptx.chart.chart.Chart]
        The created and styled PowerPoint chart object, or None on failure.
    """
    try:
        # 1. Create the final style dictionary by merging custom with defaults
        final_style = create_chart_style(custom_style=style_config)

        # 2. Instantiate the formatter
        formatter = PowerPointChartFormatter(slide, dataframe, final_style)

        # 3. Run the creation and styling process
        chart = formatter.add_to_slide()

        return chart

    except (ValueError, TypeError) as init_err:
        # Catch known initialization errors from PowerPointChartFormatter
        logger.error(f"Invalid input for chart creation: {init_err}", exc_info=True)
        return None
    except Exception as e:
        # Catch unexpected errors during the overall process
        logger.error(
            f"An unexpected error occurred in add_chart function: {e}", exc_info=True
        )
        return None


# --- Example Usage (Optional - keep commented out or use for testing) ---
# if __name__ == "__main__":
#     from pptx import Presentation
#     print(f"Using pandas version: {pd.__version__}")
#     # Add sample data and presentation creation code here from original example...
#     # ... (Create df, prs, slide, custom_style)

#     # --- Define a custom style ---
#     custom_style = {
#         "chart_type": "stacked_bar",  # Horizontal stacked bar
#         #"position": {"left": 0.5, "top": 1.0}, # Position uses defaults if not specified
#         "dimensions": {"width": 9.0, "height": 5.5}, # Override dimensions
#         "title": {"text": "Sample Stacked Bar Chart", "font": {"size": 18, "color": "FF0000"}}, # Red title
#         "legend": {"position": "top", "include_in_layout": True},
#         "data_labels": {
#             "enabled": True,
#             "font": {"size": 8, "bold": True, "color": (255, 255, 255)}, # White font
#             "position": "center",
#             "number_format": "0",  # Show integers
#         },
#         "category_axis": {
#             "tick_labels": {"font": {"size": 9}},
#             "line": {"visible": False} # Hide category axis line
#             },
#         "value_axis": {
#             "visible": True,
#             "major_gridlines": False,
#             "max_scale": 60,  # Set max value axis limit
#             "number_format": "#,##0",  # Format axis labels with comma
#             "line": {"visible": False} # Hide value axis line
#         },
#         "plot_area": {"border": {"visible": False}}, # Explicitly hide plot area border
#         "colors": {  # Custom colors by series name (DataFrame column name)
#             "Series 1": "4F81BD",  # Blue
#             "Series 2": "C0504D",  # Red
#             "Series 3": "9BBB59",  # Green
#         },
#         "gap_width": 100, # Narrower gap between bars
#     }

#     # Create a sample DataFrame
#     data = {
#        "Category": ["Alpha", "Bravo", "Charlie", "Delta"],
#        "Series 1": [10, 20, 15, 25],
#        "Series 2": [5, 15, 10, 20],
#        "Series 3": [8, 12, 18, 6],
#     }
#     df = pd.DataFrame(data)

#     # Create a presentation
#     prs = Presentation()
#     slide_layout = prs.slide_layouts[5]  # Choose a blank layout
#     slide = prs.slides.add_slide(slide_layout)


#     # --- Add the chart using the refactored function ---
#     print("\nAdding chart 1 (Styled Stacked Bar)...")
#     chart_object = add_chart(slide, df, style_config=custom_style)

#     if chart_object:
#         print(f"Chart 1 added successfully. Type: {chart_object.chart_type.name}")
#     else:
#         print("Failed to add chart 1.")

#     # --- Add a second chart (column) with fewer custom styles ---
#     slide2 = prs.slides.add_slide(slide_layout)
#     style2 = {
#         "chart_type": "stacked_column",  # Vertical
#         "title": {"text": "Stacked Column (Fewer Customizations)"},
#         "data_labels": {"enabled": False},  # Turn off labels
#         "legend": {"enabled": False},  # Turn off legend
#         "value_axis": {"major_gridlines": True}, # Ensure gridlines are on
#         # Colors will use DEFAULT_COLORS
#     }
#     print("\nAdding chart 2 (Stacked Column)...")
#     chart_object2 = add_chart(slide2, df, style_config=style2)
#     if chart_object2:
#         print(f"Chart 2 added successfully. Type: {chart_object2.chart_type.name}")
#     else:
#         print("Failed to add chart 2.")

#     # Save the presentation
#     try:
#         output_filename = "refactored_chart_test.pptx"
#         prs.save(output_filename)
#         print(f"\nPresentation saved as {output_filename}")
#     except PermissionError:
#         print(f"\nError: Permission denied saving {output_filename}. Close the file if it's open.")
#     except Exception as e:
#          print(f"\nAn error occurred while saving: {e}")
