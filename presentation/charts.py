# -*- coding: utf-8 -*-
"""
Utilities for adding pandas DataFrame charts to PowerPoint slides.

Provides functions to create and style charts based on pandas DataFrames,
using a configurable dictionary for styling options. Raises standard errors
for configuration issues or unsupported operations.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

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

# --- Constants ---

# Mapping for data label positions (user string to pptx enum)
LABEL_POSITION_MAP: Dict[str, XL_LABEL_POSITION] = {
    "inside_end": XL_LABEL_POSITION.INSIDE_END,
    "outside_end": XL_LABEL_POSITION.OUTSIDE_END,
    "center": XL_LABEL_POSITION.CENTER,
    "inside_base": XL_LABEL_POSITION.INSIDE_BASE,
    "best_fit": XL_LABEL_POSITION.BEST_FIT,
    # Add other positions if needed
}

# Mapping for legend positions (user string to pptx enum)
LEGEND_POSITION_MAP: Dict[str, XL_LEGEND_POSITION] = {
    "top": XL_LEGEND_POSITION.TOP,
    "bottom": XL_LEGEND_POSITION.BOTTOM,
    "left": XL_LEGEND_POSITION.LEFT,
    "right": XL_LEGEND_POSITION.RIGHT,
    "top_right": XL_LEGEND_POSITION.CORNER,  # Use CORNER for top-right
    "corner": XL_LEGEND_POSITION.CORNER,
}

# Mapping for tick mark types (user string to pptx enum)
TICK_MARK_MAP: Dict[str, XL_TICK_MARK] = {
    "none": XL_TICK_MARK.NONE,
    "inside": XL_TICK_MARK.INSIDE,
    "outside": XL_TICK_MARK.OUTSIDE,
    "cross": XL_TICK_MARK.CROSS,
}

# Default colors for series (used if not specified in style['colors'])
# Consider defining based on presentation template theme colors for consistency.
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

# Supported chart type strings mapped to pptx enums
CHART_TYPE_MAP: Dict[str, XL_CHART_TYPE] = {
    "stacked_bar": XL_CHART_TYPE.BAR_STACKED,  # Horizontal
    "stacked_column": XL_CHART_TYPE.COLUMN_STACKED,  # Vertical
    "clustered_bar": XL_CHART_TYPE.BAR_CLUSTERED,
    "clustered_column": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "line": XL_CHART_TYPE.LINE,
    "line_markers": XL_CHART_TYPE.LINE_MARKERS,
    # Add more mappings as needed (pie, area, xy_scatter etc.)
}


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
        # If custom_dict is not a dict, return a deep copy of it,
        # replacing the default entirely at this level.
        return copy.deepcopy(custom_dict)

    result = copy.deepcopy(default_dict)

    for key, value in custom_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            # Ensure deep copy of custom values, overwriting or adding the key
            result[key] = copy.deepcopy(value)
    return result


def get_default_chart_style() -> Dict[str, Any]:
    """Returns a simplified default style configuration dictionary for charts."""
    # Defined as a function to ensure a fresh copy each time
    return {
        "chart_type": "stacked_bar",
        "position": {"left": 1.0, "top": 1.5},  # Inches
        "dimensions": {"width": 8.0, "height": 5.0},  # Inches
        "title": {
            "text": None,  # No title by default
            "visible": True,  # Allocate space even if text is None
            "font": {"size": 14, "bold": True},  # Simplified font options
        },
        "data_labels": {
            "enabled": True,
            "position": "center",  # Default position
            "font": {
                "size": 9,
                "bold": False,
                "color": (255, 255, 255),
            },  # Default white
            # "number_format": "0", # Removed, default is 'General'
        },
        "legend": {
            "enabled": True,
            "position": "bottom",
            "font": {"size": 10},
        },
        "category_axis": {
            "visible": True,
            "tick_labels": {"visible": True, "font": {"size": 10}},
            # Removed: major/minor gridlines, tick_marks, line visibility defaults
        },
        "value_axis": {
            "visible": True,
            "major_gridlines": True,  # Keep major gridlines on by default
            "tick_labels": {"visible": True, "font": {"size": 10}},
            # Removed: minor gridlines, tick_marks, number_format, min/max_scale, line visibility defaults
        },
        # Removed plot_area defaults (border, fill) - assume invisible/none
        "gap_width": 150,  # Default gap width for bar/column
        "overlap": 100,  # Default overlap (100 for stacked)
        "colors": {},  # Map series names to hex colors (empty means use DEFAULT_COLORS)
    }


def create_chart_style(custom_style: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Creates a complete chart style by merging custom settings with defaults.

    Raises:
    -------
    TypeError
        If custom_style is provided but is not a dictionary.
    """
    base_style = get_default_chart_style()
    if custom_style is None:
        return base_style
    if not isinstance(custom_style, dict):
        raise TypeError("custom_style must be a dictionary or None.")
    return deep_merge_dicts(base_style, custom_style)


def _apply_font_settings(font_obj: Font, font_config: Dict[str, Any]) -> None:
    """
    Applies font settings from a config dict to a Font object.

    Raises:
    -------
    ValueError
        If size is not numeric, color tuple is invalid, or hex string is invalid.
    TypeError
        If size is of an incompatible type.
    AttributeError
        If font_obj does not have expected attributes like .color.rgb.
    """
    # Apply settings only if they exist in the config
    if "name" in font_config:
        font_obj.name = font_config["name"]
    if "size" in font_config:
        font_obj.size = Pt(int(font_config["size"]))  # Raises ValueError/TypeError
    if "bold" in font_config:
        font_obj.bold = bool(font_config["bold"])
    if "italic" in font_config:
        font_obj.italic = bool(font_config["italic"])

    color_val = font_config.get("color")
    if color_val is not None:
        if isinstance(color_val, tuple) and len(color_val) == 3:
            # Raises ValueError/TypeError if elements aren't int 0-255
            font_obj.color.rgb = RGBColor(
                int(color_val[0]), int(color_val[1]), int(color_val[2])
            )
        elif isinstance(color_val, str):  # Allow hex string
            # Raises ValueError if hex string is invalid
            font_obj.color.rgb = RGBColor.from_string(color_val)
        else:
            raise ValueError(
                f"Invalid color format: {color_val}. Use 3-tuple RGB or hex string."
            )


def _get_chart_type_enum(type_str: str) -> XL_CHART_TYPE:
    """
    Maps a string chart type to its python-pptx enum.

    Raises:
    -------
    ValueError
        If the type_str is not a supported chart type key in CHART_TYPE_MAP.
    """
    type_str_lower = type_str.lower()
    if type_str_lower not in CHART_TYPE_MAP:
        raise ValueError(
            f"Unsupported chart type: '{type_str}'. Supported types are: "
            f"{', '.join(CHART_TYPE_MAP.keys())}"
        )
    return CHART_TYPE_MAP[type_str_lower]


# --- PowerPoint Chart Formatter Class ---


class PowerPointChartFormatter:
    """
    Creates and styles a PowerPoint chart from a pandas DataFrame based on a style config.
    Handles chart creation and applies formatting options, raising errors on failure
    or configuration issues.
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
            and subsequent columns contain series data. NaN values will be
            passed as None to the chart data, potentially creating gaps (e.g., in line charts).
            Users should impute data (e.g., with 0) before passing if gaps are not desired.
        style : dict
            A complete style configuration dictionary (result of create_chart_style()).

        Raises:
        -------
        ValueError
            If the input DataFrame is empty or has fewer than two columns.
        TypeError
            If 'dataframe' is not a pandas DataFrame or 'style' is not a dictionary.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input 'dataframe' must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("Input 'dataframe' cannot be empty.")
        if dataframe.shape[1] < 2:
            raise ValueError(
                "DataFrame must have at least two columns (categories + 1 series)."
            )
        if not isinstance(style, dict):
            # Should be caught by create_chart_style, but defensive check.
            raise TypeError("Input 'style' must be a dictionary.")

        self.slide = slide
        # Work with a copy to avoid modifying the original DataFrame.
        # Handle potential NaN values according to user preference (None).
        self.dataframe = dataframe.copy()
        self.style = style  # Assumes style is already merged with defaults
        self.chart: Optional[Chart] = None  # Will hold the pptx chart object
        self._chart_shape: Optional[GraphicFrame] = (
            None  # Will hold the chart container
        )

    def _prepare_chart_data(self) -> CategoryChartData:
        """Prepares the CategoryChartData object from the DataFrame."""
        chart_data = CategoryChartData()

        # Assume first column is category labels
        categories_col = self.dataframe.columns[0]
        # Ensure categories are strings for pptx
        chart_data.categories = self.dataframe[categories_col].astype(str).tolist()

        # Subsequent columns are series
        data_cols = self.dataframe.columns[1:]
        for col in data_cols:
            # Convert to numeric, coercing errors. Fill resulting NaN with None.
            # Users should handle imputation (e.g., to 0) *before* calling
            # add_chart if None values (gaps) are not desired.
            series_data = pd.to_numeric(self.dataframe[col], errors="coerce")
            # Convert pandas NA (pd.NA or np.nan) to None for pptx
            series_data_list: List[Union[float, int, None]] = [
                val if pd.notna(val) else None for val in series_data
            ]
            chart_data.add_series(
                str(col), series_data_list
            )  # Series name must be string

        return chart_data

    def _create_chart_object(self) -> None:
        """Creates the chart shape on the slide and populates with data."""
        pos_cfg = self.style.get("position", {})
        dim_cfg = self.style.get("dimensions", {})
        # Default to stacked_bar if not provided (though default style has it)
        chart_type_str = self.style.get("chart_type", "stacked_bar")

        # Raises ValueError for unsupported type string
        xl_chart_type = _get_chart_type_enum(chart_type_str)

        chart_data = self._prepare_chart_data()

        # Use .get with defaults for safety, raise errors on conversion if values invalid
        try:
            left = Inches(float(pos_cfg.get("left", 1.0)))
            top = Inches(float(pos_cfg.get("top", 1.5)))
            width = Inches(float(dim_cfg.get("width", 8.0)))
            height = Inches(float(dim_cfg.get("height", 5.0)))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid position or dimension value: {e}") from e

        # Add chart shape to slide - this can raise various pptx internal errors
        self._chart_shape = self.slide.shapes.add_chart(
            xl_chart_type, left, top, width, height, chart_data
        )
        self.chart = self._chart_shape.chart  # Assign the chart object

    def _apply_chart_title(self) -> None:
        """Applies title settings based on the style configuration."""
        if not self.chart:
            return  # Should not happen if called correctly

        title_config = self.style.get("title", {})
        title_text = title_config.get("text")
        is_visible = title_config.get("visible", True)  # Default to visible

        self.chart.has_title = bool(is_visible)

        if self.chart.has_title and title_text is not None:
            # Accessing chart_title assumes it exists when has_title is True
            title_obj = self.chart.chart_title
            title_obj.has_text_frame = True
            tf = title_obj.text_frame
            tf.text = str(title_text)  # Ensure text is string
            tf.auto_size = MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT

            font_config = title_config.get("font")
            if font_config:
                # Assuming text is in the first paragraph, first run
                run = (
                    tf.paragraphs[0].runs[0]
                    if tf.paragraphs[0].runs
                    else tf.paragraphs[0].add_run()
                )
                # _apply_font_settings will raise errors for invalid configs
                _apply_font_settings(run.font, font_config)

    def _apply_category_axis_settings(self) -> None:
        """Applies settings for the category axis."""
        if not self.chart:
            return

        config = self.style.get("category_axis", {})
        # Access axis directly, let AttributeError raise if it doesn't exist (unexpected)
        axis = self.chart.category_axis

        if "visible" in config:
            axis.visible = bool(config["visible"])

        if axis.visible:  # Only apply sub-settings if axis is visible
            if "major_gridlines" in config:
                axis.has_major_gridlines = bool(config["major_gridlines"])
            if "minor_gridlines" in config:
                axis.has_minor_gridlines = bool(config["minor_gridlines"])

            if "tick_marks" in config:
                tick_mark_str = str(config["tick_marks"]).lower()
                # Use .get for safety, default to NONE if key invalid
                axis.major_tick_mark = TICK_MARK_MAP.get(
                    tick_mark_str, XL_TICK_MARK.NONE
                )
                # axis.minor_tick_mark = ... # Typically NONE for category

            # Tick Labels
            tick_labels_cfg = config.get("tick_labels", {})
            if "visible" in tick_labels_cfg:
                axis.has_tick_labels = bool(tick_labels_cfg["visible"])

            if axis.has_tick_labels:
                font_config = tick_labels_cfg.get("font")
                if font_config:
                    # Access tick_labels directly, assuming it exists if has_tick_labels=True
                    _apply_font_settings(axis.tick_labels.font, font_config)

            # Axis Line Visibility/Formatting
            line_cfg = config.get("line", {})
            if "visible" in line_cfg:
                # Access format.line, let AttributeError raise if structure is unexpected
                if not bool(line_cfg["visible"]):
                    axis.format.line.fill.background()  # Make invisible
                else:
                    # TODO: Add line color/style customization if needed when visible=True
                    pass  # Line is visible by default if not explicitly hidden

    def _apply_value_axis_settings(self) -> None:
        """Applies settings for the value axis."""
        if not self.chart:
            return

        config = self.style.get("value_axis", {})
        # Access axis directly, let AttributeError raise if it doesn't exist
        axis = self.chart.value_axis

        if "visible" in config:
            axis.visible = bool(config["visible"])

        if axis.visible:  # Only apply sub-settings if axis is visible
            if "major_gridlines" in config:
                axis.has_major_gridlines = bool(config["major_gridlines"])
            if "minor_gridlines" in config:
                axis.has_minor_gridlines = bool(config["minor_gridlines"])

            if "tick_marks" in config:
                tick_mark_str = str(config["tick_marks"]).lower()
                axis.major_tick_mark = TICK_MARK_MAP.get(
                    tick_mark_str, XL_TICK_MARK.NONE
                )
                # axis.minor_tick_mark = ... # Could be configured too

            # Axis Scale Limits (raise ValueError/TypeError on float conversion)
            if "max_scale" in config and config["max_scale"] is not None:
                axis.maximum_scale = float(config["max_scale"])
            if "min_scale" in config and config["min_scale"] is not None:
                axis.minimum_scale = float(config["min_scale"])

            # Tick Labels
            tick_labels_cfg = config.get("tick_labels", {})
            if "visible" in tick_labels_cfg:
                axis.has_tick_labels = bool(tick_labels_cfg["visible"])

            if axis.has_tick_labels:
                # Access tick_labels directly
                if "number_format" in config:
                    axis.tick_labels.number_format = str(config["number_format"])
                    # Link format to source data only if using Excel's default
                    axis.tick_labels.number_format_is_linked = (
                        axis.tick_labels.number_format == "General"
                    )
                else:
                    # Explicitly set default if not in config
                    axis.tick_labels.number_format = "General"
                    axis.tick_labels.number_format_is_linked = True

                font_config = tick_labels_cfg.get("font")
                if font_config:
                    _apply_font_settings(axis.tick_labels.font, font_config)

            # Axis Line Visibility/Formatting
            line_cfg = config.get("line", {})
            if "visible" in line_cfg:
                # Access format.line directly
                if not bool(line_cfg["visible"]):
                    axis.format.line.fill.background()  # Make invisible
                else:
                    # TODO: Add line color/style customization if needed when visible=True
                    pass  # Line is visible by default if not explicitly hidden

    def _apply_plot_area_settings(self) -> None:
        """Applies settings for the plot area (border, fill), if available."""
        if not self.chart:
            return

        # Check if the chart object has the plot_area attribute before proceeding
        if not hasattr(self.chart, "plot_area"):
            # Silently skip if plot_area attribute doesn't exist for this chart type
            # Alternatively, could issue a warning:
            # import warnings
            # warnings.warn(f"Chart type {self.chart.chart_type.name} might not support direct plot_area formatting.")
            return

        config = self.style.get("plot_area", {})
        # Access plot_area only if we know it exists
        plot_area = self.chart.plot_area

        # --- Plot Area Border ---
        border_cfg = config.get("border", {})
        if "visible" in border_cfg:
            # Access format.line directly; assume it exists on plot_area if plot_area exists
            if not bool(border_cfg["visible"]):
                # Check if line formatting exists before trying to modify it
                if hasattr(plot_area.format, "line") and plot_area.format.line:
                    plot_area.format.line.fill.background()
            else:
                # TODO: Add plot area border color/style customization if visible
                if hasattr(plot_area.format, "line") and plot_area.format.line:
                    # Example structure, assumes line object exists:
                    # line = plot_area.format.line
                    # line.color.rgb = RGBColor.from_string(border_cfg.get("color", "000000")) # Raises ValueError if invalid
                    # line.width = Pt(border_cfg.get("width", 1)) # Raises ValueError/TypeError if invalid
                    pass  # Border visible, potentially styled

        # --- Plot Area Fill ---
        fill_cfg = config.get("fill", {})
        if "type" in fill_cfg:
            # Access format.fill directly; assume it exists on plot_area if plot_area exists
            if hasattr(plot_area.format, "fill") and plot_area.format.fill:
                fill = plot_area.format.fill
                fill_type = str(fill_cfg["type"]).lower()
                if fill_type == "none":
                    fill.background()
                elif fill_type == "solid":
                    fill.solid()
                    if "color" in fill_cfg:
                        # Raises ValueError for invalid hex
                        fill.fore_color.rgb = RGBColor.from_string(
                            str(fill_cfg["color"])
                        )
                # TODO: Add support for other fill types like 'gradient' if needed
                else:
                    # Raise error for unsupported fill type configuration
                    raise ValueError(f"Unsupported plot area fill type: '{fill_type}'")

        # --- Plot Area Fill ---
        fill_cfg = config.get("fill", {})
        if "type" in fill_cfg:
            # Access format.fill directly
            fill = plot_area.format.fill
            fill_type = str(fill_cfg["type"]).lower()
            if fill_type == "none":
                fill.background()
            elif fill_type == "solid":
                fill.solid()
                if "color" in fill_cfg:
                    # Raises ValueError for invalid hex
                    fill.fore_color.rgb = RGBColor.from_string(str(fill_cfg["color"]))
            # TODO: Add support for other fill types like 'gradient' if needed
            else:
                # Raise error for unsupported fill type configuration
                raise ValueError(f"Unsupported plot area fill type: '{fill_type}'")

    def _apply_legend_settings(self) -> None:
        """Applies legend settings (position, visibility, font)."""
        if not self.chart:
            return

        config = self.style.get("legend", {})

        if "enabled" in config:
            self.chart.has_legend = bool(config["enabled"])

        if self.chart.has_legend:
            # Access legend directly, assume it exists if has_legend is True
            legend = self.chart.legend

            if "position" in config:
                position_str = str(config["position"]).lower()
                # Use .get for safety, default to BOTTOM if key invalid
                legend.position = LEGEND_POSITION_MAP.get(
                    position_str, XL_LEGEND_POSITION.BOTTOM
                )

            if "include_in_layout" in config:
                # Let pptx manage layout (True) or allow overlap (False)
                legend.include_in_layout = bool(config["include_in_layout"])

            font_config = config.get("font")
            if font_config:
                # Access legend.font directly
                _apply_font_settings(legend.font, font_config)

    def _apply_data_labels(self) -> None:
        """Applies data label settings (visibility, position, font, number format)."""
        if not self.chart or not self.chart.plots:
            return

        config = self.style.get("data_labels", {})
        # Apply to the first plot (common for bar/column/line)
        # Access plot directly, let IndexError raise if no plots (unexpected)
        plot = self.chart.plots[0]

        # Check if plot object supports data labels - raise error if trying to enable on unsupported plot
        if "enabled" in config:
            enable_labels = bool(config["enabled"])
            # Try setting has_data_labels. If it fails, the plot type might not support it.
            # Let AttributeError raise in that case.
            plot.has_data_labels = enable_labels

        if plot.has_data_labels:
            # Access data_labels directly, assume it exists if has_data_labels is True
            data_labels = plot.data_labels

            if "position" in config:
                position_str = str(config["position"]).lower()
                # Use .get, default to CENTER
                data_labels.position = LABEL_POSITION_MAP.get(
                    position_str, XL_LABEL_POSITION.CENTER
                )

            if "number_format" in config:
                data_labels.number_format = str(config["number_format"])
                data_labels.number_format_is_linked = (
                    data_labels.number_format == "General"
                )
            else:
                # Explicitly set default if not in config
                data_labels.number_format = "General"
                data_labels.number_format_is_linked = True

            font_config = config.get("font")
            if font_config:
                # Access data_labels.font directly
                _apply_font_settings(data_labels.font, font_config)

    def _apply_series_settings(self) -> None:
        """Applies settings per series, primarily colors."""
        if not self.chart or not self.chart.series:
            return

        series_colors_config = self.style.get(
            "colors", {}
        )  # Map series name -> hex color

        for i, series in enumerate(self.chart.series):
            # Determine color: Use specific color if defined, else default palette
            color_hex = series_colors_config.get(
                series.name
            )  # Use series.name (matches DataFrame col)
            if not color_hex:
                color_index = i % len(DEFAULT_COLORS)
                color_hex = DEFAULT_COLORS[color_index]

            # Apply fill color (works for bar, column, area, pie points)
            # Access format.fill directly. Let AttributeError/ValueError propagate.
            try:
                fill = series.format.fill
                fill.solid()
                fill.fore_color.rgb = RGBColor.from_string(
                    str(color_hex)
                )  # Ensure hex is str
            except ValueError as e:
                raise ValueError(
                    f"Invalid hex color string '{color_hex}' for series '{series.name}'."
                ) from e
            except AttributeError:
                # If .format.fill is missing, this series type might not support solid fill.
                # Raise error indicating potential incompatibility or unexpected object structure.
                raise AttributeError(
                    f"Cannot apply fill color to series '{series.name}'. Series type may not support direct fill formatting."
                )

            # TODO: Apply line color/style (for line charts) based on config
            # if self.chart.chart_type in [XL_CHART_TYPE.LINE, XL_CHART_TYPE.LINE_MARKERS]:
            #     try:
            #         line = series.format.line
            #         line.color.rgb = RGBColor.from_string(str(color_hex)) # Or use separate line color config
            #         # line.width = Pt(style.get("line_width", 2.0))
            #     except ValueError as e: ...
            #     except AttributeError: ...

    def _apply_bar_column_settings(self) -> None:
        """Applies settings specific to bar/column charts (gap width, overlap)."""
        if not self.chart or not self.chart.plots:
            return

        # Check if the chart type conceptually supports these settings
        supported_types = (
            XL_CHART_TYPE.BAR_STACKED,
            XL_CHART_TYPE.COLUMN_STACKED,
            XL_CHART_TYPE.BAR_CLUSTERED,
            XL_CHART_TYPE.COLUMN_CLUSTERED,
            # Add BAR_STACKED_100 etc. if needed
        )

        if self.chart.chart_type in supported_types:
            # Access plot directly, assume first plot holds these settings
            plot = self.chart.plots[0]

            # Apply gap width if present in style config
            if "gap_width" in self.style:
                try:
                    # Access plot.gap_width directly, let AttributeError raise if missing
                    # Let ValueError/TypeError raise on int conversion
                    plot.gap_width = int(self.style["gap_width"])
                except (ValueError, TypeError) as e:
                    raise type(e)(
                        f"Invalid gap_width value: {self.style['gap_width']}"
                    ) from e
                except AttributeError:
                    raise AttributeError(
                        f"Chart type {self.chart.chart_type.name} plot does not support 'gap_width'."
                    )

            # Apply overlap if present in style config
            if "overlap" in self.style:
                try:
                    # Access plot.overlap directly
                    plot.overlap = int(self.style["overlap"])
                except (ValueError, TypeError) as e:
                    raise type(e)(
                        f"Invalid overlap value: {self.style['overlap']}"
                    ) from e
                except AttributeError:
                    raise AttributeError(
                        f"Chart type {self.chart.chart_type.name} plot does not support 'overlap'."
                    )

        # If chart type doesn't support these, attempting to set them from the
        # config might be considered an error, but we silently ignore if not in supported_types for now.
        # Alternatively, could raise ValueError if gap_width/overlap provided for non-bar/col chart.

    def add_to_slide(self) -> Chart:
        """
        Executes the full process: creates the chart object and applies all styles.

        Returns:
        --------
        pptx.chart.chart.Chart
            The created and styled PowerPoint chart object.

        Raises:
        -------
        ValueError, TypeError, AttributeError, IndexError
            Propagates exceptions occurring during chart creation or styling
            due to invalid configuration, data issues, or mismatches between
            style options and chart capabilities.
        """
        # 1. Create the chart shape and load data
        # This step raises errors if basic creation fails (bad type, data, pos/dims)
        self._create_chart_object()

        # Ensure chart object exists before proceeding (should always unless _create failed)
        if not self.chart:
            # This condition should ideally not be reached if _create_chart_object raises on failure
            raise RuntimeError("Chart object was not created successfully.")

        # 2. Apply style settings - order can matter
        # These methods raise errors if config is invalid or attributes don't exist when expected
        self._apply_chart_title()
        self._apply_legend_settings()  # Legend affects plot area
        self._apply_category_axis_settings()  # Axes affect plot area
        self._apply_value_axis_settings()
        self._apply_plot_area_settings()  # Plot area border/fill
        self._apply_series_settings()  # Colors/styles for data points/bars/lines
        self._apply_data_labels()  # Apply labels after series exist
        self._apply_bar_column_settings()  # Gap/overlap specific to bar/column types

        return self.chart


# --- Main Public Function ---


def add_chart(
    slide: Slide, dataframe: pd.DataFrame, style_config: Optional[Dict[str, Any]] = None
) -> Chart:
    """
    High-level function to add a pandas DataFrame as a styled chart to a PowerPoint slide.

    Parameters:
    -----------
    slide : pptx.slide.Slide
        The PowerPoint slide object.
    dataframe : pd.DataFrame
        The data to display. Assumes first column is categories, rest are series.
        NaN values will be passed as None, potentially creating gaps in charts.
        Impute data (e.g., with 0) beforehand if gaps are not desired.
    style_config : dict, optional
        A dictionary with custom style settings. These will be merged with
        the default style settings.

    Returns:
    --------
    pptx.chart.chart.Chart
        The created and styled PowerPoint chart object.

    Raises:
    -------
    ValueError, TypeError, AttributeError, IndexError
        Propagates exceptions from chart creation or styling, typically due to
        invalid configuration (e.g., unknown chart type, bad color format),
        data problems, or attempting to apply styles incompatible with the
        chosen chart type.
    """
    # 1. Create the final style dictionary. Raises TypeError if style_config is not dict.
    final_style = create_chart_style(custom_style=style_config)

    # 2. Instantiate the formatter. Raises TypeError/ValueError for bad inputs.
    formatter = PowerPointChartFormatter(slide, dataframe, final_style)

    # 3. Run the creation and styling process. Raises various errors on failure.
    chart = formatter.add_to_slide()

    return chart


# --- Example Usage (Optional - keep commented out or use for testing) ---
# if __name__ == "__main__":
#     from pptx import Presentation
#     print(f"Using pandas version: {pd.__version__}")

#     # --- Define a custom style (using the simplified structure + extras) ---
#     custom_style = {
#         "chart_type": "stacked_bar",
#         "dimensions": {"width": 9.0, "height": 5.5},
#         "title": {"text": "Sample Stacked Bar Chart", "font": {"size": 18, "color": "FF0000"}}, # Red title
#         "legend": {"position": "top"}, # Override legend position
#         "data_labels": {
#             "enabled": True, # Keep labels enabled
#             "font": {"size": 8, "bold": True, "color": "FFFFFF"}, # White, smaller, bold
#             "position": "inside_base", # Change position
#             "number_format": "0", # Add number format back
#         },
#         "category_axis": {
#             "tick_labels": {"font": {"size": 9}},
#             "line": {"visible": False} # Add axis line visibility setting
#             },
#         "value_axis": {
#             "visible": True,
#             "major_gridlines": False, # Turn off gridlines
#             "max_scale": 60,         # Add max scale
#             "number_format": "#,##0", # Add number format
#             "line": {"visible": False}
#         },
#         "plot_area": {"border": {"visible": True, "color": "0000FF"}}, # Add blue plot area border
#         "colors": { # Custom colors by series name (DataFrame column name)
#             "Series 1": "4F81BD", # Blue
#             "Series 2": "C0504D", # Red
#             "Series 3": "9BBB59", # Green
#         },
#         "gap_width": 100, # Narrower gap
#     }

#     # Create a sample DataFrame with NaN
#     data = {
#         "Category": ["Alpha", "Bravo", "Charlie", "Delta", "Echo"],
#         "Series 1": [10, 20, 15, 25, 18],
#         "Series 2": [5, 15, None, 20, 10], # Contains None/NaN
#         "Series 3": [8, 12, 18, 6, None], # Contains None/NaN
#     }
#     df = pd.DataFrame(data)
#     # If you wanted zeros instead of gaps/None:
#     # df_filled = df.fillna(0)

#     # Create a presentation
#     prs = Presentation()
#     slide_layout = prs.slide_layouts[5] # Choose a blank layout
#     slide = prs.slides.add_slide(slide_layout)

#     # --- Add the chart using the refactored function ---
#     print("\nAdding chart 1 (Styled Stacked Bar with NaN)...")
#     try:
#         # Use df, which contains None values
#         chart_object = add_chart(slide, df, style_config=custom_style)
#         print(f"Chart 1 added successfully. Type: {chart_object.chart_type.name}")
#     except Exception as e:
#         print(f"Failed to add chart 1: {type(e).__name__}: {e}")
#         # import traceback
#         # traceback.print_exc() # Uncomment for full traceback

#     # --- Add a second chart (line) with fewer custom styles and NaN ---
#     slide2 = prs.slides.add_slide(slide_layout)
#     style2 = {
#         "chart_type": "line_markers", # Line chart
#         "title": {"text": "Line Chart with Gaps (NaN as None)"},
#         "data_labels": {"enabled": False},
#         "legend": {"enabled": True, "position": "right"},
#         "value_axis": {"major_gridlines": True},
#         # Colors will use DEFAULT_COLORS
#     }
#     print("\nAdding chart 2 (Line with Markers and NaN)...")
#     try:
#         # Use df, which contains None values - should create gaps in the line
#         chart_object2 = add_chart(slide2, df, style_config=style2)
#         print(f"Chart 2 added successfully. Type: {chart_object2.chart_type.name}")
#     except Exception as e:
#         print(f"Failed to add chart 2: {type(e).__name__}: {e}")
#         # import traceback
#         # traceback.print_exc()

#     # Save the presentation
#     try:
#         output_filename = "refactored_chart_test_v2.pptx"
#         prs.save(output_filename)
#         print(f"\nPresentation saved as {output_filename}")
#     except PermissionError:
#         print(f"\nError: Permission denied saving {output_filename}. Close the file if it's open.")
#     except Exception as e:
#          print(f"\nAn error occurred while saving: {e}")
