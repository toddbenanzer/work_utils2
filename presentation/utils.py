"""
Common utilities for the presentation module.
Contains shared functions, constants, and mappings used across multiple modules.
"""

import copy
from typing import Any, Dict, Optional, Tuple, Union

from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_MARK
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.text.text import Font


# --- Common Constants and Mappings ---

# Text alignment mapping (user string to pptx enum)
ALIGNMENT_MAP: Dict[str, PP_ALIGN] = {
    "left": PP_ALIGN.LEFT,
    "center": PP_ALIGN.CENTER,
    "right": PP_ALIGN.RIGHT,
    "justify": PP_ALIGN.JUSTIFY,
    "distributed": PP_ALIGN.DISTRIBUTE,
    "thai_distributed": PP_ALIGN.THAI_DISTRIBUTE,
}

# Vertical alignment mapping (user string to pptx enum)
VERTICAL_ALIGNMENT_MAP: Dict[str, MSO_ANCHOR] = {
    "top": MSO_ANCHOR.TOP,
    "middle": MSO_ANCHOR.MIDDLE,
    "bottom": MSO_ANCHOR.BOTTOM,
}

# Text frame auto-sizing mapping (user string to pptx enum)
AUTOSIZE_MAP: Dict[str, MSO_AUTO_SIZE] = {
    "none": MSO_AUTO_SIZE.NONE,
    "shape_to_fit_text": MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT,
    "text_to_fit_shape": MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE,
}

# Chart-specific mappings
LABEL_POSITION_MAP: Dict[str, XL_LABEL_POSITION] = {
    "inside_end": XL_LABEL_POSITION.INSIDE_END,
    "outside_end": XL_LABEL_POSITION.OUTSIDE_END,
    "center": XL_LABEL_POSITION.CENTER,
    "inside_base": XL_LABEL_POSITION.INSIDE_BASE,
    "best_fit": XL_LABEL_POSITION.BEST_FIT,
}

LEGEND_POSITION_MAP: Dict[str, XL_LEGEND_POSITION] = {
    "top": XL_LEGEND_POSITION.TOP,
    "bottom": XL_LEGEND_POSITION.BOTTOM,
    "left": XL_LEGEND_POSITION.LEFT,
    "right": XL_LEGEND_POSITION.RIGHT,
    "top_right": XL_LEGEND_POSITION.CORNER,
    "corner": XL_LEGEND_POSITION.CORNER,
}

TICK_MARK_MAP: Dict[str, XL_TICK_MARK] = {
    "none": XL_TICK_MARK.NONE,
    "inside": XL_TICK_MARK.INSIDE,
    "outside": XL_TICK_MARK.OUTSIDE,
    "cross": XL_TICK_MARK.CROSS,
}

CHART_TYPE_MAP: Dict[str, XL_CHART_TYPE] = {
    "stacked_bar": XL_CHART_TYPE.BAR_STACKED,
    "stacked_column": XL_CHART_TYPE.COLUMN_STACKED,
    "clustered_bar": XL_CHART_TYPE.BAR_CLUSTERED,
    "clustered_column": XL_CHART_TYPE.COLUMN_CLUSTERED,
    "line": XL_CHART_TYPE.LINE,
    "line_markers": XL_CHART_TYPE.LINE_MARKERS,
}

# Default chart colors
DEFAULT_CHART_COLORS: Tuple[str, ...] = (
    "3C2F80", "2C1F10", "4C4C4C", "7030A0", "0070C0",
    "00B050", "FFC000", "FF0000", "A9A9A9", "5A5A5A",
)


# --- Common Utility Functions ---

def deep_merge_dicts(default_dict: Dict[str, Any], custom_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. Values from custom_dict take precedence.
    Creates deep copies to avoid modifying original dictionaries.
    
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
        return copy.deepcopy(custom_dict)
    if not isinstance(custom_dict, dict):
        return copy.deepcopy(custom_dict)

    result = copy.deepcopy(default_dict)

    for key, value in custom_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_font_settings(font_obj: Font, font_config: Dict[str, Any]) -> None:
    """
    Applies font settings from a config dict to a Font object.
    
    Parameters:
    -----------
    font_obj : pptx.text.text.Font
        The font object to modify.
    font_config : dict
        Dictionary with font settings (name, size, bold, italic, color).
        
    Raises:
    -------
    ValueError
        If size is not numeric, color tuple is invalid, or hex string is invalid.
    TypeError
        If size is of an incompatible type.
    """
    if "name" in font_config:
        font_obj.name = font_config["name"]
    if "size" in font_config:
        font_obj.size = Pt(int(font_config["size"]))
    if "bold" in font_config:
        font_obj.bold = bool(font_config["bold"])
    if "italic" in font_config:
        font_obj.italic = bool(font_config["italic"])

    color_val = font_config.get("color")
    if color_val is not None:
        if isinstance(color_val, tuple) and len(color_val) == 3:
            font_obj.color.rgb = RGBColor(
                int(color_val[0]), int(color_val[1]), int(color_val[2])
            )
        elif isinstance(color_val, str):
            font_obj.color.rgb = RGBColor.from_string(color_val)
        else:
            raise ValueError(
                f"Invalid color format: {color_val}. Use 3-tuple RGB or hex string."
            )


def get_color_from_value(color_val: Union[str, Tuple[int, int, int]]) -> RGBColor:
    """
    Converts a color value (hex string or RGB tuple) to an RGBColor object.
    
    Parameters:
    -----------
    color_val : Union[str, Tuple[int, int, int]]
        Either a hex string (e.g., "FF0000") or RGB tuple (e.g., (255, 0, 0))
        
    Returns:
    --------
    pptx.dml.color.RGBColor
        The converted RGBColor object
        
    Raises:
    -------
    ValueError
        If the color value format is invalid
    """
    if isinstance(color_val, tuple) and len(color_val) == 3:
        return RGBColor(int(color_val[0]), int(color_val[1]), int(color_val[2]))
    elif isinstance(color_val, str):
        return RGBColor.from_string(color_val)
    else:
        raise ValueError(f"Invalid color format: {color_val}. Use 3-tuple RGB or hex string.")


def Pt(points: Union[int, float]) -> int:
    """
    Converts points to English Metric Units (EMUs).
    
    Parameters:
    -----------
    points : Union[int, float]
        The point value to convert
        
    Returns:
    --------
    int
        The equivalent value in EMUs
    """
    return int(points * 12700)


def Inches(inches: Union[int, float]) -> int:
    """
    Converts inches to English Metric Units (EMUs).
    
    Parameters:
    -----------
    inches : Union[int, float]
        The inch value to convert
        
    Returns:
    --------
    int
        The equivalent value in EMUs
    """
    return int(inches * 914400)
