# -*- coding: utf-8 -*-
"""
Utilities for adding styled text boxes to PowerPoint slides.

Provides functions to create and style text boxes based on a list of strings,
using a configurable dictionary for styling options. Raises standard errors
for configuration issues or unsupported operations.
"""

import copy
from typing import Any, Dict, List, Optional, Union

# Assuming pandas is NOT needed for this specific module, but pptx is.
# import pandas as pd # Removed if not directly used for text boxes
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN  # Added PP_ALIGN
from pptx.shapes.autoshape import Shape  # For type hinting textbox_shape
from pptx.slide import Slide
from pptx.text.text import Font  # Added _Paragraph for type hint
from pptx.util import Inches, Pt

# --- Constants ---

# Mapping for text alignment (user string to pptx enum)
ALIGNMENT_MAP: Dict[str, PP_ALIGN] = {
    "left": PP_ALIGN.LEFT,
    "center": PP_ALIGN.CENTER,
    "right": PP_ALIGN.RIGHT,
    "justify": PP_ALIGN.JUSTIFY,
    "distributed": PP_ALIGN.DISTRIBUTE,  # If needed
    "thai_distributed": PP_ALIGN.THAI_DISTRIBUTE,  # If needed
}

# Mapping for text frame auto-sizing (user string to pptx enum)
AUTOSIZE_MAP: Dict[str, MSO_AUTO_SIZE] = {
    "none": MSO_AUTO_SIZE.NONE,
    "shape_to_fit_text": MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT,
    "text_to_fit_shape": MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE,
}

# --- Helper Functions (Assumed to be available from your chart code) ---


def deep_merge_dicts(
    default_dict: Dict[str, Any], custom_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merges two dictionaries. Values from custom_dict take precedence.
    Creates deep copies to avoid modifying original dictionaries.
    (Implementation copied from your provided chart code)
    """
    if not isinstance(default_dict, dict):
        return copy.deepcopy(custom_dict)
    if not isinstance(custom_dict, dict):
        return copy.deepcopy(custom_dict)  # Replace default if custom is not dict

    result = copy.deepcopy(default_dict)

    for key, value in custom_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_font_settings(font_obj: Font, font_config: Dict[str, Any]) -> None:
    """
    Applies font settings from a config dict to a Font object.
    (Implementation adapted from your provided chart code)

    Raises:
    -------
    ValueError
        If size is not numeric, color tuple is invalid, or hex string is invalid.
    TypeError
        If size is of an incompatible type.
    AttributeError
        If font_obj does not have expected attributes like .color.rgb.
    """
    if "name" in font_config:
        font_obj.name = font_config["name"]
    if "size" in font_config:
        try:
            font_obj.size = Pt(int(font_config["size"]))
        except (ValueError, TypeError) as e:
            raise type(e)(f"Invalid font size value: {font_config['size']}") from e
    if "bold" in font_config:
        font_obj.bold = bool(font_config["bold"])
    if "italic" in font_config:
        font_obj.italic = bool(font_config["italic"])

    color_val = font_config.get("color")
    if color_val is not None:
        try:
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
        except (ValueError, TypeError) as e:
            raise type(e)(f"Invalid color value: {color_val}. {e}") from e
        except AttributeError:
            # Handle cases where font.color or font.color.rgb might not be directly available
            # This is less common for standard text runs, but good practice.
            raise AttributeError(
                "Font object does not support RGB color setting directly."
            )


# --- Textbox Specific Functions ---


def get_default_textbox_style() -> Dict[str, Any]:
    """Returns a default style configuration dictionary for text boxes."""
    # Defined as a function to ensure a fresh copy each time
    return {
        "position": {"left": 1.0, "top": 1.0},  # Inches
        "dimensions": {
            "width": 6.0,
            "height": 1.0,
        },  # Inches (height often adjusted by auto_size)
        "font": {
            "name": None,  # Use theme default
            "size": 12,  # Pt
            "bold": False,
            "italic": False,
            "color": "000000",  # Default black as hex
        },
        "paragraph": {
            "alignment": "left",  # Default alignment
            "space_before": 0,  # Pt
            "space_after": 0,  # Pt (or maybe Pt(6) for some spacing?)
            "line_spacing": 1.0,  # Multiple of line height
        },
        "bullets": {
            "enabled": False,  # Are bullets active for all paragraphs?
            "level": 1,  # Which bullet level (1-based for config)
            # "type": None, # Future: Could specify bullet char/style here
        },
        "fill": {
            "type": "none",  # 'none' or 'solid'
            # "color": "FFFFFF" # Required if type is 'solid'
        },
        "border": {
            "visible": False,
            # "color": "000000", # Required if visible is True
            # "width": 1.0,     # Pt, required if visible is True
        },
        "auto_size": "shape_to_fit_text",  # Default auto-size behavior
        "margins": {"left": 0.1, "right": 0.1, "top": 0.05, "bottom": 0.05},  # Inches
        "vertical_anchor": "top",  # MSO_ANCHOR: top, middle, bottom
    }


def create_textbox_style(
    custom_style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Creates a complete textbox style by merging custom settings with defaults.

    Raises:
    -------
    TypeError
        If custom_style is provided but is not a dictionary.
    """
    base_style = get_default_textbox_style()
    if custom_style is None:
        return base_style
    if not isinstance(custom_style, dict):
        raise TypeError("custom_style must be a dictionary or None.")
    return deep_merge_dicts(base_style, custom_style)


# --- PowerPoint Textbox Formatter Class ---


class PowerPointTextboxFormatter:
    """
    Creates and styles a PowerPoint textbox from a list of strings based on a style config.
    """

    def __init__(self, slide: Slide, text_content: List[str], style: Dict[str, Any]):
        """
        Initializes the textbox formatter.

        Parameters:
        -----------
        slide : pptx.slide.Slide
            The PowerPoint slide to add the textbox to.
        text_content : List[str]
            A list of strings, where each string becomes a paragraph.
        style : dict
            A complete style configuration dictionary (result of create_textbox_style()).

        Raises:
        -------
        TypeError
            If 'text_content' is not a list or 'style' is not a dictionary.
        """
        if not isinstance(text_content, list):
            # Ensure input is a list, even if a single string was intended (convert it)
            # If the user *must* provide a list, raise TypeError here.
            # Let's assume flexibility and convert single strings later if needed,
            # but the primary design expects a list.
            raise TypeError("Input 'text_content' must be a list of strings.")
        if not isinstance(style, dict):
            raise TypeError("Input 'style' must be a dictionary.")

        self.slide = slide
        # Ensure all items in the list are strings
        self.text_content = [str(item) for item in text_content]
        self.style = style
        self._textbox_shape: Optional[Shape] = None
        self.text_frame = None  # Will hold the TextFrame object

    def _create_textbox_object(self) -> None:
        """Creates the textbox shape on the slide."""
        pos_cfg = self.style.get("position", {})
        dim_cfg = self.style.get("dimensions", {})

        try:
            left = Inches(float(pos_cfg.get("left", 1.0)))
            top = Inches(float(pos_cfg.get("top", 1.0)))
            width = Inches(float(dim_cfg.get("width", 6.0)))
            height = Inches(float(dim_cfg.get("height", 1.0)))  # Initial height
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid position or dimension value: {e}") from e

        try:
            self._textbox_shape = self.slide.shapes.add_textbox(
                left, top, width, height
            )
            self.text_frame = self._textbox_shape.text_frame
        except Exception as e:
            # Catch potential errors during shape addition
            raise RuntimeError(f"Failed to add textbox shape to slide: {e}") from e

    def _apply_textbox_properties(self) -> None:
        """Applies shape-level properties like fill, border, margins, auto-size."""
        if not self._textbox_shape or not self.text_frame:
            raise RuntimeError("Textbox shape or text frame not initialized.")

        # --- Auto Size ---
        auto_size_cfg = self.style.get("auto_size", "none")
        auto_size_enum = AUTOSIZE_MAP.get(str(auto_size_cfg).lower())
        if auto_size_enum is None:
            raise ValueError(
                f"Invalid auto_size value: '{auto_size_cfg}'. Use one of {list(AUTOSIZE_MAP.keys())}"
            )
        self.text_frame.auto_size = auto_size_enum

        # --- Margins ---
        margin_cfg = self.style.get("margins", {})
        try:
            if "left" in margin_cfg:
                self.text_frame.margin_left = Inches(float(margin_cfg["left"]))
            if "right" in margin_cfg:
                self.text_frame.margin_right = Inches(float(margin_cfg["right"]))
            if "top" in margin_cfg:
                self.text_frame.margin_top = Inches(float(margin_cfg["top"]))
            if "bottom" in margin_cfg:
                self.text_frame.margin_bottom = Inches(float(margin_cfg["bottom"]))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid margin value: {e}") from e

        # --- Fill ---
        fill_cfg = self.style.get("fill", {})
        fill_type = str(fill_cfg.get("type", "none")).lower()
        shape_fill = self._textbox_shape.fill
        if fill_type == "none":
            shape_fill.background()
        elif fill_type == "solid":
            fill_color = fill_cfg.get("color")
            if fill_color is None:
                raise ValueError(
                    "Solid fill specified but no 'color' provided in fill style."
                )
            shape_fill.solid()
            try:
                # Use foreground color for solid fill
                shape_fill.fore_color.rgb = RGBColor.from_string(str(fill_color))
            except ValueError as e:
                raise ValueError(
                    f"Invalid fill color value: '{fill_color}'. {e}"
                ) from e
        else:
            raise ValueError(
                f"Unsupported fill type: '{fill_type}'. Use 'none' or 'solid'."
            )

        # --- Border (Line) ---
        border_cfg = self.style.get("border", {})
        is_visible = bool(border_cfg.get("visible", False))
        shape_line = self._textbox_shape.line
        if is_visible:
            border_color = border_cfg.get("color")
            border_width = border_cfg.get("width", 1.0)  # Default width 1pt if visible
            if border_color is None:
                raise ValueError(
                    "Border visibility is true but no 'color' provided in border style."
                )

            try:
                shape_line.color.rgb = RGBColor.from_string(str(border_color))
            except ValueError as e:
                raise ValueError(
                    f"Invalid border color value: '{border_color}'. {e}"
                ) from e

            try:
                shape_line.width = Pt(float(border_width))
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid border width value: '{border_width}'. {e}"
                ) from e
        else:
            shape_line.fill.background()  # Makes line invisible

        # --- Vertical Alignment ---
        valign_cfg = self.style.get("vertical_anchor", "top").upper()
        # MSO_ANCHOR enums are TOP, MIDDLE, BOTTOM, TOP_CENTERED, MIDDLE_CENTERED, BOTTOM_CENTERED
        # Map simple terms to common enums
        valign_map = {
            "TOP": MSO_ANCHOR.TOP,
            "MIDDLE": MSO_ANCHOR.MIDDLE,
            "BOTTOM": MSO_ANCHOR.BOTTOM,
        }
        valign_enum = valign_map.get(valign_cfg)
        if valign_enum is None:
            raise ValueError(
                f"Invalid vertical_anchor value: '{valign_cfg}'. Use 'top', 'middle', or 'bottom'."
            )
        self.text_frame.vertical_anchor = valign_enum

    def _apply_text_and_paragraphs(self) -> None:
        """Adds text content and applies paragraph/font styles."""
        if not self.text_frame:
            raise RuntimeError("Text frame not initialized.")

        # Clear any default paragraph/text in the newly added textbox
        self.text_frame.clear()

        font_style = self.style.get("font", {})
        para_style = self.style.get("paragraph", {})
        bullet_style = self.style.get("bullets", {})
        bullets_enabled = bool(bullet_style.get("enabled", False))

        # --- Get Paragraph Settings ---
        align_str = str(para_style.get("alignment", "left")).lower()
        alignment = ALIGNMENT_MAP.get(align_str)
        if alignment is None:
            raise ValueError(
                f"Invalid paragraph alignment: '{align_str}'. Use one of {list(ALIGNMENT_MAP.keys())}"
            )

        try:
            space_before = Pt(int(para_style.get("space_before", 0)))
            space_after = Pt(int(para_style.get("space_after", 0)))
            line_spacing = float(
                para_style.get("line_spacing", 1.0)
            )  # Can be float for multiple
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid paragraph spacing value: {e}") from e

        # --- Get Bullet Settings (if enabled) ---
        bullet_level = 0  # Default to 0 (no indent/bullet)
        if bullets_enabled:
            try:
                # Config uses 1-based level, pptx uses 0-based
                level_1_based = int(bullet_style.get("level", 1))
                if level_1_based < 1 or level_1_based > 9:  # pptx supports levels 0-8
                    raise ValueError("Bullet level must be between 1 and 9.")
                bullet_level = level_1_based - 1
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid bullet level value: {e}") from e

        # --- Add Paragraphs and Apply Styles ---
        for i, line_text in enumerate(self.text_content):
            p = self.text_frame.add_paragraph()
            p.text = line_text

            # Apply paragraph settings
            p.alignment = alignment
            p.space_before = space_before
            p.space_after = space_after
            p.line_spacing = line_spacing

            # Apply default font settings to the entire paragraph's font
            # (More complex formatting would require manipulating p.runs)
            _apply_font_settings(p.font, font_style)  # Raises errors on bad font config

            # Apply bullets if enabled
            if bullets_enabled:
                p.level = bullet_level
            # else: p.level = 0 # Default is 0 anyway

    def add_to_slide(self) -> Shape:
        """
        Executes the full process: creates the textbox shape and applies all styles.

        Returns:
        --------
        pptx.shapes.autoshape.Shape
            The created and styled PowerPoint textbox shape object.

        Raises:
        -------
        ValueError, TypeError, RuntimeError
            Propagates exceptions occurring during textbox creation or styling.
        """
        # 1. Create the basic shape
        self._create_textbox_object()

        # 2. Apply shape-level properties (fill, border, margins, etc.)
        self._apply_textbox_properties()

        # 3. Add text content and apply paragraph/font styles
        self._apply_text_and_paragraphs()

        # Auto-size might need a final adjustment after text is added,
        # especially for TEXT_TO_FIT_SHAPE. Re-applying isn't usually necessary
        # as setting it once often suffices. If issues arise, consider reapplying here.
        # if self.text_frame.auto_size == MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE:
        #    pass # Usually handled internally by pptx after text addition

        if not self._textbox_shape:
            raise RuntimeError("Textbox shape was not created successfully.")

        return self._textbox_shape


# --- Main Public Function ---


def add_textbox(
    slide: Slide,
    text_content: Union[str, List[str]],
    style_config: Optional[Dict[str, Any]] = None,
) -> Shape:
    """
    High-level function to add a styled textbox to a PowerPoint slide.

    Parameters:
    -----------
    slide : pptx.slide.Slide
        The PowerPoint slide object.
    text_content : Union[str, List[str]]
        The text to display. If a single string, it's treated as one paragraph.
        If a list of strings, each string becomes a separate paragraph.
    style_config : dict, optional
        A dictionary with custom style settings. These will be merged with
        the default style settings.

    Returns:
    --------
    pptx.shapes.autoshape.Shape
        The created and styled PowerPoint textbox shape object.

    Raises:
    -------
    ValueError, TypeError, RuntimeError
        Propagates exceptions from textbox creation or styling, typically due to
        invalid configuration (e.g., unknown alignment, bad color format) or
        issues adding the shape or text.
    """
    # 1. Ensure text_content is a list
    if isinstance(text_content, str):
        content_list = [text_content]
    elif isinstance(text_content, list):
        content_list = [
            str(item) for item in text_content
        ]  # Ensure all items are strings
    else:
        raise TypeError("text_content must be a string or a list of strings.")

    # 2. Create the final style dictionary. Raises TypeError if style_config is not dict.
    final_style = create_textbox_style(custom_style=style_config)

    # 3. Instantiate the formatter. Raises TypeError if internal checks fail.
    formatter = PowerPointTextboxFormatter(slide, content_list, final_style)

    # 4. Run the creation and styling process. Raises various errors on failure.
    textbox_shape = formatter.add_to_slide()

    return textbox_shape


# --- Example Usage (Optional - keep commented out or use for testing) ---
# if __name__ == "__main__":
#     from pptx import Presentation

#     # from pptx.util import Inches # Already imported above

#     # --- Create a presentation ---
#     prs = Presentation()
#     blank_slide_layout = prs.slide_layouts[5]  # Choose a blank layout

#     # --- Slide 1: Default Textbox ---
#     slide1 = prs.slides.add_slide(blank_slide_layout)
#     text_lines1 = ["This is the first paragraph.", "This is the second paragraph."]
#     print("Adding Textbox 1 (Default Style)...")
#     try:
#         shape1 = add_textbox(slide1, text_lines1)
#         print(f"Textbox 1 added. Shape ID: {shape1.shape_id}")
#     except Exception as e:
#         print(f"Failed to add Textbox 1: {type(e).__name__}: {e}")
#         import traceback

#         traceback.print_exc()

#     # --- Slide 2: Custom Styled Textbox with Bullets ---
#     slide2 = prs.slides.add_slide(blank_slide_layout)
#     text_lines2 = ["Introduction Point", "Detail A", "Detail B", "Conclusion Point"]
#     custom_style = {
#         "position": {"left": 1.5, "top": 2.0},
#         "dimensions": {"width": 7.0, "height": 2.5},  # Initial height, might grow
#         "font": {"size": 14, "bold": True, "color": "2F5496"},  # A dark blue color
#         "paragraph": {
#             "alignment": "left",
#             "space_after": Pt(6),  # Add some space after paragraphs
#         },
#         "bullets": {"enabled": True, "level": 1},  # Use level 1 bullets (standard)
#         "fill": {"type": "solid", "color": "F0F0F0"},  # Light grey background
#         "border": {
#             "visible": True,
#             "color": "2F5496",  # Match font color
#             "width": 1.5,
#         },
#         "auto_size": "shape_to_fit_text",  # Let shape grow vertically
#         "margins": {"left": 0.2, "right": 0.2, "top": 0.1, "bottom": 0.1},
#     }
#     print("\nAdding Textbox 2 (Custom Style with Bullets)...")
#     try:
#         shape2 = add_textbox(slide2, text_lines2, style_config=custom_style)
#         print(f"Textbox 2 added. Shape ID: {shape2.shape_id}")
#     except Exception as e:
#         print(f"Failed to add Textbox 2: {type(e).__name__}: {e}")
#         import traceback

#         traceback.print_exc()

#     # --- Slide 3: Single Line, Centered ---
#     slide3 = prs.slides.add_slide(blank_slide_layout)
#     text_line3 = "Centered Title Text"
#     style3 = {
#         "position": {"left": 1.0, "top": 0.5},
#         "dimensions": {"width": 8.0, "height": 0.5},
#         "font": {"size": 24, "bold": True, "color": (255, 0, 0)},  # Red via RGB tuple
#         "paragraph": {"alignment": "center"},
#         "auto_size": "none",  # Keep dimensions fixed
#         "vertical_anchor": "middle",  # Center text vertically too
#     }
#     print("\nAdding Textbox 3 (Single Line, Centered)...")
#     try:
#         # Pass a single string, function should handle it
#         shape3 = add_textbox(slide3, text_line3, style_config=style3)
#         print(f"Textbox 3 added. Shape ID: {shape3.shape_id}")
#     except Exception as e:
#         print(f"Failed to add Textbox 3: {type(e).__name__}: {e}")
#         import traceback

#         traceback.print_exc()

#     # --- Save the presentation ---
#     try:
#         output_filename = "styled_textboxes_test.pptx"
#         prs.save(output_filename)
#         print(f"\nPresentation saved as {output_filename}")
#     except PermissionError:
#         print(
#             f"\nError: Permission denied saving {output_filename}. Close the file if it's open."
#         )
#     except Exception as e:
#         print(f"\nAn error occurred while saving: {e}")
