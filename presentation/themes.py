"""
Utilities for managing and applying PowerPoint themes.

Provides classes and functions for creating, saving, loading, and applying themes
to PowerPoint presentations, including color schemes, font schemes, and master slide management.
"""

import json
from typing import Any, Dict, Optional, Tuple, Union

from pptx import Presentation as PptxPresentation

from .utils import get_color_from_value


class ColorScheme:
    """
    Represents a PowerPoint color scheme with primary, accent, and neutral colors.
    """

    def __init__(self, name: str = "Default"):
        """
        Initialize a color scheme.

        Parameters:
        -----------
        name : str, default="Default"
            The name of the color scheme.
        """
        self.name = name
        self._colors = {
            # Theme colors (MSO_THEME_COLOR enum values as keys)
            "dark1": "#000000",  # Text/background - dark
            "light1": "#FFFFFF",  # Text/background - light
            "dark2": "#44546A",  # Text/background - dark 2
            "light2": "#E7E6E6",  # Text/background - light 2
            "accent1": "#4472C4",  # Accent 1
            "accent2": "#ED7D31",  # Accent 2
            "accent3": "#A5A5A5",  # Accent 3
            "accent4": "#FFC000",  # Accent 4
            "accent5": "#5B9BD5",  # Accent 5
            "accent6": "#70AD47",  # Accent 6
            "hyperlink": "#0563C1",  # Hyperlink color
            "followed_hyperlink": "#954F72",  # Followed hyperlink color
        }

    @property
    def colors(self) -> Dict[str, str]:
        """Get the color dictionary."""
        return self._colors.copy()

    def set_color(self, name: str, color: Union[str, Tuple[int, int, int]]) -> None:
        """
        Set a color in the scheme.

        Parameters:
        -----------
        name : str
            The name of the color to set.
        color : Union[str, Tuple[int, int, int]]
            The color value as a hex string or RGB tuple.
        """
        if name not in self._colors and not name.startswith("custom_"):
            raise ValueError(
                f"Invalid color name: {name}. Use a theme color name or a custom name starting with 'custom_'."
            )

        # Convert RGB tuple to hex string if needed
        if isinstance(color, tuple) and len(color) == 3:
            r, g, b = color
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            self._colors[name] = hex_color
        elif isinstance(color, str):
            # Ensure hex string format
            if color.startswith("#") and (len(color) == 7 or len(color) == 9):
                self._colors[name] = color
            else:
                raise ValueError(
                    f"Invalid hex color format: {color}. Use #RRGGBB format."
                )
        else:
            raise TypeError("Color must be a hex string or RGB tuple.")

    def get_color(self, name: str) -> str:
        """
        Get a color from the scheme.

        Parameters:
        -----------
        name : str
            The name of the color to get.

        Returns:
        --------
        str
            The color value as a hex string.

        Raises:
        -------
        KeyError
            If the color name is not found in the scheme.
        """
        if name not in self._colors:
            raise KeyError(f"Color '{name}' not found in the scheme.")
        return self._colors[name]

    def get_rgb_color(self, name: str) -> Tuple[int, int, int]:
        """
        Get a color from the scheme as an RGB tuple.

        Parameters:
        -----------
        name : str
            The name of the color to get.

        Returns:
        --------
        Tuple[int, int, int]
            The color value as an RGB tuple.

        Raises:
        -------
        KeyError
            If the color name is not found in the scheme.
        """
        hex_color = self.get_color(name)
        # Remove # and convert to RGB
        hex_str = hex_color.lstrip("#")
        return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the color scheme to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            A dictionary representation of the color scheme.
        """
        return {"name": self.name, "colors": self._colors.copy()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColorScheme":
        """
        Create a color scheme from a dictionary.

        Parameters:
        -----------
        data : Dict[str, Any]
            A dictionary containing color scheme data.

        Returns:
        --------
        ColorScheme
            A new ColorScheme instance.
        """
        scheme = cls(name=data.get("name", "Default"))
        colors = data.get("colors", {})
        for name, color in colors.items():
            scheme.set_color(name, color)
        return scheme


class FontScheme:
    """
    Represents a PowerPoint font scheme with heading and body fonts.
    """

    def __init__(self, name: str = "Default"):
        """
        Initialize a font scheme.

        Parameters:
        -----------
        name : str, default="Default"
            The name of the font scheme.
        """
        self.name = name
        self._fonts = {
            "heading": {
                "latin": "Calibri Light",
                "east_asian": None,
                "complex_script": None,
            },
            "body": {"latin": "Calibri", "east_asian": None, "complex_script": None},
        }
        self._font_sizes = {
            "title": 44,
            "subtitle": 32,
            "heading1": 28,
            "heading2": 24,
            "heading3": 20,
            "heading4": 16,
            "body": 12,
            "footer": 10,
            "small": 8,
        }

    @property
    def fonts(self) -> Dict[str, Dict[str, Optional[str]]]:
        """Get the font dictionary."""
        return self._fonts.copy()

    @property
    def font_sizes(self) -> Dict[str, int]:
        """Get the font size dictionary."""
        return self._font_sizes.copy()

    def set_font(self, category: str, script: str, font_name: str) -> None:
        """
        Set a font in the scheme.

        Parameters:
        -----------
        category : str
            The font category ('heading' or 'body').
        script : str
            The script type ('latin', 'east_asian', or 'complex_script').
        font_name : str
            The name of the font.

        Raises:
        -------
        ValueError
            If the category or script is invalid.
        """
        if category not in ["heading", "body"]:
            raise ValueError("Category must be 'heading' or 'body'.")
        if script not in ["latin", "east_asian", "complex_script"]:
            raise ValueError(
                "Script must be 'latin', 'east_asian', or 'complex_script'."
            )

        self._fonts[category][script] = font_name

    def set_font_size(self, element: str, size: int) -> None:
        """
        Set a font size in the scheme.

        Parameters:
        -----------
        element : str
            The element type (e.g., 'title', 'body').
        size : int
            The font size in points.

        Raises:
        -------
        ValueError
            If the size is not a positive integer.
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Font size must be a positive integer.")

        self._font_sizes[element] = size

    def get_font(self, category: str, script: str = "latin") -> Optional[str]:
        """
        Get a font from the scheme.

        Parameters:
        -----------
        category : str
            The font category ('heading' or 'body').
        script : str, default="latin"
            The script type ('latin', 'east_asian', or 'complex_script').

        Returns:
        --------
        Optional[str]
            The font name, or None if not set.

        Raises:
        -------
        ValueError
            If the category or script is invalid.
        """
        if category not in ["heading", "body"]:
            raise ValueError("Category must be 'heading' or 'body'.")
        if script not in ["latin", "east_asian", "complex_script"]:
            raise ValueError(
                "Script must be 'latin', 'east_asian', or 'complex_script'."
            )

        return self._fonts[category][script]

    def get_font_size(self, element: str) -> int:
        """
        Get a font size from the scheme.

        Parameters:
        -----------
        element : str
            The element type (e.g., 'title', 'body').

        Returns:
        --------
        int
            The font size in points.

        Raises:
        -------
        KeyError
            If the element is not found in the scheme.
        """
        if element not in self._font_sizes:
            raise KeyError(f"Element '{element}' not found in the font size scheme.")

        return self._font_sizes[element]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the font scheme to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            A dictionary representation of the font scheme.
        """
        return {
            "name": self.name,
            "fonts": self._fonts.copy(),
            "font_sizes": self._font_sizes.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FontScheme":
        """
        Create a font scheme from a dictionary.

        Parameters:
        -----------
        data : Dict[str, Any]
            A dictionary containing font scheme data.

        Returns:
        --------
        FontScheme
            A new FontScheme instance.
        """
        scheme = cls(name=data.get("name", "Default"))

        # Set fonts
        fonts = data.get("fonts", {})
        for category, scripts in fonts.items():
            if category in ["heading", "body"] and isinstance(scripts, dict):
                for script, font_name in scripts.items():
                    if (
                        script in ["latin", "east_asian", "complex_script"]
                        and font_name
                    ):
                        scheme.set_font(category, script, font_name)

        # Set font sizes
        font_sizes = data.get("font_sizes", {})
        for element, size in font_sizes.items():
            if isinstance(size, int) and size > 0:
                scheme.set_font_size(element, size)

        return scheme


class PowerPointTheme:
    """
    Represents a complete PowerPoint theme with color and font schemes.
    """

    def __init__(self, name: str = "Default"):
        """
        Initialize a PowerPoint theme.

        Parameters:
        -----------
        name : str, default="Default"
            The name of the theme.
        """
        self.name = name
        self.color_scheme = ColorScheme(name=f"{name} Colors")
        self.font_scheme = FontScheme(name=f"{name} Fonts")
        self._background_style = {
            "type": "solid",
            "color": "#FFFFFF",
            "gradient": None,
            "image": None,
        }
        self._slide_layouts = {}

    @property
    def background_style(self) -> Dict[str, Any]:
        """Get the background style dictionary."""
        return self._background_style.copy()

    def set_background_style(self, style_type: str, **kwargs) -> None:
        """
        Set the background style for the theme.

        Parameters:
        -----------
        style_type : str
            The type of background ('solid', 'gradient', or 'image').
        **kwargs
            Additional style parameters depending on the type.
            For 'solid': color (str or tuple)
            For 'gradient': start_color, end_color, direction
            For 'image': path, transparency

        Raises:
        -------
        ValueError
            If the style type is invalid or required parameters are missing.
        """
        if style_type not in ["solid", "gradient", "image"]:
            raise ValueError("Style type must be 'solid', 'gradient', or 'image'.")

        new_style = {"type": style_type}

        if style_type == "solid":
            if "color" not in kwargs:
                raise ValueError("Solid background requires a 'color' parameter.")
            color = kwargs["color"]
            if isinstance(color, tuple) and len(color) == 3:
                r, g, b = color
                hex_color = f"#{r:02X}{g:02X}{b:02X}"
                new_style["color"] = hex_color
            elif isinstance(color, str):
                if color.startswith("#") and (len(color) == 7 or len(color) == 9):
                    new_style["color"] = color
                else:
                    raise ValueError(
                        f"Invalid hex color format: {color}. Use #RRGGBB format."
                    )
            else:
                raise TypeError("Color must be a hex string or RGB tuple.")

        elif style_type == "gradient":
            required = ["start_color", "end_color", "direction"]
            for param in required:
                if param not in kwargs:
                    raise ValueError(
                        f"Gradient background requires a '{param}' parameter."
                    )

            gradient_info = {
                "start_color": None,
                "end_color": None,
                "direction": kwargs["direction"],
            }

            # Process start color
            start_color = kwargs["start_color"]
            if isinstance(start_color, tuple) and len(start_color) == 3:
                r, g, b = start_color
                gradient_info["start_color"] = f"#{r:02X}{g:02X}{b:02X}"
            elif isinstance(start_color, str) and start_color.startswith("#"):
                gradient_info["start_color"] = start_color
            else:
                raise ValueError("Invalid start_color format.")

            # Process end color
            end_color = kwargs["end_color"]
            if isinstance(end_color, tuple) and len(end_color) == 3:
                r, g, b = end_color
                gradient_info["end_color"] = f"#{r:02X}{g:02X}{b:02X}"
            elif isinstance(end_color, str) and end_color.startswith("#"):
                gradient_info["end_color"] = end_color
            else:
                raise ValueError("Invalid end_color format.")

            new_style["gradient"] = gradient_info

        elif style_type == "image":
            if "path" not in kwargs:
                raise ValueError("Image background requires a 'path' parameter.")

            image_info = {
                "path": kwargs["path"],
                "transparency": kwargs.get("transparency", 0),
            }

            new_style["image"] = image_info

        self._background_style = new_style

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the theme to a dictionary.

        Returns:
        --------
        Dict[str, Any]
            A dictionary representation of the theme.
        """
        return {
            "name": self.name,
            "color_scheme": self.color_scheme.to_dict(),
            "font_scheme": self.font_scheme.to_dict(),
            "background_style": self._background_style.copy(),
            "slide_layouts": self._slide_layouts.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PowerPointTheme":
        """
        Create a theme from a dictionary.

        Parameters:
        -----------
        data : Dict[str, Any]
            A dictionary containing theme data.

        Returns:
        --------
        PowerPointTheme
            A new PowerPointTheme instance.
        """
        theme = cls(name=data.get("name", "Default"))

        # Set color scheme
        color_scheme_data = data.get("color_scheme")
        if color_scheme_data:
            theme.color_scheme = ColorScheme.from_dict(color_scheme_data)

        # Set font scheme
        font_scheme_data = data.get("font_scheme")
        if font_scheme_data:
            theme.font_scheme = FontScheme.from_dict(font_scheme_data)

        # Set background style
        background_style = data.get("background_style")
        if background_style and "type" in background_style:
            style_type = background_style["type"]

            if style_type == "solid" and "color" in background_style:
                theme.set_background_style("solid", color=background_style["color"])

            elif style_type == "gradient" and "gradient" in background_style:
                gradient = background_style["gradient"]
                if all(
                    k in gradient for k in ["start_color", "end_color", "direction"]
                ):
                    theme.set_background_style(
                        "gradient",
                        start_color=gradient["start_color"],
                        end_color=gradient["end_color"],
                        direction=gradient["direction"],
                    )

            elif style_type == "image" and "image" in background_style:
                image = background_style["image"]
                if "path" in image:
                    theme.set_background_style(
                        "image",
                        path=image["path"],
                        transparency=image.get("transparency", 0),
                    )

        # Set slide layouts
        theme._slide_layouts = data.get("slide_layouts", {})

        return theme

    def save(self, filepath: str) -> None:
        """
        Save the theme to a JSON file.

        Parameters:
        -----------
        filepath : str
            The path where the theme will be saved.
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "PowerPointTheme":
        """
        Load a theme from a JSON file.

        Parameters:
        -----------
        filepath : str
            The path to the theme file.

        Returns:
        --------
        PowerPointTheme
            A new PowerPointTheme instance.

        Raises:
        -------
        FileNotFoundError
            If the file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def apply_to_presentation(self, presentation: PptxPresentation) -> None:
        """
        Apply the theme to a PowerPoint presentation.

        Parameters:
        -----------
        presentation : pptx.Presentation
            The PowerPoint presentation to apply the theme to.

        Note:
        -----
        This method applies the theme to the presentation's slide master,
        which affects all slides in the presentation. It does not modify
        existing content, only the theme elements.
        """
        # Apply to slide master
        if presentation.slide_masters:
            master = presentation.slide_masters[0]

            # Apply theme colors
            # Note: Direct theme color modification is limited in python-pptx
            # This is a partial implementation

            # Apply theme fonts
            # Note: Direct theme font modification is limited in python-pptx
            # This is a partial implementation

            # Apply background style to slide master
            background = master.background
            fill = background.fill

            if self._background_style["type"] == "solid":
                fill.solid()
                color = get_color_from_value(self._background_style["color"])
                fill.fore_color.rgb = color

            # Note: Gradient and image backgrounds require more complex OOXML manipulation
            # which is beyond the scope of this implementation

            # Apply to layouts
            for layout in master.slide_layouts:
                # Apply background if not inherited
                if not layout.follow_master_background:
                    layout_background = layout.background
                    layout_fill = layout_background.fill

                    if self._background_style["type"] == "solid":
                        layout_fill.solid()
                        color = get_color_from_value(self._background_style["color"])
                        layout_fill.fore_color.rgb = color


def create_corporate_theme() -> PowerPointTheme:
    """
    Create a corporate-style theme.

    Returns:
    --------
    PowerPointTheme
        A new PowerPointTheme instance with corporate styling.
    """
    theme = PowerPointTheme(name="Corporate")

    # Set color scheme
    theme.color_scheme.set_color("dark1", "#000000")
    theme.color_scheme.set_color("light1", "#FFFFFF")
    theme.color_scheme.set_color("dark2", "#44546A")
    theme.color_scheme.set_color("light2", "#E7E6E6")
    theme.color_scheme.set_color("accent1", "#4472C4")
    theme.color_scheme.set_color("accent2", "#ED7D31")
    theme.color_scheme.set_color("accent3", "#A5A5A5")
    theme.color_scheme.set_color("accent4", "#FFC000")
    theme.color_scheme.set_color("accent5", "#5B9BD5")
    theme.color_scheme.set_color("accent6", "#70AD47")

    # Set font scheme
    theme.font_scheme.set_font("heading", "latin", "Arial")
    theme.font_scheme.set_font("body", "latin", "Arial")

    # Set background style
    theme.set_background_style("solid", color="#FFFFFF")

    return theme


def create_modern_theme() -> PowerPointTheme:
    """
    Create a modern-style theme.

    Returns:
    --------
    PowerPointTheme
        A new PowerPointTheme instance with modern styling.
    """
    theme = PowerPointTheme(name="Modern")

    # Set color scheme
    theme.color_scheme.set_color("dark1", "#212121")
    theme.color_scheme.set_color("light1", "#FFFFFF")
    theme.color_scheme.set_color("dark2", "#333333")
    theme.color_scheme.set_color("light2", "#F5F5F5")
    theme.color_scheme.set_color("accent1", "#03A9F4")
    theme.color_scheme.set_color("accent2", "#FF5722")
    theme.color_scheme.set_color("accent3", "#607D8B")
    theme.color_scheme.set_color("accent4", "#FFC107")
    theme.color_scheme.set_color("accent5", "#4CAF50")
    theme.color_scheme.set_color("accent6", "#9C27B0")

    # Set font scheme
    theme.font_scheme.set_font("heading", "latin", "Segoe UI Light")
    theme.font_scheme.set_font("body", "latin", "Segoe UI")

    # Set background style
    theme.set_background_style("solid", color="#FFFFFF")

    return theme


def create_minimal_theme() -> PowerPointTheme:
    """
    Create a minimal-style theme.

    Returns:
    --------
    PowerPointTheme
        A new PowerPointTheme instance with minimal styling.
    """
    theme = PowerPointTheme(name="Minimal")

    # Set color scheme
    theme.color_scheme.set_color("dark1", "#000000")
    theme.color_scheme.set_color("light1", "#FFFFFF")
    theme.color_scheme.set_color("dark2", "#333333")
    theme.color_scheme.set_color("light2", "#F8F8F8")
    theme.color_scheme.set_color("accent1", "#2196F3")
    theme.color_scheme.set_color("accent2", "#F44336")
    theme.color_scheme.set_color("accent3", "#9E9E9E")
    theme.color_scheme.set_color("accent4", "#FFEB3B")
    theme.color_scheme.set_color("accent5", "#4CAF50")
    theme.color_scheme.set_color("accent6", "#9C27B0")

    # Set font scheme
    theme.font_scheme.set_font("heading", "latin", "Helvetica Neue")
    theme.font_scheme.set_font("body", "latin", "Helvetica Neue")

    # Set background style
    theme.set_background_style("solid", color="#FFFFFF")

    return theme


def get_preset_theme(name: str) -> PowerPointTheme:
    """
    Get a predefined theme by name.

    Parameters:
    -----------
    name : str
        The name of the theme ('corporate', 'modern', or 'minimal').

    Returns:
    --------
    PowerPointTheme
        A new PowerPointTheme instance.

    Raises:
    -------
    ValueError
        If the theme name is not recognized.
    """
    name_lower = name.lower()

    if name_lower == "corporate":
        return create_corporate_theme()
    elif name_lower == "modern":
        return create_modern_theme()
    elif name_lower == "minimal":
        return create_minimal_theme()
    else:
        raise ValueError(
            f"Unknown theme name: {name}. Available themes: 'corporate', 'modern', 'minimal'."
        )


def extract_theme_from_presentation(presentation: PptxPresentation) -> PowerPointTheme:
    """
    Extract theme information from an existing presentation.

    Parameters:
    -----------
    presentation : pptx.Presentation
        The PowerPoint presentation to extract the theme from.

    Returns:
    --------
    PowerPointTheme
        A new PowerPointTheme instance with the extracted theme.

    Note:
    -----
    This is a partial implementation. Full theme extraction requires
    complex OOXML manipulation which is beyond the scope of this implementation.
    """
    theme = PowerPointTheme(name="Extracted Theme")

    # Extract from slide master if available
    if presentation.slide_masters:
        master = presentation.slide_masters[0]

        # Extract background style
        background = master.background
        fill = background.fill

        if fill.type:
            if fill.type == 1:  # MSO_FILL.SOLID
                theme.set_background_style(
                    "solid", color=fill.fore_color.rgb.hex_string
                )

    return theme


def apply_theme_to_presentation(
    presentation: PptxPresentation, theme: PowerPointTheme
) -> None:
    """
    Apply a theme to a PowerPoint presentation.

    Parameters:
    -----------
    presentation : pptx.Presentation
        The PowerPoint presentation to apply the theme to.
    theme : PowerPointTheme
        The theme to apply.

    Note:
    -----
    This is a convenience function that calls theme.apply_to_presentation().
    """
    theme.apply_to_presentation(presentation)
