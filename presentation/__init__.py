"""
Presentation module for creating PowerPoint presentations with tables, charts, textboxes, and themes.
"""

from .charts import add_chart
from .presentation import PowerPointPresentation, PowerPointSlide
from .tables import add_table
from .textbox import add_textbox
from .themes import (
    ColorScheme,
    FontScheme,
    PowerPointTheme,
    apply_theme_to_presentation,
    get_preset_theme,
)

__version__ = "0.3.0"
