"""
Presentation module for creating PowerPoint presentations with tables, charts, and textboxes.
"""

from .presentation import PowerPointPresentation, PowerPointSlide
from .charts import add_chart
from .tables import add_table
from .textbox import add_textbox

__version__ = "0.2.0"
