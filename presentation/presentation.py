from pptx import Presentation
from pptx.util import Inches, Pt

from .table_creator import create_table


class PowerPointPresentation:
    def __init__(self, template_filepath=None, config="presentation/config.json"):
        self.presentation = Presentation(template_filepath)
        self.config = self._read_config(config)

    def _read_config(self, config):
        import json

        with open(config, "r") as f:
            config_data = json.load(f)
        return config_data

    def add_slide(self, template_index_or_name, config=None):
        if config is None:
            config = self.config
        return PowerPointSlide(self.presentation, template_index_or_name, config)

    def save(self, filepath):
        self.presentation.save(filepath)


class PowerPointSlide:
    def __init__(self, presentation, template_index_or_name=None, config=None):
        self.presentation = presentation
        self.config = config
        self.slide = self._initialize_slide(template_index_or_name)

    def _get_layout_index_by_name(self, name):
        template_name_dict = {
            sl.name: i for i, sl in enumerate(self.presentation.slide_layouts)
        }
        if name in template_name_dict:
            return template_name_dict[name]
        else:
            raise ValueError(
                f"Layouts name '{name}' not found. Available layouts: {list(template_name_dict.keys())}"
            )

    def _initialize_slide(self, template_index_or_name=None):
        if template_index_or_name is None:
            template_index = 0
        elif isinstance(template_index_or_name, int):
            template_index = template_index_or_name
        elif isinstance(template_index_or_name, str):
            template_index = self._get_layout_index_by_name(template_index_or_name)
        else:
            raise ValueError(
                "template_index_or_name must be an int or str, got {type(template_index_or_name)}"
            )
        slide_layout = self.presentation.slide_layouts[template_index]
        return self.presentation.slides.add_slide(slide_layout)

    def set_title(self, title, subtitle=None):
        # set title
        title_placeholder = self.slide.shapes.title
        title_text_frame = title_placeholder.text_frame
        title_p = title_text_frame.paragraphs[0]
        title_p.text = title
        title_p.font.name = self.config["elements"]["title"]["font"]["name"]
        title_p.font.size = Pt(self.config["elements"]["title"]["font"]["size"])
        title_p.font.bold = self.config["elements"]["title"]["font"]["bold"]
        title_p.font.italic = self.config["elements"]["title"]["font"]["italic"]

        # set subtitle
        if subtitle is not None:
            subtitle_run = title_p.add_run()
            subtitle_run.text = "\n" + subtitle
            subtitle_run.font.name = self.config["elements"]["subtitle"]["font"]["name"]
            subtitle_run.font.size = Pt(
                self.config["elements"]["subtitle"]["font"]["size"]
            )
            subtitle_run.font.bold = self.config["elements"]["subtitle"]["font"]["bold"]
            subtitle_run.font.italic = self.config["elements"]["subtitle"]["font"][
                "italic"
            ]

    def add_textbox(self, text, left, top, width, height):
        left = Pt(left)
        top = Pt(top)
        width = Pt(width)
        height = Pt(height)

        textbox = self.slide.shapes.add_textbox(left, top, width, height)
        text_frame = textbox.text_frame
        text_frame.word_wrap = True

        p = text_frame.add_paragraph()
        p.text = text
        p.font.name = self.config["elements"]["textbox"]["font"]["name"]
        p.font.size = Pt(self.config["elements"]["textbox"]["font"]["size"])
        p.font.bold = self.config["elements"]["textbox"]["font"]["bold"]
        p.font.italic = self.config["elements"]["textbox"]["font"]["italic"]

        return textbox

    def add_table(self, data, left, top, width=None, height=None, style_settings=None):
        """
        Add a table to the slide based on DataFrame data.

        Args:
            data: pandas DataFrame containing the table data
            left: Left position of the table in inches
            top: Top position of the table in inches
            width: Width of the table in inches, or None for auto-width
            height: Height of the table in inches, or None for auto-height
            style_settings: Optional dictionary with table style settings.
                            If None, uses settings from the presentation config file.

        Returns:
            The created table object
        """
        # Convert position to inches
        left_inches = Inches(left)
        top_inches = Inches(top)

        # Handle width and height
        width_inches = Inches(width) if width is not None else None
        height_inches = Inches(height) if height is not None else None

        # Prepare style settings
        if style_settings is None:
            # Get default table styling from config
            style_settings = {}
            if self.config and "elements" in self.config:
                # Extract relevant table styling from config
                if "table_header" in self.config["elements"]:
                    style_settings["header"] = self.config["elements"]["table_header"]
                if "defaults" in self.config:
                    style_settings["defaults"] = self.config["defaults"]

        # Call the table creator function
        table = create_table(
            self.slide,
            data,
            left_inches,
            top_inches,
            width_inches,
            height_inches,
            style_settings,
        )

        return table
