from pptx import Presentation


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
        if subtitle:
            title_placeholder = self.slide.shapes.title
            title_placeholder.text = title
            subtitle_placeholder = self.slide.placeholders[1]
            subtitle_placeholder.text = subtitle
        else:
            title_placeholder = self.slide.shapes.title
            title_placeholder.text = title

    def add_textbox(self, left, top, width, height, text):
        textbox = self.slide.shapes.add_textbox(left, top, width, height)
        textbox.text = text
