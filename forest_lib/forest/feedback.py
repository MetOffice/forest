import os
import configparser

import bokeh.moodels.widgets

KEY_SECTION = 'section'
KEY_PERFORMANCE_MATRIX = 'performance_matrix'
KEY_YES_NO = 'yes_no'
KEY_SELECTION = 'selection'
KEY_TEXT = 'text'


class UserFeedback(object):
    """
    Class for display widgets to gather feedback on model performance from
    research and operational meteorologists using Forest. Implementation
    aims to conform to the Composite software design pattern, so the feedback form
    can easily be modified or extended.
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.config_dir = os.path.split(self.config_path)
        self.loaders = None
        self._setup_loaders()

        self.bokeh_widgets_top_level = None
        self._create_widgets()

    def __str__(self):
        return 'Class to collect feedback on model assessment from Meteorologists.'

    def _setup_loaders(self):
        self.loaders = {}
        self.loaders[KEY_SECTION] = self._section_loader
        self.loaders[KEY_YES_NO] = self._yes_no_loader
        self.loaders[KEY_PERFORMANCE_MATRIX] = self._perf_matrix_loader
        self.loaders[KEY_SELECTION] = self._selection_loader
        self.loaders[KEY_TEXT] = self._text_input_loader


    def _create_widgets(self):
        self.feedback_layout, self.feedback_dict = self._process_config(self.config_path)

        self.submit_button = bokeh.models.widgets.Button(label='Submit',
                                                         button_type='warning',
                                                         width=100)
        self.submit_button.on_click(self._on_submit)

    def _process_config(self, path1):
        parser1 = configparser.RawConfigParser()
        parser1.read(self.config_path)

        widgets_list = []
        config_dict = {}
        for section1 in parser1.sections():
            section_dict = parser1.items(section1)
            w1, d1 = self.loaders[section_dict['type']]()
            widgets_list += [w1]
            config_dict[section1] = section_dict
        config_layout = bokeh.layouts.column(*widgets_list)
        config_dict['widget'] = config_layout
        return config_layout, config_dict

    def _section_loader(self, section_dict):
        header_text = section_dict['title'] + '\n' + section_dict['description']
        header_widget = bokeh.models.widgets.Div(text=header_text,
                                               height=100,
                                               width=400,
                                               )
        section_path = os.path.join(self.config_dir,
                                    section_dict['file'])
        section_layout, section_dict = self._process_config(section_path)
        return section_layout, section_dict


    def _yes_no_loader(self, section_dict):
        raise NotImplementedError()

    def _perf_matrix_loader(self, section_dict):
        raise NotImplementedError()
        min1 = int(section_dict['min'])
        max1 = int(section_dict['max'])
        display_list1 = ['{0:d}'.format(i1) for i1 in range(min1,max1+1)]
        row_list1 = []
        row_layout_list1 = []
        for cat1 in section_dict['categories']:
            row_label = bokeh.models.widgets.Paragraph(text=cat1)
            row_buttons1 = bokeh.models.widgets.RadioButtonGroup(labels=display_list1,
                                                                 active=0,
                                                                 )
            row_layout1 = bokeh.layouts.row(row_label, row_buttons1)
            row_list1 += [row_buttons1]
            row_layout_list1 += [row_layout1]

        matrix_layout = bokeh.layouts.colum(*row_list1)
        section_dict['widget'] = matrix_layout
        section_dict['category_widgets'] = row_list1

        return matrix_layout, section_dict

    def _selection_loader(self, section_dict):
        bokeh.models.widgets.DropDown(label=section_dict['question'])

    def _text_input_loader(self, section_dict):
        text_widget1 = bokeh.models.widgets.TextInput(value='',
                                       title=section_dict['question'])
        return text_widget1

    def get_feedback_widgets(self):
        return self.feedback_layout


    def _on_submit(self):
        raise NotImplementedError()