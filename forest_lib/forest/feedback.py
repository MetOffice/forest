import os
import configparser

import bokeh.models.widgets

import pdb

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
        self.config_dir = os.path.split(self.config_path)[0]
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
        input_fields, self.feedback_dict = self._process_config(self.config_path)

        self.submit_button = bokeh.models.widgets.Button(label='Submit',
                                                         button_type='warning',
                                                         width=100)
        self.submit_button.on_click(self._on_submit)
        self.feedback_layout = bokeh.layouts.column(input_fields,
                                                    self.submit_button)


    def _process_config(self, path1, header_text=None):
        print('processing config file {0}'.format(path1))

        parser1 = configparser.RawConfigParser()
        parser1.read(path1)

        widgets_list = []
        if header_text:
            header_widget = bokeh.models.widgets.Div(text=header_text,
                                                     height=100,
                                                     width=400,
                                                     )
            widgets_list += [header_widget]

        config_dict = {}
        for section1 in parser1.sections():
            print('processing section {0}'.format(section1))
            section_dict = dict(parser1.items(section1))
            w1, d1 = self.loaders[section_dict['type']](section_dict)
            widgets_list += [w1]
            config_dict[section1] = section_dict
        config_layout = bokeh.layouts.column(*widgets_list)
        config_dict['widget'] = config_layout
        return config_layout, config_dict

    def _section_loader(self, section_dict, ):
        print('processing section {0}'.format(section_dict['title']))

        header_text = section_dict['title'] + '\n' + section_dict['description']
        section_path = os.path.join(self.config_dir,
                                    section_dict['file'])
        section_layout, section_dict = self._process_config(section_path,
                                                            header_text)
        return section_layout, section_dict


    def _yes_no_loader(self, section_dict):
        min1 = int(section_dict['min'])
        max1 = int(section_dict['max'])
        display_list1 = ['{0:d}'.format(i1) for i1 in range(min1,max1+1)]
        row_label = bokeh.models.widgets.Paragraph(text='No=1,Yes=10',
                                                   height=30,
                                                   width=200,
                                                   )
        row_buttons1 = bokeh.models.widgets.RadioButtonGroup(labels=display_list1,
                                                             active=0,
                                                             height=30,
                                                             width=500,
                                                             )
        row_layout1 = bokeh.layouts.Row(row_label, row_buttons1)

        section_dict['widget'] = row_layout1
        return row_layout1, section_dict

    def _perf_matrix_loader(self, section_dict):
        min1 = int(section_dict['min'])
        max1 = int(section_dict['max'])
        display_list1 = ['{0:d}'.format(i1) for i1 in range(min1,max1+1)]
        question_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict['question'],
                                           height=30,
                                           width=600,
                                           )
        row_list1 = [question_txt]
        row_layout_list1 = []
        category_list = section_dict['categories'].strip().split(',')
        category_list = [s1.strip() for s1 in category_list]
        for cat1 in category_list:
            row_label = bokeh.models.widgets.Paragraph(text=cat1,
                                                       height=30,
                                                       width=200,
                                                       )
            row_buttons1 = bokeh.models.widgets.RadioButtonGroup(labels=display_list1,
                                                                 active=0,
                                                                 height=30,
                                                                 width=500,
                                                                 )
            row_layout1 = bokeh.layouts.Row(row_label, row_buttons1)
            row_list1 += [row_layout1]
            row_layout_list1 += [row_layout1]
        row_list1 += [bokeh.layouts.Spacer(width=600,height=30)]
        matrix_layout = bokeh.layouts.column(*row_list1)
        section_dict['widget'] = matrix_layout
        section_dict['category_widgets'] = row_list1

        return matrix_layout, section_dict

    def _selection_loader(self, section_dict):
        option_list = [(s1.strip(),s1.strip()) for s1 in section_dict['values'].split(',')]
        selection_dd = bokeh.models.widgets.Dropdown(label=section_dict['question'],
                                                     menu=option_list)
        section_dict['widget'] = selection_dd
        return selection_dd, section_dict

    def _text_input_loader(self, section_dict):
        text_widget1 = bokeh.models.widgets.TextInput(value='',
                                       title=section_dict['question'])
        section_dict['widget'] = text_widget1
        return text_widget1, section_dict

    def get_feedback_widgets(self):
        return self.feedback_layout


    def _on_submit(self):
        raise NotImplementedError()