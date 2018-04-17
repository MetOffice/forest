"""
Module for creating GUI elements to get user feedback on the data displayed
by forest, and gather up the feedback and write it out as a csv file.
"""
import os
import configparser
import datetime
import csv

import bokeh.models.widgets


KEY_SECTION = 'section'
KEY_PERFORMANCE_MATRIX = 'performance_matrix'
KEY_PERFMAT_LABELLED = 'performance_matrix_labelled'
KEY_YES_NO = 'yes_no'
KEY_SELECTION = 'selection'
KEY_TEXT = 'text'
KEY_MULTISELECT = 'multiselect'

LABEL_QUESTION = 'question'
LABEL_CATEGORY = 'category'
LABEL_TAG = 'tag'
LABEL_VALUE = 'value'
LABEL_TITLE = 'title'

WIDGET_HEIGHT = 30
SECTION_TITLE_HEIGHT = 100
TEXT_LABEL_WIDTH = 200
SINGLE_COLUMN_WIDTH = 400
MULTI_COLUMN_WIDTH = 800
BUTTON_WIDTH = 100
DETAILED_TEXT_HEIGHT = 400
DETAILED_TEXT_WIDTH = 800

FEEDBACK_HEADER_ORDER = [LABEL_TAG,
                         LABEL_QUESTION,
                         LABEL_CATEGORY,
                         LABEL_VALUE]

class UserFeedback(object):
    """
    Class for display widgets to gather feedback on model performance from
    research and operational meteorologists using Forest. Implementation
    aims to conform to the Composite software design pattern, so the feedback
    form can easily be modified or extended.
    """

    def __init__(self,
                 config_path,
                 feedback_dir,
                 bokeh_id,
                ):
        """

        Arguments
        ---------

        - config_path -- string: the path to the main config file to load
        - feedback_dir -- string: the directory path to write the user
                                  feedback file to.
        - bokeh_id -- string: The bokeh id for the current user session.
        """
        self.config_path = config_path
        self.config_dir = os.path.split(self.config_path)[0]
        self.feedback_dir = feedback_dir
        self.bokeh_id = bokeh_id

        self.loaders = None
        self._setup_loaders()

        self.bokeh_widgets_top_level = None
        self._create_widgets()
        self.answer_list = []

    def __str__(self):

        return \
            'Class to collect feedback on model assessment ' \
            + 'from Meteorologists.'

    def _setup_loaders(self):
        """
        Setup function to create dictionaries of loader and getter functions
        to create GUI elements based on conf files and retrieve values
        from those GUI elements.
        """
        self.loaders = {}
        self.loaders[KEY_SECTION] = self._section_loader
        self.loaders[KEY_YES_NO] = self._yes_no_loader
        self.loaders[KEY_PERFORMANCE_MATRIX] = self._perf_matrix_loader
        self.loaders[KEY_SELECTION] = self._selection_loader
        self.loaders[KEY_TEXT] = self._text_input_loader
        self.loaders[KEY_MULTISELECT] = self._multiselect_loader
        self.loaders[KEY_PERFMAT_LABELLED] = self._perfmatlab_loader

        self.getters = {}
        self.getters[KEY_SECTION] = self._section_getter
        self.getters[KEY_YES_NO] = self._yes_no_getter
        self.getters[KEY_PERFORMANCE_MATRIX] = self._perf_matrix_getter
        self.getters[KEY_SELECTION] = self._selection_getter
        self.getters[KEY_TEXT] = self._text_input_getter
        self.getters[KEY_MULTISELECT] = self._multiselect_getter
        self.getters[KEY_PERFMAT_LABELLED] = self._perfmatlab_getter


    def _create_widgets(self):
        """
        Main setup function for creating GUI widgets for user feedback.
        """
        input_fields, self.feedback_dict = \
            self._process_config(self.config_path)

        self.submit_button = bokeh.models.widgets.Button(label='Submit',
                                                         button_type='warning',
                                                         width=BUTTON_WIDTH)
        self.submit_button.on_click(self._on_submit)
        self.feedback_layout = bokeh.layouts.column(input_fields,
                                                    self.submit_button)


    def _process_config(self, path1,
                        title_text=None,
                        header_text=None):
        """
        Called by _create_widgets to set up the GUI. Parses the config files
        and creates the specified widgets.
        - path1 -- string: path to the config file.
        - title_text -- : Title of the config section to be created by this
                          config file.
        - header_text -- : Descriptive header text of the config section to be
                           created by this config file.
        """
        print('processing config file {0}'.format(path1))

        parser1 = configparser.RawConfigParser()
        parser1.read(path1)

        widgets_list = []
        if header_text:
            title_widget = \
                bokeh.models.widgets.Div(text=title_text,
                                         height=SECTION_TITLE_HEIGHT,
                                         width=MULTI_COLUMN_WIDTH,
                                        )
            header_widget = \
                bokeh.models.widgets.Div(text=header_text,
                                         height=SECTION_TITLE_HEIGHT,
                                         width=MULTI_COLUMN_WIDTH,
                                        )
            widgets_list += [title_widget, header_widget]

        config_dict = {}
        for section1 in parser1.sections():
            print('processing section {0}'.format(section1))
            section_dict = dict(parser1.items(section1))
            section_dict[LABEL_TAG] = section1
            w1, d1 = self.loaders[section_dict['type']](section_dict)
            widgets_list += [w1]
            config_dict[section1] = section_dict
        config_layout = bokeh.layouts.column(*widgets_list)

        return config_layout, config_dict

    def _section_loader(self, section_dict, ):
        """
        The loader function for a feedback section, which creates GUI elements
        from a config file.
        - section_dict -- python dictionary: the specs for the feedback
                                             section, based on the config
                                             file
        return: tuple containing a bokeh gui widget representing this section
                and an updated dictionary file.`
        """
        print('processing section {0}'.format(section_dict[LABEL_TITLE]))

        title_text = '<h1>{label}</h1>'.format(label=section_dict[LABEL_TITLE])
        header_text = \
            '<h2>{desc}</h2>'.format(desc=section_dict['description'])


        section_path = os.path.join(self.config_dir,
                                    section_dict['file'])
        children_layout, children_dict = \
            self._process_config(section_path,
                                 title_text=title_text,
                                 header_text=header_text)

        section_dict['widget'] = children_layout
        section_dict['children'] = children_dict

        return children_layout, section_dict

    def _section_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a section, this will be a list of the dictionaries representing the
        questions in this section.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback.
        """
        answer_list = []
        for child1 in section_dict['children']:
            child_dict1 = section_dict['children'][child1]
            answer1 = self.getters[child_dict1['type']](child_dict1)
            answer_list += answer1

        return answer_list

    def _yes_no_loader(self, section_dict):
        """
        The loader function for a yes/no feedback element, which is based on
        a section of the feedback file, with type 'yes_no'
        - section_dict -- python dictionary: the specs for the yes/no GUI
                                             element, based on the config
                                             file
        return: tuple containing a bokeh gui widget representing this yes_no
                element and an updated dictionary file.`
        """

        min1 = int(section_dict['min'])
        max1 = int(section_dict['max'])
        display_list1 = ['{0:d}'.format(i1) for i1 in range(min1, max1+1)]
        question_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict[LABEL_QUESTION],
                                           height=WIDGET_HEIGHT,
                                           width=MULTI_COLUMN_WIDTH,
                                          )
        row_label = bokeh.models.widgets.Paragraph(text='No=1,Yes=10',
                                                   height=WIDGET_HEIGHT,
                                                   width=TEXT_LABEL_WIDTH,
                                                  )
        row_buttons1 = \
            bokeh.models.widgets.RadioButtonGroup(labels=display_list1,
                                                  active=0,
                                                  height=WIDGET_HEIGHT,
                                                  width=MULTI_COLUMN_WIDTH,
                                                 )
        row_layout1 = bokeh.layouts.Row(row_label, row_buttons1)
        yes_no_layout = bokeh.layouts.Column(question_txt, row_layout1)
        section_dict['widget'] = yes_no_layout
        section_dict['sub_widget'] = row_buttons1
        return yes_no_layout, section_dict

    def _yes_no_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a yes/no question, there is only one tuple representing the answer
        to the yes/no question.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback.
        """
        answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                   LABEL_QUESTION: section_dict[LABEL_QUESTION],
                   LABEL_CATEGORY: '',
                   LABEL_VALUE: str(section_dict['sub_widget'].active + 1)
                  }

        return [answer1,]


    def _perf_matrix_loader(self, section_dict):
        """
        The loader function for a performance matrix feedback element, which
        is based on a section of the feedback file, with type
        'performance_matrix'
        - section_dict -- python dictionary: the specs for the performance
                                             matrix  GUI element, based on the
                                             config file
        return: tuple containing a bokeh gui widget representing this performance
                matrix  GUI element and an updated dictionary file.`
        """
        min1 = int(section_dict['min'])
        max1 = int(section_dict['max'])
        display_list1 = ['{0:d}'.format(i1) for i1 in range(min1, max1+1)]
        question_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict[LABEL_QUESTION],
                                           height=WIDGET_HEIGHT,
                                           width=SINGLE_COLUMN_WIDTH,
                                          )
        row_list1 = [question_txt]
        row_layout_list1 = []
        category_list = section_dict['categories'].strip().split(',')
        category_list = [s1.strip() for s1 in category_list]
        section_dict['category_list'] = category_list
        buttons_list = []
        for cat1 in category_list:
            row_label = \
                bokeh.models.widgets.Paragraph(text=cat1,
                                               height=WIDGET_HEIGHT,
                                               width=TEXT_LABEL_WIDTH,
                                              )
            row_buttons1 = \
                bokeh.models.widgets.RadioButtonGroup(labels=display_list1,
                                                      active=0,
                                                      height=WIDGET_HEIGHT,
                                                      width=MULTI_COLUMN_WIDTH,
                                                     )
            row_layout1 = bokeh.layouts.Row(row_label, row_buttons1)
            row_list1 += [row_layout1]
            row_layout_list1 += [row_layout1]
            buttons_list += [row_buttons1]
        row_list1 += [bokeh.layouts.Spacer(width=600, height=30)]
        matrix_layout = bokeh.layouts.column(*row_list1)
        section_dict['widget'] = matrix_layout
        section_dict['category_widgets'] = row_list1
        section_dict['buttons'] = buttons_list

        return matrix_layout, section_dict

    def _perf_matrix_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a performance matrix question, there is one tuple per category
        specified. Each category will be present in the output as an
        independent answer.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback. There will be one tuple per category.
        """
        answer_list = []

        for buttons1, cat1 in zip(section_dict['buttons'],
                                  section_dict['category_list']):
            answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                       LABEL_QUESTION: section_dict[LABEL_QUESTION],
                       LABEL_CATEGORY: cat1,
                       LABEL_VALUE: str(buttons1.active + 1),
                      }
            answer_list += [answer1]
        return answer_list

    def _perfmatlab_loader(self, section_dict):
        """
        The loader function for a labelled performance matrix feedback element, which
        is based on a section of the feedback file, with type
        'performance_matrix'. See feedback_readme.md for more details.
        - section_dict -- python dictionary: the specs for the performance
                                             matrix  GUI element, based on the
                                             config file
        return: tuple containing a bokeh gui widget representing this performance
                matrix  GUI element and an updated dictionary object.`
        """
        labels1 = [s1.strip()
                   for s1 in section_dict['labels'].strip().split(',')
                   if s1]
        section_dict['labels_list'] = labels1
        question_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict[LABEL_QUESTION],
                                           height=WIDGET_HEIGHT,
                                           width=MULTI_COLUMN_WIDTH,
                                          )
        row_list1 = [question_txt]
        row_layout_list1 = []
        category_list = section_dict['categories'].strip().split(',')
        category_list = [s1.strip() for s1 in category_list]
        section_dict['category_list'] = category_list
        buttons_list = []
        for cat1 in category_list:
            row_label = \
                bokeh.models.widgets.Paragraph(text=cat1,
                                               height=WIDGET_HEIGHT,
                                               width=TEXT_LABEL_WIDTH,
                                              )
            row_buttons1 = \
                bokeh.models.widgets.RadioButtonGroup(labels=labels1,
                                                      active=0,
                                                      height=WIDGET_HEIGHT,
                                                      width=MULTI_COLUMN_WIDTH,
                                                     )
            row_layout1 = bokeh.layouts.Row(row_label, row_buttons1)
            row_list1 += [row_layout1]
            row_layout_list1 += [row_layout1]
            buttons_list += [row_buttons1]
        row_list1 += [bokeh.layouts.Spacer(width=600, height=30)]
        matrix_layout = bokeh.layouts.column(*row_list1)
        section_dict['widget'] = matrix_layout
        section_dict['category_widgets'] = row_list1
        section_dict['buttons'] = buttons_list

        return matrix_layout, section_dict


    def _perfmatlab_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a performance matrix question, there is one tuple per category
        specified. Each category will be present in the output as an
        independent answer.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback. There will be one tuple per category.
        """
        answer_list = []

        for buttons1, cat1 in zip(section_dict['buttons'],
                                  section_dict['category_list']):
            selected1 = section_dict['labels_list'][buttons1.active]
            answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                       LABEL_QUESTION: section_dict[LABEL_QUESTION],
                       LABEL_CATEGORY: cat1,
                       LABEL_VALUE: selected1,
                      }
            answer_list += [answer1]
        return answer_list

    def _text_input_loader(self, section_dict):
        """
        The loader function for a text input feedback element, which
        is based on a section of the feedback file, with type
        'text'. See feedback_readme.md for more details.
        - section_dict -- python dictionary: the specs for the text input
                                             GUI element, based on the config
                                             file
        return: tuple containing a bokeh gui widget representing this text
                input GUI element and an updated dictionary object.`
        """
        if section_dict['size'] == 'large':
            widget_height1 = DETAILED_TEXT_HEIGHT
            widget_width1 = DETAILED_TEXT_WIDTH
        elif section_dict['size'] == 'small':
            widget_height1 = WIDGET_HEIGHT
            widget_width1 = SINGLE_COLUMN_WIDTH
        else:
            raise AttributeError('Invalid value for size attribute')

        text_widget1 = \
            bokeh.models.widgets.TextInput(value='',
                                           title=section_dict[LABEL_QUESTION],
                                           height=widget_height1,
                                           width=widget_width1)
        section_dict['widget'] = text_widget1
        return text_widget1, section_dict

    def _text_input_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a text input question, there is only one tuple in the list.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback. There will be one tuple for a text input.
        """
        answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                   LABEL_QUESTION: section_dict[LABEL_QUESTION],
                   LABEL_CATEGORY: '',
                   LABEL_VALUE: section_dict['widget'].value,
                  }
        return [answer1,]


    def _selection_loader(self, section_dict):
        """
        The loader function for a selection feedback element, which
        is based on a section of the feedback file, with type
        'selection'. See feedback_readme.md for more details.
        - section_dict -- python dictionary: the specs for the selection
                                             GUI element, based on the config
                                             file
        return: tuple containing a bokeh gui widget representing this selection
                GUI element and an updated dictionary object.`
        """
        try:
            multiselect = section_dict['multi']
        except KeyError:
            multiselect = False

        question_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict[LABEL_QUESTION],
                                           height=WIDGET_HEIGHT,
                                           width=MULTI_COLUMN_WIDTH,
                                          )

        title_txt = \
            bokeh.models.widgets.Paragraph(text=section_dict[LABEL_TITLE],
                                           height=WIDGET_HEIGHT,
                                           width=TEXT_LABEL_WIDTH,
                                          )
        option_list = [(s1.strip(), s1.strip())
                       for s1 in section_dict['values'].split(',')]
        if multiselect:
            selection_widget = \
                bokeh.models.widgets.MultiSelect(options=option_list)
        else:
            selection_widget = \
                bokeh.models.widgets.Select(options=option_list)

        section_dict['selection_widget'] = selection_widget
        selection_layout = \
            bokeh.layouts.Column(question_txt,
                                 bokeh.layouts.Row(title_txt,
                                                   selection_widget))
        section_dict['widget'] = selection_layout
        return selection_layout, section_dict

    def _selection_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a selection question, there is only one tuple in the list.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback. There will be one tuple for a selection.
        """
        answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                   LABEL_QUESTION: section_dict[LABEL_QUESTION],
                   LABEL_CATEGORY: '',
                   LABEL_VALUE: section_dict['selection_widget'].value,
                  }
        return [answer1,]

    def _multiselect_loader(self, section_dict):
        """
        The loader function for a selection feedback element where the user
        can select multiple items, which is based on a section of the feedback
        file, with type 'multiselect'. See feedback_readme.md for more details.
        This function is a wrapper for _selection_loader, which just adds
        a "multi" key and item to the dictionary, so that a multiple selection
        widget is created.
        - section_dict -- python dictionary: the specs for the selection
                                             GUI element, based on the config
                                             file
        return: tuple containing a bokeh gui widget representing this selection
                GUI element and an updated dictionary object.`
        """
        section_dict['multi'] = True
        return self._selection_loader(section_dict)

    def _multiselect_getter(self, section_dict):
        """
        The function to retrieve the feedback entered by the user from
        the GUI elements, and compile the data into a list of answers. For
        a multiple selection question, there is only one tuple in the list,
        which has nultiple items in the answer element of the tuple.

        - section_dict -- python dictionary: the description of this question
                                             of the feedback form, including
                                             the GUI object to be user to
                                             retrieve the users input.
        return: A list of tuples, each tuple representing one element of user
                feedback. There will be one tuple for a selection.
        """
        selected_items1 = ','.join(section_dict['selection_widget'].value)
        answer1 = {LABEL_TAG: section_dict[LABEL_TAG],
                   LABEL_QUESTION: section_dict[LABEL_QUESTION],
                   LABEL_CATEGORY: '',
                   LABEL_VALUE: selected_items1
                  }
        return [answer1,]


    def get_feedback_widgets(self):
        """
        Get the main bokeh widget containing all the feedback widgets, for
        including in a bokeh document.
        return: A bokeh widget containing all the feedback elements.
        """
        return self.feedback_layout


    def compile_feedback(self):
        """
        This function traverses the tree of user feedback GUI elements created
        by _create_widgets, retrieves the user input, and compiles the input
        values into a list of answer tuples. The user feedback is then written
        out as a csv file, with one answer tuple per line of the csv file.
        The name of the user feedback file is based on the current time and the
        bokeh ID of this session and is written to the directory specified when
        this class was created.
        """
        self.answer_list = []
        for question_tag in self.feedback_dict:
            question_type = self.feedback_dict[question_tag]['type']
            self.answer_list += \
                self.getters[question_type](self.feedback_dict[question_tag])


        time_str_tpl = '{dt.year:04d}{dt.month:02d}{dt.day:02d}_' + \
                       '{dt.hour:02d}{dt.minute:02d}'
        time_str = time_str_tpl.format(dt=datetime.datetime.now())
        feedback_path = os.path.join(self.feedback_dir,
                                     'survey_{id}_{dt}.csv'.format(
                                         id=self.bokeh_id,
                                         dt=time_str))
        with open(feedback_path, 'w') as feedback_file1:
            writer1 = csv.DictWriter(feedback_file1, FEEDBACK_HEADER_ORDER)
            writer1.writeheader()
            for answer1 in self.answer_list:
                writer1.writerow(answer1)



    def _on_submit(self):
        """"
        Event handler for the user clicking on the submit button.
        """
        self.compile_feedback()
