# FOREST meteorology visualisation tool user feedback confoguration

The forest tool displays data from high-resolution weather simulations
and observations, as well as providing an opportunity for perational
meteorologists to give feedback on the usefulness of the simulations. The
feedback is entered through a GUI form. The form consists of bokeh GUI widgets
which are specified through a .conf configuration, which interprets the
elements of the file and translates them into a GUI for obtaining feedback
from users of forest.

This file describes how create user feedback GUI elements by specifying
elements of the .conf file. Each section of the configuration file is a GUI
element.

The user will fill in the form through the relevant visualisation page. When
the user clicks on the submit button, the feedback infrastructure will collect
the values selected and write the feedback values to .csv file, with
one row per GUI element.

## Format Overview
The feedback questions are specified in configuration files with the .conf
extension using the INI file format described here:
https://en.wikipedia.org/wiki/INI_file

Each question is a section in the file, with the type of the question
specified by the type element. All other elements are present or absent based
on the type. The available question type are described in more detail in
the type format section.

## Type format

Each type of question causes a different widget to be displayed. The different
types of question are described in this section.

### Section
The section question type (type=section) allows nesting of files, so that user
feedback question can be split into multiple files. The sections are:

  * type=section
  * file - The path to the file containing more user feedback questions.
  * title - The title of the section to be included.
  * description - The description of th section to be included.

### Text
The text question type creates a basic text input GUI element, and allows for
unconstrained text feedback.

  * type=text
  * question - The text of the question being asked.
  * size - Can be small or large, depending on how much space for feedback is
           required. small displays a single line widget for items like a name,
           while large displays a wider mutliline widget for entering a
           paragraph of feedback.


### Selection
The selection question type prompts users to select one of a list of options.

  * type=selection
  * question - The description of what the user is selecting a feedback option
               for.
  * title - The title of what is being selected.
  * values - The range of values that can be selected from, of which exactly
             one must be selected.


### Mutliple selection
The multiple selection question type prompts users to select one or more
options from of a list of options.

  * type=multiselect
  * question - The description of what the user is selecting a feedback option
               for.
  * title - The title of what is being selected.
  * values - The range of values that can be selected from, of which one or
             more must be selected.

### Performance Matrix

The performance matrix option present the user with a grid of buttons. Each
row represents a category to be rated. The rating for the category is a number
from min to max inclusive. For example, the question might be what is the
performance of a particular model, and each row, specific by the list of
categories, isaspects of model performance like convective precipitation or
wind gusts. The user can rate performance of that aspect of that model from
1 to 10.

  * type=performance_matrix
  * min - The minimum rating for a category
  * max - The maximum rating for a category
  * categories - The list of categories to be rated for the question.
  * question - The description of what the user is rating the performance of.

### Labelled Performance Matrix
The labelled performance matrix option present the user with a grid of
buttons. Each row represents a category to be rated. For the labelled
performance matrix, each catergory is not a numerical rating, but one of a list
of properties to be assigned to the category. For example, the question might
be how confident are you in the perfoamnce of each of the following models, and
the categories might be low confidence, moderate confidence, not enought info,
did not use this model.

  * type=performance_matrix_labelled
  * question - The description of what the user is rating the performance of.
  * categories - The list of categories to be rated for the question.
  * labels - The labels for the performance in each category.

### Yes/No
The yes/no option asks the user to agree or disagree with a statement, with the
min value representing total disagreement and the max representing total
agreement. For example, the question might be "Did the Met Office Global UM
help to highlight key areas of HIW yesterday?", where an answer of 1 mean the
UM did not help at all, and an answer of 10 means it was very helpful.

  * type=yes_no
  * min - The minimum value, representing an answer of no and complete
          disagreement.
  * max - The maximum value, representing an answer of yes and complete
          agreement.
  * question - The question stating what the user is agreeing or disagreeing
               with.


