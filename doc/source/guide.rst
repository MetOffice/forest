
Developer guide
===============

Contributing to FOREST should be simple. Building, testing and documenting
the code should be straight forward.

To begin to develop FOREST first take a look at
the existing issues_ list available at the github_ repository. If a
suitable issue exists that you would like to work on, go ahead and
assign yourself to it. Alternatively, if there are no issues that address
your need open a new issue that captures the feature/bug/enhancement you would
like to make with enough information to describe the problem at hand. This
information will make it easy for other developers to contribute, review
or even see if similar functionality is being worked on elsewhere.

Issues are organised into projects_ to make it clear to new
and existing developers what tasks are currently being tackled and which
we should think about tackling next. Projects are a good way to manage
the many potential features of FOREST so that nothing falls through the cracks.

While the aim is to distribute FOREST to users via conda, e.g.
using ``conda install -c conda-forge forest``, developers need the source code.
Clone the github_ repository into a sensible location on your machine to begin
work.

.. code::

   git clone git@github.com:informatics-lab/forest.git

Next create a branch related to your issue. Your branch will
be reviewed and merged back to master once your developments
are complete and the reviewer is happy with your improvements.

.. code::

   git co -b name_of_branch

The above command checks out a local branch called **name_of_branch**. Please
specify a good name for the branch which is not ``master`` or ``develop``.
To make this branch appear in the list of branches on the GitHub repository,
attempt to run ``git push``. This will probably fail but should include the
latest way to set an upstream branch in the error message. For example,

.. code::

   git co --set-upstream origin name_of_branch

The repository ships with a setup.py that should be suitable to
use inside a virtual environment, either venv or conda.

The test suite is run using the Python standard library unittest module.

.. code::

   python -m unittest discover

Remember to check code in regularly, the granularity of commits is not
extremely important but it may help the reviewer to understand the motivation
behind each change. Also, push changes to make sure the GitHub copy of your
branch is up to date prior to review or during discussion with other developers.



.. _github: https://github.com/informatics-lab/forest
.. _issues: https://github.com/informatics-lab/forest/issues
.. _projects: https://github.com/informatics-lab/forest/projects
