
Developer guide
===============

Contributing to FOREST should be simple. Building, testing and documenting
the code should be as easy as falling off a log, pardon the pun.

To get ideas on how to contribute, first take a look at
the existing issues_ list. If an
issue exists that you would like to work on, go ahead and
assign yourself to it or drop a comment so that one of our
developers can get in touch. 

Alternatively, if there are no issues that address
your need open a new issue that describes the feature or bug you
are thinking about. A well written issue will make it easy for other
developers to help you, review your changes or even point out similar
functionality that may already be being worked on elsewhere.

The issue list is organised into projects_. Projects help us
prioritise which issues should to be tackled next. It's a good
idea to take a peak at the currently active project to see
what is considered urgent and what is deemed less of a priority.

Contributing guidelines
-----------------------

We recommend forking the repository and submitting a pull request.

1. Fork the repo
2. Create a branch, named with your feature/bug
3. Write code
4. Commit and push changes
5. Submit pull request

One of our developers will pick it up and after a friendly review will merge it into
``master`` ready for the next release.

Getting set up
--------------

Clone the github_ repository into a sensible location on your machine to begin
work.


.. code::

   git clone git@github.com:informatics-lab/forest.git
   

.. note::

   If you don't have permissions to work directly on the repository, please fork
   it under your own GitHub user account.


Branch
------

The next thing to do is to create a branch related to your issue. To do this, navigate to
the top level directory of your cloned repository. The
name of the branch should be related to the issue or bug you are
solving.

.. code::

   git checkout -b name_of_branch

To make this branch appear in the list of branches on the GitHub repository
run ``git push``. This will fail but should tell you the best way
to set an upstream branch. For example,

.. code::

   git push --set-upstream origin name_of_branch
   

Development environment
-----------------------

It is recommended to develop FOREST in a virtual environment of some kind,
we recommend miniconda_ since it's fairly light weight. To make sure
all of the various dependencies and development tools are installed
in your environment run the following command with your chosen
environment activated.

.. code::

   conda install -c conda-forge --file requirements.txt --file requirements-dev.txt

The repository ships with
a ``setup.py`` that should be suitable to
use inside a virtual environment, either ``venv`` or ``conda``. Once
you've created and activated an environment it should be possible
to set the code base to development mode.

.. code::

   python setup.py develop
   
Of course, if you develop Python projects using an IDE or if you can't use virtual environments,
go ahead and use the tools that are most comfortable to you.

Running tests
-------------

The test suite uses pytest_, to run all of the tests navigate to the root directory
of your working copy and run the ``pytest`` command. This should produce
a lot of output but hopefully should indicate to you that all tests
have passed.

.. code::

   pytest -q

To supress noisy pytest messages the ``-q`` flag is useful.

Advice
------

Remember to commit and push code regularly, the granularity of commits is a personal
preference but we recommend small, orthogonal changes to ease the burden
on our developers. We also appreciate documentation that describes your
changes and tests that cover your code will go a long way to making
sure we don't accidentally deprecate your feature.


.. _github: https://github.com/informatics-lab/forest
.. _issues: https://github.com/informatics-lab/forest/issues
.. _projects: https://github.com/informatics-lab/forest/projects
.. _pytest: https://docs.pytest.org/en/latest
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
