
Developer guide
===============

Contributing to FOREST should be simple. Building, testing and documenting
the code should be straight forward.

Introduction
------------

To contribute to FOREST development first take a look at
the existing issues_ list available at the github_ repository. Alternatively
open a new issue that captures the feature/bug/enhancement you would like
to make.

Issues have recently been organised into projects_ to make it clear to new
and existing developers what tasks are currently being tackled and which
we should think about tackling next.


.. _github: https://github.com/informatics-lab/forest
.. _issues: https://github.com/informatics-lab/forest/issues
.. _projects: https://github.com/informatics-lab/forest/projects


While the ambition is for FOREST to be distributed via conda, e.g.
using ``conda install -c conda-forge forest``, to develop the code
a clone of the github_ repository is needed.

.. code::

   git clone git@github.com:informatics-lab/forest.git

The next step is to create a branch related to your issue that can be
reviewed and merged back to master after your developments are complete.

.. code::

   git co -b $NAME


The repository ships with a setup.py that should be suitable to
use inside a virtual environment, either venv or conda.

The test suite is run using the Python standard library unittest module.

.. code::

   python -m unittest discover

