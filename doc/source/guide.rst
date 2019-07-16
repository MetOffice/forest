
Developer guide
===============

Contributing to FOREST should be simple. Building, testing and documenting
the code should be straight forward.

Github
------

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


Installation
------------

While the ambition is for general users to be able to use widely available
build tools to install working versions of FOREST, e.g.
`conda install -c conda-forge forest`. The recommended approach is to
clone the repository.

Development
-----------

The FOREST repository ships with a setup.py that should be suitable to
use inside a virtual environment, either venv or conda. The test suite
is run using the Python standard library unittest module.

.. code::

   python -m unittest discover

