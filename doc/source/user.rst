

User guide
----------

Forest is a wrapper around a Python library called bokeh. In the future it
is intended to harness the full power of Tornado web server to allow user
specific settings, to capture survey results and to allow annotations but
for now it is only a light weight viewer.


Installation
~~~~~~~~~~~~

The first step is to clone the GitHub repository, then to install
either inside a virtual or conda environment with the setup.py script

.. code-block:: sh

   git clone git@github.com:informatics-lab/forest.git


There is a conda-spec.txt file shipped with the repository or create
a fresh conda environment with bokeh and iris should be sufficient to
run the forest command.

.. code-block:: sh

   conda create -n forest python=3.6
   conda activate forest
   conda install -c conda-forge bokeh iris

Then once the dependencies are available inside your virtual environment
it is possible to install forest in development mode.

.. code-block:: sh

   cd /path/to/repo/
   python setup.py develop


You can check if you have a successful install by bringing up the
help message

.. code::

   forest -h

.. note:: Future releases will simply use ``conda install -c conda-forge forest``

Basic usage
~~~~~~~~~~~

A typical approach to running forest is to craft a configuration file
that lets forest know where each dataset lives on the file system.

.. code-block:: yaml

   models:
           - name: Operational GA6
             pattern: '*global_africa*.nc'
           - name: Operational Tropical Africa
             pattern: '*os42_ea*.nc'

At present the ``--database`` flag and ``--config`` flags are mandatory,
this will change in the very near future.

To construct a database run the ``forestdb --database file.db *.nc`` command
on the files you intend to navigate through. This is a highly unnecessary step
and will be removed in the very near future. It was introduced to optimise
communication between cloud computing services.

An example to run a local implementation of FOREST looks like the
following:

.. code-block:: sh

   forest \
      --dev \
      --port 8080 \
      --allow-websocket-origin eld388:8080 \
      --database /path/to/file.db \
      --config /path/to/file.yaml \
      --directory /replacement/directory

Where the various ``bokeh serve`` options are passed straight through
to the underlying bokeh server. FOREST specific options will be refined
in the coming releases.

.. warning:: While the intention is to support file(s) directly on the
             command line, at present this functionality is not supported
