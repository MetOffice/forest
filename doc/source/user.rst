

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


There is an conda-spec.txt file shipped with the repository or create
a fresh conda environment with bokeh and iris should be sufficient to
run the forest command.

.. code-block:: sh

   conda create -n forest python=3.6
   conda activate forest
   conda install -c conda-forge bokeh iris


You can check if you have a successful install by bringing up the
help message

.. code::

   forest -h


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
this will change in the very near future. An example of a command line
argument used to run a local implementation of FOREST looks like the following

.. code-block:: sh

   forest \
      --dev \
      --port 8080 \
      --allow-websocket-origin eld388:8080 \
      --database /path/to/file.db \
      --config /path/to/file.yaml \
      --directory /replacement/directory

.. warning:: While the intention is to support file(s) directly on the
             command line, at present this functionality is not supported
