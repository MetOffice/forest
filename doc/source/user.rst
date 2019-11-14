

User guide
----------

Forest is a wrapper around a Python library called bokeh. In the future it
is intended to harness the full power of Tornado web server to allow user
specific settings, to capture survey results and to allow annotations but
for now it is only a light weight viewer.


Installation
~~~~~~~~~~~~

FOREST is available on conda-forge. To install it inside a conda
environment run the following command.

.. code-block:: sh

    conda install -c conda-forge


Basic usage
~~~~~~~~~~~

FOREST can be run as a command to quickly view files on disk.

.. code-block:: sh

   forest --show file.nc

Database
~~~~~~~~

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
      --config /path/to/file.yaml

Where the various ``bokeh serve`` options are passed straight through
to the underlying bokeh server.


Configuration file
~~~~~~~~~~~~~~~~~~

A configuration file is a convenient way to compare multiple
datasets spread across file systems and web based catalogues.
There is support for variable substitution of either
environment variables or through the command line ``--var KEY VALUE``
flag. Multiple ``--var`` flags can be specified to substitute
more than one variable.

.. code-block:: yaml
   :caption: some.yaml

   # some.yaml
   files:
     - label: UM
       pattern: ${HOME}/file.nc
     - label: RDT
       pattern: ${prefix}/file.json

Would be equivalent to the following file

.. code-block:: yaml
   :caption: some.yaml.processed

   files:
     - label: UM
       pattern: /Users/Bob/file.nc
     - label: RDT
       pattern: /some/dir/file.json


