

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

    conda install -c conda-forge forest=${VERSION}

.. note:: An explicit version is required to prevent earlier cross-platform versions
          being prioritised


Basic usage
~~~~~~~~~~~

FOREST can be run as a command to quickly view files on disk.

.. code-block:: sh

   forest view --driver ${DRIVER} file.nc

For convenience the view command opens a browser tab to display the
file.

To disable the automatic browser tab, e.g. when running the command
repeatedly or starting the server in a browserless environment use the
``--no-open-tab`` flag.

.. code-block:: sh

   forest view --no-open-tab --driver ${DRIVER} file.nc

Server settings
===============

FOREST uses `Bokeh <https://bokeh.org/>`_ which in turn uses a `Tornado <https://www.tornadoweb.org>`_ server. To
run two or more servers or indeed if the default port `5006` is in use it is possible to specify a unique
port number.

.. code-block:: sh

   forest view --port 5050 --driver ${DRIVER} file.nc

Other server settings that are exposed by Bokeh but not by FOREST can be accessed by re-running the equivalent
bokeh serve command. For convenience, the bokeh command is printed by FOREST when launching Bokeh.

.. code-block:: bash

   > # Example bokeh command print
   > forest view --driver fake settings
   Launching Bokeh...
   bokeh serve /path/to/forest --show --args --file-type fake settings

Then simply use the above ``bokeh serve`` command to further tweak settings. If there is a common
setting that is repeatedly being overwritten perhaps it would be worth a pull request.

Configuration file
~~~~~~~~~~~~~~~~~~

A configuration file is a convenient way to compare multiple
datasets spread across file systems and web based catalogues.


Edition 2022
============

Recent versions of FOREST have incorporated a more flexible structure
to configure drivers. To be able to let FOREST know which syntax you
are using the top-level ``edition`` key should be set to **2022**.

.. code-block:: yaml

   edition: 2022
   datasets:
     - label: Example
       driver:
          name: gridded_forecast
          settings:
            path: example.pp

To generate a template ``forest.config.yaml`` run ``forest init``.

.. code-block:: sh

   forest init

The skeleton configuration file can then be edited to point at your
data. When you are ready to launch the server process and see
your data run the ``ctl`` command, short for control.

.. code-block:: sh

   forest ctl forest.config.yaml



Edition 2018
============

There is support for variable substitution of either
environment variables or through the command line ``--var KEY VALUE``
flag. Multiple ``--var`` flags can be specified to substitute
more than one variable.

.. code-block:: yaml
   :caption: example.yaml

   files:
     - label: UM
       pattern: ${HOME}/file.nc
     - label: RDT
       pattern: ${prefix}/file.json

Would be equivalent to the following file

.. code-block:: yaml
   :caption: example.yaml.processed

   files:
     - label: UM
       pattern: /Users/Bob/file.nc
     - label: RDT
       pattern: /some/dir/file.json

.. seealso:: :mod:`forest.config` for the latest config file syntax


