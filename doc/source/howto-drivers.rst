
Write a driver
--------------

In a nutshell a driver is just a collection of code that loads and visualises
a particular data type, e.g. observation, model run, user feedback etc. To
play nicely with other components a driver must implement certain
interfaces, a class with specifically named methods. This guide
walks through the interfaces and provides a rough guide on how best to
structure your code.

At present all drivers are modules placed in ``forest/drivers`` directory and
must at least implement a ``Dataset`` class.

A minimal driver would be the following;

.. code-block:: python

   """Hypothetical module forest/drivers/minimal.py"""

   class Dataset:
       def __init__(self, **kwargs):
           pass

To invoke this hypothetical driver from the command line a user would
either specify it as ``--file-type minimal`` or place it in a config
file.

.. code-block:: yaml

   files:
     - label: Minimal driver
       file_type: minimal

But this driver is not very useful, it doesn't speak to data, allow a user
to navigate or represent anything on the map.

To allow a user to navigate a dataset a ``dataset.navigator()``
method that returns a ``Navigator`` instance. A Navigator has four methods
``valid_times``, ``initial_times``, ``variables`` and ``pressures``, each
of which provides data for the menu system.

.. code-block:: python

   """Driver with a minimal Navigator"""

   class Dataset:
       ...
       def navigator(self):
           return Navigator()

   class Navigator:
       def variables(self, *args, **kwargs):
           return ["relative_humidity"]

       def initial_times(self, *args, **kwargs):
           return [datetime(2020, 1, 1)]

       def valid_times(self, *args, **kwargs):
           return [datetime(2020, 1, 1), datetime(2020, 1, 1, 3), ...]

       def pressures(self, *args, **kwargs):
           return [1000, 750, ...]

This interface is a little clunky and likely to change in the near future, but
for now it provides all of the information needed to populate the dropdown
navigation menus.

To add a visualisation to the map the ``map_view()`` must return an instance
that implements the ``MapView`` interface.

.. code-block:: python

   class Dataset:
       ...
       def map_view(self):
           return MapView()

Where a typical map view must support adding itself to a figure and
updating when the application state changes. It does this by implementing
``add_figure(figure)`` and ``render(state)`` methods. For example to
plot circles on a map.

.. code-block:: python

   class MapView:
       def __init__(self):
           self.source = bokeh.models.ColumnDataSource({
               "x": [],
               "y": []
           })

       def add_figure(self, figure):
           return figure.circle(x="x", y="y", source=self.source)

       def render(state):
           self.source.data = {
                "x": [1, 2, 3],
                "y": [1, 2, 3],
           }


Thus concludes our walk through implementing a driver. It's not a perfect
design but hopefully it is enough to get started.


