
Write a driver
--------------

In a nutshell a driver is a collection of code that loads and visualises
a particular data type, e.g. observation, model forecasts, user feedback etc. To
play nicely with other components a driver must implement certain
interfaces, a.k.a classes with specifically named methods. This guide
walks through the interfaces and provides a rough guide on how best to
structure your code.

At present drivers are Python modules placed in ``forest/drivers`` directory
that implement a ``Dataset`` class. We'll see later how a ``Dataset`` can
be used to navigate and visualise its data.

A minimal driver would be the following;

.. code-block:: python

   """Hypothetical module forest/drivers/minimal.py"""

   class Dataset:
       def __init__(self, **kwargs):
           pass

To invoke this driver a user would either specify it as ``--file-type minimal``
on the command line or place it in a config file.

.. code-block:: yaml

   files:
     - label: Minimal driver
       file_type: minimal

But this driver is not very useful, it doesn't speak to data, support
navigation or add anything to the map. Let's fix it.

To allow a user to navigate a dataset we need a ``navigator()``
method that returns a ``Navigator`` instance. You may be wondering what
the Navigator interface is, any class that has four methods
``valid_times()``, ``initial_times()``, ``variables()`` and ``pressures()``
is a navigator. At the moment these are the four basic dimensions, in
future a navigator may be able to define the dimensionality of its underlying
data or even custom user interfaces to explore it.

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

This interface is a little clunky, but for now it provides enough information
to populate dropdown menus.

To add a visualisation to the map the ``map_view()`` is used to return an instance
that implements the ``MapView`` interface.

.. code-block:: python

   class Dataset:
       ...
       def map_view(self):
           return MapView()

Again, a map view is an interface that supports adding glyphs to a figure and
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

While it is nice to plot circles on a map, having access to general purpose
scripting and a full application state is the real benefit of defining
a MapView. The possibilites at this point are endless.

Thus concludes our walk through implementing a driver. It's not a perfect
design but hopefully it is enough to get started.


