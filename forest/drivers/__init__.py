"""
Drivers
-------

To add a new file format or custom visualisation one or more
of the following interfaces need to be implemented. Each
tries to solve a separate part of the data visualisation
puzzle.

Navigator
~~~~~~~~~

The navigator is responsible for populating the
controls. It encapsulates the information needed
to navigate your data source.

View
~~~~

Views are responsible for adding glyphs to figures and
updating their underlying sources. They react to application
state changes through ``View.render(state)``

Loader
~~~~~~

Loaders separate the I/O layer from the Bokeh layer and
are most useful for implementing per-server caching. A
single loader can be shared across Bokeh documents.

They are controlled by their `View` but allow for a single
view to ingest data from multiple data sources.

.. automodule:: forest.drivers.example
    :members:

"""
from forest.drivers.util import by_name
