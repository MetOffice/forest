
Write a driver
--------------

In a nutshell a driver is just a collection of code that loads and visualises
a particular data type, e.g. observation, model run, user feedback etc. To
play nicely with other components a driver must implement certain
interfaces, typically classes with specifically named methods. This guide
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

To allow a user to navigate a hypothetical dataset a ``dataset.navigate()``
method must return a class that implements the ``Navigator`` interface. Or put
simply an object that the menu system can use to populate itself. It looks
like this;

.. code-block:: python

   """Driver with a minimal Navigator"""

   class Dataset:
       ...
       def navigator(self):
           return Navigator()

   class Navigator:
       def variables(self, *args, **kwargs):
           return []

       def initial_times(self, *args, **kwargs):
           return []

       def valid_times(self, *args, **kwargs):
           return []

       def pressures(self, *args, **kwargs):
           return []

This interface is a little clunky and likely to change in the near future, but
for now it provides all of the information needed to populate the dropdown
navigation menus.

To add a visualisation to the map ``dataset.map_view()`` must be implemented.
This method returns an object that implements the ``MapView`` interface. Namely,
``render(state)`` and ``add_figure(figure)`` methods so that the visualisation
can update in response to application state changes and it can be registered
to each of the figures.
