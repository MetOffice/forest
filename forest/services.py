"""
Services
========

Application wide services to de-couple components. For example, components
need access to navigation to populate dropdown menus and widgets.


Navigation
----------

Datasets dimensions are often represented by large arrays, instead of storing the
arrays in the ``forest.redux.Store`` a service is provided to load the arrays
from key pieces of information. The key information is stored in the application
state to allow reproduction at a later date.

.. autoclass:: NavigatorServiceLocator
   :members:

.. autoclass:: NullNavigator
   :members:

"""


class NullNavigator:
    """Empty container to allow client-code to work if service not found"""
    def variables(self, pattern):
        """
        :returns: empty list
        """
        return []

    def initial_times(self, pattern, variable):
        """
        :returns: empty list
        """
        return []

    def valid_times(self, pattern, variable, initial_time):
        """
        :returns: empty list
        """
        return []

    def pressures(self, pattern, variable, initial_time):
        """
        :returns: empty list
        """
        return []


class NavigatorServiceLocator:
    """De-couples client-code from Navigator construction

    .. note:: This service locator is accessed as ``forest.services.navigation``
              module level constant

    Configured at run-time by calling ``add_dataset(key, dataset)``

    >>> forest.services.navigation.add_dataset("name", dataset)

    And used by views and components as follows

    >>> navigator = forest.services.navigation.get_navigator("name")

    """
    def __init__(self):
        self.datasets = {}

    def add_dataset(self, name, dataset):
        """Register Dataset(s) with service at run-time

        :param name: key used to locate navigator service
        :param dataset: ``forest.drivers.Dataset`` instance
        """
        self.datasets[name] = dataset

    def get_navigator(self, dataset_name):
        """Find appropriate Navigator

        .. note:: If service unavailable a NullNavigator is returned to
                  prevent system crash

        :param name: key used to locate navigator service
        :returns: ``forest.drivers.Dataset`` instance
        """
        try:
            return self.datasets[dataset_name].navigator()
        except KeyError:
            return NullNavigator()


navigation = NavigatorServiceLocator()  # TODO: Find a better place to configure this
