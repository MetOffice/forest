"""Application wide services to de-couple components"""


class NullNavigator:
    """Navigator API"""
    def variables(self, pattern):
        return []

    def initial_times(self, pattern, variable):
        return []

    def valid_times(self, pattern, variable, initial_time):
        return []

    def pressures(self, pattern, variable, initial_time):
        return []


class NavigatorServiceLocator:
    """De-couples client-code from Navigator construction

    Configured at run-time by calling ``add_dataset(key, dataset)``
    """
    def __init__(self):
        self.datasets = {}

    def add_dataset(self, name, dataset):
        """Register Dataset(s) with service at run-time"""
        self.datasets[name] = dataset

    def get_navigator(self, dataset_name):
        """Find appropriate Navigator

        .. note:: If service unavailable a NullNavigator is returned to
                  prevent system crash
        """
        try:
            return self.datasets[dataset_name].navigator()
        except KeyError:
            return NullNavigator()


navigation = NavigatorServiceLocator()  # TODO: Find a better place to configure this
