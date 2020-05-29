"""Application wide services to de-couple components"""


class NullNavigator:
    """Navigator API"""
    def valid_times(self):
        pass


class NavigatorLocator:
    def get_navigator(self, dataset_name):
        """Find appropriate Navigator"""
        return NullNavigator()


navigation = NavigatorLocator()  # TODO: Find a better place to configure this
