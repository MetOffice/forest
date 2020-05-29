import forest.services


def test_navigation_service():
    """experimental API"""
    dataset_name = "undefined"
    navigator = forest.services.navigation.get_navigator(dataset_name)
    navigator.valid_times()
