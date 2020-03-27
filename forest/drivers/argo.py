import unittest.mock

class Dataset:
    def __init__(self, **kwargs):
        pass

    def navigator(self):
        return Navigator()

    def map_view(self):
        return View()

    def profile_view(self, figure):
        return ProfileView()


class Navigator:
    def initial_times(self, *args, **kwargs):
        return []

    def valid_times(self, *args, **kwargs):
        return []

    def variables(self, *args, **kwargs):
        return []

    def pressures(self, *args, **kwargs):
        return []


class View:
    def add_figure(self, *args, **kwargs):
        return unittest.mock.Mock()

    def render(self, *args, **kwargs):
        pass


class ProfileView:
    def connect(self, store):
        print("Hello, World!")
