"""Example driver definition

This example module can be used as a template to add
support for new data formats

"""


class Dataset:
    def __init__(self, label):
        self.label = label

    def navigator():
        return Navigator()

    def map_view(self):
        return View()

    def map_loader(self):
        return Loader()


class Navigator:
    """Dimensional data needed to navigate data

    The application uses values returned by the
    Navigator to populate dropdowns, buttons and various user
    interface widgets
    """
    def render(self, state):
        """Application state is passed through here"""
        raise NotImplementedError()

    def add_figure(self, figure):
        """Add Figure set up here, e.g. HoverTools, GlyphRenderers etc."""
        raise NotImplementedError()


class View:
    """Contract between external application and data specific visualisation"""
    def __init__(self, loader):
        self.loader = loader

    def render(self, state):
        """Application state is passed through here"""
        self.loader.example_method()

    def add_figure(self, figure):
        """Add Figure set up here, e.g. HoverTools, GlyphRenderers etc."""
        raise NotImplementedError()


class Loader:
    """Called by the View

    The View defines the interface that the Loader should implement
    """
    def example_method(self):
        raise NotImplementedError()
