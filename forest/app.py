"""Application"""
import bokeh.plotting


class Application:
    """Application structure"""
    def __init__(self):
        self.components = []

    def add_component(self, component):
        """Register component with application"""
        self.components.append(component)

    def connect(self, store):
        """Connect components to the store"""
        for component in self.components:
            if hasattr(component, "connect"):
                component.connect(store)

    @property
    def roots(self):
        """Generate component roots"""
        for component in self.components:
            if hasattr(component, "layout"):
                yield component.layout
