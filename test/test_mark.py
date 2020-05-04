from unittest.mock import Mock, sentinel
from forest.observe import Observable
import forest.mark


@forest.mark.component
class FakeComponent(Observable):
    def on_change(self):
        self.notify(None)

    def render(self):
        self.on_change()  # Should be disabled during self.render()


@forest.mark.component
class FakeOneWayComponent(Observable):
    def on_change(self):
        self.notify(None)


def test_component_render():
    """component.notify() forbidden during render"""
    component = FakeComponent()
    component.notify = Mock()
    component.render()
    assert not component.notify.called


def test_component_on_change():
    component = FakeComponent()
    component.notify = Mock()
    component.on_change()
    assert component.notify.called


def test_component_given_one_way_on_change():
    component = FakeOneWayComponent()
    component.notify = Mock()
    component.on_change()
    assert component.notify.called
