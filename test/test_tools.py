from collections import defaultdict
import pytest
import unittest
import unittest.mock
import bokeh
from forest import tools, db, redux

def test_tool_toggle_reducer():
    state = tools.reducer({}, tools.on_toggle_tool("toggle_time_series", True))
    assert state == {"tools": {"toggle_time_series": True}}

def test_tool_toggle_reducer_immutable_state():
    state = {"tools": {"toggle_time_series": True}}
    next_state = tools.reducer(
        state, 
        tools.on_toggle_tool("toggle_time_series", False)
        )
    assert state == {"tools": {"toggle_time_series": True}}
    assert next_state == {"tools": {"toggle_time_series": False}}

@pytest.mark.parametrize("actions,expect", [
    ([], {}),
    ([tools.on_toggle_tool("toggle_profile", False)], 
     {"tools": {"toggle_profile": False}}),
    ([db.set_value("key", "value")], {"key": "value"}),
    ([
        tools.on_toggle_tool("toggle_time_series", False),
        db.set_value("key", "value")], {
            "key": "value",
            "tools": {"toggle_time_series": False}}),
])
def test_combine_reducers(actions, expect):
    reducer = redux.combine_reducers(tools.reducer, db.reducer)
    state = {}
    for action in actions:
        state = reducer(state, action)
    assert state == expect

def test_tools_on_toggle_tool_action():
    action = tools.on_toggle_tool("toggle_time_series", False)
    assert action == {"kind": tools.ON_TOGGLE_TOOL, 
                      "tool_name": "toggle_time_series", "value": False}

def test_tools_panel_on_toggling_tool_emits_action():
    listener = unittest.mock.Mock()
    features = defaultdict(lambda: False)
    features["time_series"] = True
    tools_panel = tools.ToolsPanel(features)
    tools_panel.add_subscriber(listener)
    #For reasons I don't understand, this doesn't trigger a toggle event
    tools_panel.buttons["toggle_time_series"].active = True
    listener.assert_called_once_with(
        tools.on_toggle_tool("toggle_time_series", False)
    )

