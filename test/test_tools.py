from collections import defaultdict
import pytest
import unittest
import unittest.mock
import bokeh
from forest import tools, db, redux

def test_tool_toggle_reducer():
    state = tools.reducer({}, tools.on_toggle_tool("time_series", True))
    assert state == {"tools": {"time_series": True}}

def test_tool_toggle_reducer_immutable_state():
    state = {"tools": {"time_series": True}}
    next_state = tools.reducer(
        state,
        tools.on_toggle_tool("time_series", False)
        )
    assert state == {"tools": {"time_series": True}}
    assert next_state == {"tools": {"time_series": False}}

@pytest.mark.parametrize("actions,expect", [
    ([], {}),
    ([tools.on_toggle_tool("profile", False)],
     {"tools": {"profile": False}}),
    ([db.set_value("key", "value")], {"key": "value"}),
    ([
        tools.on_toggle_tool("time_series", False),
        db.set_value("key", "value")], {
            "key": "value",
            "tools": {"time_series": False}}),
])
def test_combine_reducers(actions, expect):
    reducer = redux.combine_reducers(tools.reducer, db.reducer)
    state = {}
    for action in actions:
        state = reducer(state, action)
    assert state == expect

def test_tools_on_toggle_tool_action():
    action = tools.on_toggle_tool("time_series", False)
    assert action == {"kind": tools.ON_TOGGLE_TOOL, 
                      "tool_name": "time_series", "value": False}

def test_tools_panel_on_toggle_emits_action():
    listener = unittest.mock.Mock()
    tools_panel = tools.ToolsPanel({"time_series": "wheeeyyyy"})
    tools_panel.add_subscriber(listener)
    tools_panel.on_click("time_series")(True)
    listener.assert_called_once_with(tools.on_toggle_tool("time_series", True))
