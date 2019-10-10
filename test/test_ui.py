import datetime as dt
import numpy as np
import forest
from forest import ui
import bokeh.models


def test_forecast_user_interface():
    """Should hide initial_time and variable rows"""
    state = {
        "selected": {
            "file_type": "unified_model"
        },
        "dimensions": {
            "rdt": [
                "valid_time"
            ],
            "unified_model": [
                "variable",
                "initial_time",
                "valid_time",
                "pressure"
            ]
        }
    }
    controls = forest.ui.Controls()
    controls.render(state)
    rows = controls.layout.children
    div, = rows[0].children
    assert isinstance(div, bokeh.models.Div)
    assert div.text == "Navigation:"
    for i, labels in enumerate([
            ("Previous", "Variable", "Next"),
            ("Previous", "Initial time", "Next"),
            ("Previous", "Valid time", "Next"),
            ("Previous", "Pressure", "Next"),
        ]):
        left_btn, dropdown, right_btn = rows[i + 1].children
        assert isinstance(left_btn, bokeh.models.Button)
        assert isinstance(dropdown, bokeh.models.Dropdown)
        assert isinstance(right_btn, bokeh.models.Button)
        assert labels == (left_btn.label, dropdown.label, right_btn.label)


def test_observation_user_interface():
    """Should hide initial_time and variable rows"""
    state = {
        "selected": {
            "file_type": "rdt"
        },
        "dimensions": {
            "rdt": [
                "valid_time"
            ],
            "unified_model": [
                "variable",
                "initial_time",
                "valid_time",
                "pressure"
            ]
        }
    }
    controls = forest.ui.Controls()
    controls.render(state)
    rows = controls.layout.children
    div, = rows[0].children
    left_btn, dropdown, right_btn = rows[1].children
    assert isinstance(div, bokeh.models.Div)
    assert div.text == "Navigation:"
    assert isinstance(left_btn, bokeh.models.Button)
    assert isinstance(dropdown, bokeh.models.Dropdown)
    assert isinstance(right_btn, bokeh.models.Button)
    assert left_btn.label == "Previous"
    assert dropdown.label == "Valid time"
    assert right_btn.label == "Next"


def test_search_dimensions_related_to_label():
    state = {}
    actions = [
        ui.set_dimensions(
            "RDT", ["valid_time"]),
        ui.set_dimensions(
            "UM", ["variables", "initial_time", "valid_time", "pressure"]),
    ]
    state = {}
    for action in actions:
        state = ui.reducer(state, action)
    result = ui.find_dimensions(state, "RDT")
    expect = ["valid_time"]
    assert expect == result


def test_reducer_set_dimensions():
    actions = [
        ui.set_dimensions(
            "RDT", ["valid_time"]),
        ui.set_dimensions(
            "UM", ["variables", "initial_time", "valid_time", "pressure"]),
    ]
    state = {}
    for action in actions:
        state = ui.reducer(state, action)
    result = state
    expect = {
        "groups": {
            0: {
                "label": "RDT",
                "dimensions": 0
            },
            1: {
                "label": "UM",
                "dimensions": 1
            }
        },
        "dimensions": {
            0: ["valid_time"],
            1: ["variables", "initial_time", "valid_time", "pressure"]
        }
    }
    assert expect == result


def test_insert_same_label_twice():
    actions = [
        ui.set_dimensions(
            "RDT", ["valid_time"]),
        ui.set_dimensions(
            "RDT", ["variables", "initial_time", "valid_time", "pressure"]),
    ]
    state = {}
    for action in actions:
        state = ui.reducer(state, action)
    result = state
    expect = {
        "groups": {
            0: {
                "label": "RDT",
                "dimensions": 1
            }
        },
        "dimensions": {
            0: ["valid_time"],
            1: ["variables", "initial_time", "valid_time", "pressure"]
        }
    }
    assert expect == result


def test_insert_same_dimensions_twice():
    actions = [
        ui.set_dimensions(
            "RDT", ["valid_time"]),
        ui.set_dimensions(
            "EIDA50", ["valid_time"]),
    ]
    state = {}
    for action in actions:
        state = ui.reducer(state, action)
    result = state
    expect = {
        "groups": {
            0: {
                "label": "RDT",
                "dimensions": 0
            },
            1: {
                "label": "EIDA50",
                "dimensions": 0
            }
        },
        "dimensions": {
            0: ["valid_time"],
        }
    }
    assert expect == result


def test_set_selected():
    action = ui.set_selected("Label")
    result = ui.reducer({}, action)
    expect = {
        "selected": 0,
        "groups": {
            0: {"label": "Label"}
        }
    }
    assert expect == result


def test_set_selected_twice():
    state = {}
    for action in [
            ui.set_selected("A"),
            ui.set_selected("B")]:
        state = ui.reducer(state, action)
    result = state
    expect = {
        "selected": 1,
        "groups": {
            0: {"label": "A"},
            1: {"label": "B"}
        }
    }
    assert expect == result


def test_set_time_coordinate():
    times = [dt.datetime(2019, 1, 1)]
    action = ui.set_coordinate("RDT", "valid_time", times)
    state = {}
    result = ui.reducer(state, action)
    expect = {
        "groups": {
            0: {
                "label": "RDT",
                "coordinates": {
                    "valid_time": ["2019-01-01 00:00:00"]
                }
            }
        }
    }
    assert expect == result


def test_set_multiple_coordinates():
    times = [np.datetime64("2019-01-01T00:00:00", "s")]
    pressures = [1000.0001]
    state = {}
    for action in [
            ui.set_coordinate("RDT", "valid_time", times),
            ui.set_coordinate("UM", "valid_time", times),
            ui.set_coordinate("UM", "pressures", pressures)]:
        state = ui.reducer(state, action)
    result = state
    expect = {
        "groups": {
            0: {
                "label": "RDT",
                "coordinates": {
                    "valid_time": ["2019-01-01 00:00:00"]
                }
            },
            1: {
                "label": "UM",
                "coordinates": {
                    "valid_time": ["2019-01-01 00:00:00"],
                    "pressures": [1000.0001]
                }
            }
        }
    }
    assert expect == result
