import pytest
from unittest.mock import Mock, sentinel
import bokeh.plotting
from forest import layers, redux


@pytest.mark.parametrize("action,expect", [
    ({"kind": "ANY"}, [{"kind": "ANY"}]),
    (layers.on_button_group(0, [0, 1, 2]), [layers.set_active(0, [0, 1, 2])]),
])
def test_middleware(action, expect):
    store = redux.Store(layers.reducer)
    result = list(layers.middleware(store, action))
    assert expect == result


@pytest.fixture
def listener():
    return Mock()


def test_figure_dropdown(listener):
    ui = layers.FigureUI()
    ui.add_subscriber(listener)
    ui.on_change(None, None, ui.labels[0])
    listener.assert_called_once_with(layers.set_figures(1))


def test_remove(listener):
    ui = layers.LayersUI()
    ui.add_subscriber(listener)
    ui.on_click_remove()
    listener.assert_called_once_with(layers.on_remove())


@pytest.mark.parametrize("state,actions,expect", [
    ({}, layers.set_figures(3), {"figures": 3}),
    ({}, layers.add_layer("Name"), {"labels": ["Name"]}),
    ({}, layers.on_remove(), {"labels": []}),
    ({"layers": {"labels": [None, None]}}, layers.on_remove(), {"labels": [None]}),
    ({}, layers.set_label(0, "Label"), {"labels": ["Label"]}),
    ({}, layers.set_label(0, "Other"), {"labels": ["Other"]}),
    ({}, layers.set_label(1, "Label"), {"labels": [None, "Label"]}),
    ({}, layers.set_label(2, "Label"), {"labels": [None, None, "Label"]}),
    ({}, layers.set_active(0, []), {"index": {0: {"active": []}}}),
    ({}, layers.set_active(1, []), {"index": {1: {"active": []}}}),
    ({"layers": {"index": {0: {"active": []}}}}, layers.set_active(0, [0]),
                {"index": {0: {"active": [0]}}}),
])
def test_reducer(state, actions, expect):
    if isinstance(actions, dict):
        actions = [actions]
    for action in actions:
        state = layers.reducer(state, action)
    result = state["layers"]
    assert result == expect


def test_reducer_save_layer():
    i = 42
    label = "Label"
    dataset = "Dataset"
    variable = "Variable"
    action = layers.save_layer(i, {"label": label,
                                   "dataset": dataset,
                                   "variable": variable})
    state = layers.reducer({}, action)
    assert state["layers"]["index"][i]["label"] == label
    assert state["layers"]["index"][i]["dataset"] == dataset
    assert state["layers"]["index"][i]["variable"] == variable


def test_reducer_remove_layer():
    index = 42
    state = {
        "layers": {
            "index": {
                index: {
                    "key": "value"
                }
            }
        }
    }
    action = layers.on_close(index)
    state = layers.reducer(state, action)
    assert state["layers"]["index"] == {}


def test_reducer_on_edit():
    i = 42
    action = layers.on_edit(i)
    state = layers.reducer({}, action)
    assert state["layers"]["mode"]["index"] == i
    assert state["layers"]["mode"]["state"] == "edit"


def test_reducer_on_add():
    action = layers.on_add()
    state = layers.reducer({}, action)
    assert state["layers"]["mode"]["state"] == "add"


def test_reducer_set_active():
    index = 42
    active = [0, 2]
    action = layers.set_active(index, active)
    state = layers.reducer({}, action)
    assert state["layers"]["index"][index]["active"] == active


def test_layersui_render_sets_button_groups():
    state = {
        "layers": {
            "figures": 3,
            "index": {
                0: {
                    "label": "Foo",
                    "active": [0, 1, 2]
                }
            }
        }
    }
    controls = layers.LayersUI()
    controls.render(*controls.to_props(state))
    assert len(controls.button_groups) == 1
    assert controls.button_groups[0].active == [0, 1, 2]
    assert controls.button_groups[0].labels == ["L", "C", "R"]


@pytest.mark.skip()
@pytest.mark.parametrize("from_labels,to_labels,expect", [
    ([], [], 0),
    ([], ["label"], 1),
    ([], ["label", "label"], 2),
    (["label"], [], 0),
    (["label", "label"], [], 0),
    (["label", "label"], ["label"], 1),
])
def test_controls_render_number_of_rows(from_labels, to_labels, expect):
    from_state = {
        "layers": {"labels": from_labels}
    }
    to_state = {
        "layers": {"labels": to_labels}
    }
    controls = layers.LayersUI()
    controls.render(*controls.to_props(from_state))
    controls.render(*controls.to_props(to_state))
    assert len(controls.columns["rows"].children) == expect


def test_on_button_group(listener):
    row_index = 3
    attr, old, new = None, [], [2]
    controls = layers.LayersUI()
    controls.add_subscriber(listener)
    controls.on_button_group(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_button_group(row_index, new))


@pytest.mark.skip("refactor to test on_event handlers")
def test_on_dropdown(listener):
    row_index = 3
    attr, old, new = None, None, "label"
    controls = layers.LayersUI()
    controls.add_subscriber(listener)
    controls.on_dropdown(row_index)(attr, old, new)
    listener.assert_called_once_with(layers.on_dropdown(row_index, new))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_figure_row(n):
    figures = [bokeh.plotting.figure() for _ in range(3)]
    figure_row = layers.FigureRow(figures)
    figure_row.render(n)
    assert len(figure_row.layout.children) == n


def test_visible_from_map_view():
    map_view = Mock()
    map_view.add_figure.return_value = sentinel.renderer
    figures = [sentinel.figure]
    visible = layers.Visible.from_map_view(map_view, figures)
    map_view.add_figure.assert_called_once_with(sentinel.figure)
    assert visible.renderers == [sentinel.renderer]


def test_visible_render():
    renderers = Mock(), Mock(), Mock()
    visible = layers.Visible(renderers)
    visible.active = [0, 2]
    assert renderers[0].visible == True
    assert renderers[1].visible == False
    assert renderers[2].visible == True


def test_gallery_render():
    state = {
        "layers": {
            "index": {
                42: {
                    "dataset": "Dataset",
                    "variable": "Variable",
                    "active": [0]
                }
            }
        }
    }
    pools = {
        "Dataset": Mock()
    }
    gallery = layers.Gallery(pools)
    gallery.render(state)
    pools["Dataset"].acquire.assert_called_once_with()
