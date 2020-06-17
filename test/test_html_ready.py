import forest
import forest.state
import forest.actions


def test_reducer():
    state = forest.state.State()
    action = forest.actions.html_loaded()
    state = forest.reducer(state, action.to_dict())
    state = forest.state.State.from_dict(state)
    assert state.bokeh.html_loaded == True  # noqa: E712


def test_action():
    action = forest.actions.html_loaded()
    assert action.kind == forest.actions.HTML_LOADED
