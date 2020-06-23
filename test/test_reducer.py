import pytest
import forest
from forest.state import State


@pytest.mark.parametrize("state,action,expect", [
    pytest.param(
        {},
        forest.actions.no_action(),
        {},
        id="no_action"
    ),
    pytest.param(
        {},
        forest.actions.set_palette_name("Accent"),
        {
            "colorbar": {
                "name": "Accent"
            }
        },
        id="set_palette_name"),
    pytest.param(
        {},
        forest.actions.set_user_high(100),
        {
            "colorbar": {
                "limits": {
                    "user": {"high": 100}
                }
            }
        },
        id="set_palette_name"),
    pytest.param(
        {},
        forest.actions.save_layer(0, {"key": "value"}),
        {
            "layers": {
                "index": {
                    0: {"key": "value"}
                }
            }
        },
        id="save_layer"
    ),
    pytest.param(
        {},
        forest.actions.set_state({"tile": {"name": "Wikimedia"}}).to_dict(),
        {"tile": {"name": "Wikimedia"}},
        id="set_state")
])
def test_reducer(state, action, expect):
    result = forest.reducer(state, action)
    assert State.from_dict(result) == State.from_dict(expect)
