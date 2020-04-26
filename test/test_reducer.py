import pytest
import forest


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
    )
])
def test_reducer(state, action, expect):
    assert forest.reducer(state, action) == expect
