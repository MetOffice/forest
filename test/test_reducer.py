import forest


def test_reducer():
    state = {}
    action = forest.actions.no_action()
    assert forest.reducer(state, action) == {}
