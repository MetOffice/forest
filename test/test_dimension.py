import forest.dimension


def test_reducer_set_variables():
    label = "dataset_label"
    variables = ["foo", "bar"]
    action = forest.dimension.set_variables(label, variables)
    state = forest.dimension.reducer({}, action)
    assert state["dimension"][label]["variables"] == variables
