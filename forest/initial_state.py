"""Calculate initial state using Navigator API"""


def from_pattern(navigator, pattern=None):
    """Find initial state given navigator"""
    state = {}
    state["pattern"] = pattern
    variables = navigator.variables(pattern)
    state["variables"] = variables
    if len(variables) == 0:
        return state
    variable = variables[0]
    state["variable"] = variable
    initial_times = navigator.initial_times(pattern, variable)
    state["initial_times"] = initial_times
    if len(initial_times) == 0:
        return state
    initial_time = max(initial_times)
    state["initial_time"] = initial_time
    valid_times = navigator.valid_times(pattern, variable, initial_time)
    state["valid_times"] = valid_times
    if len(valid_times) > 0:
        state["valid_time"] = min(valid_times)
    pressures = navigator.pressures(
        variable=variable, pattern=pattern, initial_time=initial_time
    )
    pressures = list(reversed(sorted(pressures)))
    state["pressures"] = pressures
    if len(pressures) > 0:
        state["pressure"] = pressures[0]
    return state


def from_labels(navigator, labels):
    """Navigator initial state"""
    result = {}

    # Patterns
    # TODO: Refactor app to use label instead of pattern
    result["patterns"] = [(label, label) for label in labels]
    if len(labels) > 0:
        label = labels[0]
        result["pattern"] = label

        # Variables
        variables = navigator.variables(label)
        result["variables"] = variables
        if len(variables) > 0:
            variable = variables[0]
            result["variable"] = variable

            # Initial times
            initial_times = navigator.initial_times(label, variable)
            result["initial_times"] = initial_times
            if len(initial_times) > 0:
                initial_time = initial_times[0]
                result["initial_time"] = initial_time

                # Valid times
                valid_times = navigator.valid_times(
                    label, variable, initial_time
                )
                result["valid_times"] = valid_times
                if len(valid_times) > 0:
                    result["valid_time"] = valid_times[0]

                # Pressures
                pressures = navigator.pressures(label, variable, initial_time)
                result["pressures"] = pressures
                if len(pressures) > 0:
                    result["pressure"] = pressures[0]
    return result
