from forest import drivers


class Navigator:
    def __init__(self, config, color_mapper=None):
        # TODO: Once the idea of a "Group" exists we can avoid using the
        # config and defer the sub-navigator creation to each of the
        # groups. This will remove the need for the `_from_group` helper
        # and the logic in FileSystemNavigator.from_file_type().
        # Also, it'd be good to switch the identification of groups from
        # using the `pattern` to using the `label`. In general, not every
        # group would have a `pattern`.
        # e.g.
        # self._navigators = {group.label: group.navigator for group in ...}
        self._navigators = {group.pattern: self._from_group(group, color_mapper)
                           for group in config.file_groups}

    @classmethod
    def _from_group(cls, group, color_mapper=None):
        settings = {
            "label": group.label,
            "pattern": group.pattern,
            "locator": group.locator,
            "database_path": group.database_path,
            "color_mapper": color_mapper,
        }
        dataset = drivers.get_dataset(group.file_type, settings)
        return dataset.navigator()

    def variables(self, pattern):
        navigator = self._navigators[pattern]
        return navigator.variables(pattern)

    def initial_times(self, pattern, variable=None):
        navigator = self._navigators[pattern]
        return navigator.initial_times(pattern, variable=variable)

    def valid_times(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.valid_times(pattern, variable, initial_time)

    def pressures(self, pattern, variable, initial_time):
        navigator = self._navigators[pattern]
        return navigator.pressures(pattern, variable, initial_time)
