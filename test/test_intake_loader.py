from forest import intake_loader


def test_intake_loader_init():
    pattern = "institution_experiment_member_grid_table_activity"
    loader = intake_loader.IntakeLoader(pattern)
    assert loader.institution_id == "institution"
    assert loader.experiment_id == "experiment"
    assert loader.member_id == "member"
    assert loader.grid_label == "grid"
    assert loader.table_id == "table"
    assert loader.activity_id == "activity"
    assert loader.variable_id == ""  # Set to empty str for some reason


class Coord:
    def name(self):
        return "coordinate"

class Cube:
    def coords(self):
        return [Coord()]


def test_intake_loader_image(monkeypatch):
    def fake(
            experiment_id,
            table_id,
            grid_label,
            variable_id,
            institution_id,
            activity_id,
            member_id):
        return Cube()

    monkeypatch.setattr(intake_loader, "_load_from_intake", fake)

    pattern = "institution_experiment_member_grid_table_activity"
    loader = intake_loader.IntakeLoader(pattern)
    state = {}
    loader.image(state)
