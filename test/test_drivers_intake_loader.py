import bokeh.models
import forest.drivers
import forest.map_view


def test_dataset_navigator():
    dataset = forest.drivers.get_dataset("intake_loader", {})
    navigator = dataset.navigator()
    assert isinstance(navigator, forest.drivers.intake_loader.Navigator)


def test_dataset_map_view():
    pattern = "a_b_c_d_e_f"
    dataset = forest.drivers.get_dataset("intake_loader", {
        "pattern": pattern,
    })
    color_mapper = bokeh.models.ColorMapper()
    map_view = dataset.map_view(color_mapper)
    assert isinstance(map_view, forest.map_view.ImageView)
    assert map_view.tooltips == forest.drivers.intake_loader.INTAKE_TOOLTIPS
    assert map_view.formatters == forest.drivers.intake_loader.INTAKE_FORMATTERS
