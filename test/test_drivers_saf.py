import bokeh.models
import forest.drivers


def test_map_view():
    color_mapper = bokeh.models.ColorMapper()
    dataset = forest.drivers.get_dataset("saf", {
        "pattern": "saf.nc",
        "color_mapper": color_mapper
    })
    view = dataset.map_view()
    assert isinstance(view, forest.view.UMView)
