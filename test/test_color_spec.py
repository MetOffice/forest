import bokeh.models
import forest.colors


def test_colorspec_apply():
    color_mapper = bokeh.models.LinearColorMapper()
    low, high = 42, 137
    spec = forest.colors.ColorSpec(low=low, high=high)
    spec.apply(color_mapper)
    assert color_mapper.low == low
    assert color_mapper.high == high
