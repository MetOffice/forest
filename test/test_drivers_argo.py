import forest.drivers
import bokeh.plotting


# I believe this ought to be a pytest fixture but I haven't learnt how to use
# those yet
def _argo_dataset_setup():
    figure = bokeh.plotting.figure()
    dataset = forest.drivers.get_dataset("argo", {})
    return figure, dataset


def test_argo_profile_view():
    figure, dataset = _argo_dataset_setup()
    dataset.paths=[None,]
    profile_view = dataset.profile_view(figure)
    state = {}
    profile_view.render(state)
