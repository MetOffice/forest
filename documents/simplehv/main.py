
# import numpy as np
# import holoviews as hv
# import holoviews.plotting.bokeh

# renderer = hv.renderer('bokeh')

# points = hv.Points(np.random.randn(1000,2 )).opts(plot=dict(tools=['box_select', 'lasso_select']))
# selection = hv.streams.Selection1D(source=points)

# def selected_info(index):
#     arr = points.array()[index]
#     if index:
#         label = 'Mean x, y: %.3f, %.3f' % tuple(arr.mean(axis=0))
#     else:
#         label = 'No selection'
#     return points.clone(arr, label=label).opts(style=dict(color='red'))

# layout = points + hv.DynamicMap(selected_info, streams=[selection])

# doc = renderer.server_doc(layout)
# doc.title = 'HoloViews App'

import numpy as np
import holoviews as hv
import holoviews.plotting.bokeh
from holoviews.operation.datashader import datashade

hv.Store.current_backend = 'bokeh'
renderer = hv.Store.renderers['bokeh'].instance(mode='server', holomap='server')
points = hv.HoloMap({i: hv.Points(np.random.multivariate_normal((0,0), [[0.1, 0.1], [0.1, 1.0]], (1000000,))) for i in range(10)})
datashaded = datashade(points, x_sampling=0.01, y_sampling=0.01)
doc, _ = renderer(datashaded)
doc.title = 'HoloViews Datashade'