"""Time navigation component"""
from forest import rx
from forest.observe import Observable
from forest.util import to_datetime as _to_datetime
import forest.db.control
import bokeh.plotting
import numpy as np


class _Axis:
    """Helper to find datetimes

    Maps all datetime variants using str(datetime.datetime) as a key
    """
    def __init__(self):
        self._mapping = {}
        self._values = []

    @property
    def datetimes(self):
        return [_to_datetime(t) for t in self.times]

    @property
    def times(self):
        return self._values

    @times.setter
    def times(self, values):
        """Intercept assignment to populate mapping"""
        self._mapping = {self._key(v): i for i, v in enumerate(values)}
        self._values = values

    def index(self, time):
        """Map random datetime object to index"""
        return self._mapping[self._key(time)]

    def value(self, i):
        """Recover original value at index"""
        return self._values[i]

    @staticmethod
    def _key(t):
        return str(_to_datetime(t))



class TimeUI(Observable):
    """Allow navigation through time"""
    def __init__(self):
        self._axis = _Axis()
        self.source = bokeh.models.ColumnDataSource(dict(
            x=[],
            y=[],
        ))
        self.figure = bokeh.plotting.figure(
            plot_height=80,
            plot_width=800,
            border_fill_alpha=0,
            tools='xpan',
            x_axis_type='datetime')
        self.figure.toolbar.active_drag = 'auto'

        active_scroll = bokeh.models.WheelZoomTool(dimensions='width')
        self.figure.add_tools(active_scroll)
        self.figure.toolbar.active_scroll = active_scroll

        renderer = self.figure.square(x="x", y="y", source=self.source,
                      fill_color='black',
                      line_color='black')
        renderer.selection_glyph = bokeh.models.Square(
            fill_color="red",
            line_color="red")
        renderer.nonselection_glyph = bokeh.models.Square(
            fill_color="black",
            line_color="black",
            fill_alpha=0.2,
            line_alpha=0.2,
        )

        # X-axis formatter breakpoints
        formatter = self.figure.xaxis[0].formatter
        formatter.hourmin = ['%H:%M']
        formatter.hours = ['%H:%M']
        formatter.days = ["%d %B"]
        formatter.months = ["%b %Y"]

        # Customize figure to be a time slider widget
        self.figure.grid.grid_line_color = None
        self.figure.yaxis.visible = False
        self.figure.toolbar_location = None
        self.figure.xaxis.fixed_location = 0
        self.figure.title.text = "Select time"
        self.figure.title.align = "center"

        # Hover interaction
        hover_tool = bokeh.models.HoverTool(tooltips=None)
        self.figure.add_tools(hover_tool)

        selected_span = bokeh.models.Span(
            dimension="height",
            line_color="red",
            location=0)
        self.figure.add_layout(selected_span)
        custom_js = bokeh.models.CustomJS(args=dict(
            span=selected_span,
            source=self.source), code="""
            if (cb_obj.indices.length === 0) {
                return;
            }
            span.location = source.data['x'][cb_obj.indices[0]];
        """)
        self.source.selected.js_on_change('indices', custom_js)

        # Wire up Tap event
        tap_js = bokeh.models.CustomJS(args=dict(
            source=self.source), code="""
            let x = source.data['x'];
            let distance = x.map((t) => Math.abs(t - cb_obj.x));
            let minIndex = distance.reduce(function(bestIndex, value, index, values) {
                if (value < values[bestIndex]) {
                    return index;
                } else {
                    return bestIndex;
                }
            }, 0);
            source.selected.indices = [minIndex];
            source.change.emit();
        """)
        self.figure.js_on_event(bokeh.events.Tap, tap_js)

        # Span that follows cursor
        if len(self.source.data["x"]) > 0:
            location = self.source.data["x"][0]
        else:
            location = 0
        span = bokeh.models.Span(
            dimension="height",
            line_color="grey",
            location=location)
        js = bokeh.models.CustomJS(args=dict(span=span), code="""
            span.location = cb_data.geometry.x;
        """)
        self.figure.add_layout(span)
        hover_tool.callback = js

        self.source.selected.on_change('indices', self.on_selected)

        # Band to highlight valid times
        self.band_source = bokeh.models.ColumnDataSource(dict(
            base=[-1, 1],
            upper=[0, 0],
            lower=[0, 0]
        ))
        band = bokeh.models.Band(
            dimension='width',
            base='base',
            lower='lower',
            upper='upper',
            fill_color='grey',
            fill_alpha=0.2,
            source=self.band_source
        )
        self.figure.add_layout(band)

        # Controls
        self.buttons = {
            "play": bokeh.models.Button(label="Play",
                                        css_classes=['play'],
                                        width=75),
            "pause": bokeh.models.Button(label="Pause",
                                         css_classes=['pause'],
                                         width=75),
            "next": bokeh.models.Button(label="Next",
                                        width=75),
            "previous": bokeh.models.Button(label="Previous",
                                            width=75)
        }

        # Play JS
        custom_js = bokeh.models.CustomJS(args=dict(source=self.source), code="""
            // Simple JS animation
            console.log('Play');
            window.playing = true;
            var interval = 500;
            let nextFrame = function() {
                if (window.playing) {
                    if (source.selected.indices.length > 0) {
                        let i = source.selected.indices[0];
                        let n = source.data['x'].length;
                        source.selected.indices = [(i + 1) % n];
                        source.change.emit();
                    }
                    setTimeout(nextFrame, interval);
                }
            };
            setTimeout(nextFrame, interval);
        """)
        self.buttons["play"].js_on_click(custom_js)

        # Pause behaviour
        custom_js = bokeh.models.CustomJS(args=dict(source=self.source), code="""
            console.log('Pause');
            window.playing = false;
        """)
        self.buttons["pause"].js_on_click(custom_js)

        # Previous behaviour
        custom_js = bokeh.models.CustomJS(args=dict(source=self.source), code="""
            // Simple JS animation
            console.log('Previous');
            if (source.selected.indices.length > 0) {
                let i = source.selected.indices[0];
                source.selected.indices = [i - 1];
                source.change.emit();
            }
        """)
        self.buttons["previous"].js_on_click(custom_js)

        # Next behaviour
        custom_js = bokeh.models.CustomJS(args=dict(source=self.source), code="""
            // Simple JS animation
            console.log('Next');
            if (source.selected.indices.length > 0) {
                let i = source.selected.indices[0];
                let n = source.data['x'].length;
                source.selected.indices = [(i + 1) % n];
                source.change.emit();
            }
        """)
        self.buttons["next"].js_on_click(custom_js)

        self.layout = bokeh.layouts.column(
            bokeh.layouts.row(
                self.figure, sizing_mode="stretch_width"),
            bokeh.layouts.row(
                self.buttons["previous"],
                self.buttons["play"],
                self.buttons["pause"],
                self.buttons["next"],
                sizing_mode="stretch_width"
            ),
            sizing_mode="stretch_width")

        super().__init__()

    def on_selected(self, attr, old, new):
        """Notify store of set valid time action"""
        if len(new) > 0:
            i = new[0]
            value = self._axis.value(i)
            print(value, self.source.data["x"][i])
            self.notify(forest.db.control.set_value('valid_time', value))

    def connect(self, store):
        """Connect component to store

        Converts state into stream of unique property changes suitable
        for use with render method
        """
        self.add_subscriber(store.dispatch)
        stream = (rx.Stream()
                    .listen_to(store)
                    .map(self.to_props)
                    .filter(lambda x: x is not None)
                    .distinct())
        stream.map(lambda props: self.render(*props))
        return self

    def to_props(self, state):
        """Convert state to properties needed by component"""
        if ('valid_time' not in state) or ('valid_times' not in state):
            return
        return state['valid_time'], sorted(state['valid_times'])

    def render(self, time, times):
        """React to state changes"""
        self._axis.times = times
        self.source.data = {
            "x": self._axis.datetimes,
            "y": np.zeros(len(times))
        }

        try:
            index = self._axis.index(time)
        except KeyError:
            return
        self.source.selected.indices = [index]

        # Title
        time = self._axis.datetimes[index]
        self.figure.title.text = f"{time:%A %d %B %Y %H:%M}"

        # Band
        if len(times) > 0:
            upper = [max(times), max(times)]
            lower = [min(times), min(times)]
        else:
            upper = [0, 0]
            lower = [0, 0]
        self.band_source.data = dict(
            base=[-1, 1],
            upper=upper,
            lower=lower
        )
