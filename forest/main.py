import bokeh.plotting
import bokeh.models
import bokeh.events
import bokeh.colors
import numpy as np
import os
import glob
from forest import (
        satellite,
        screen,
        series,
        data,
        load,
        view,
        earth_networks,
        rdt,
        nearcast,
        geo,
        colors,
        layers,
        db,
        keys,
        presets,
        redux,
        rx,
        unified_model,
        intake_loader,
        navigate,
        parse_args)
import forest.config as cfg
import forest.middlewares as mws
from forest.observe import Observable
from forest.db.util import autolabel
import datetime as dt


def main(argv=None):
    args = parse_args.parse_args(argv)
    if len(args.files) > 0:
        config = cfg.from_files(args.files, args.file_type)
    else:
        config = cfg.Config.load(
                args.config_file,
                variables=cfg.combine_variables(
                    os.environ,
                    args.variables))

    # Full screen map
    lon_range = (90, 140)
    lat_range = (-23.5, 23.5)
    x_range, y_range = geo.web_mercator(
        lon_range,
        lat_range)
    figure = bokeh.plotting.figure(
        x_range=x_range,
        y_range=y_range,
        x_axis_type="mercator",
        y_axis_type="mercator",
        active_scroll="wheel_zoom")
    tile = bokeh.models.WMTSTileSource(
        url="https://maps.wikimedia.org/osm-intl/{Z}/{X}/{Y}.png",
        attribution=""
    )

    figures = [figure]
    for _ in range(2):
        f = bokeh.plotting.figure(
            x_range=figure.x_range,
            y_range=figure.y_range,
            x_axis_type="mercator",
            y_axis_type="mercator",
            active_scroll="wheel_zoom")
        figures.append(f)

    for f in figures:
        f.axis.visible = False
        f.toolbar.logo = None
        f.toolbar_location = None
        f.min_border = 0
        f.add_tile(tile)

    figure_row = layers.FigureRow(figures)

    color_mapper = bokeh.models.LinearColorMapper(
            low=0,
            high=1,
            palette=bokeh.palettes.Plasma[256])
    for figure in figures:
        colorbar = bokeh.models.ColorBar(
            color_mapper=color_mapper,
            orientation="horizontal",
            background_fill_alpha=0.,
            location="bottom_center",
            major_tick_line_color="black",
            bar_line_color="black")
        figure.add_layout(colorbar, 'center')

    # Database/File system loader(s)
    for group in config.file_groups:
        if group.label not in data.LOADERS:
            database = None
            if group.locator == "database":
                database = db.get_database(group.database_path)
            loader = load.Loader.group_args(
                    group, args, database=database)
            data.add_loader(group.label, loader)

    renderers = {}
    viewers = {}
    for name, loader in data.LOADERS.items():
        if isinstance(loader, rdt.Loader):
            viewer = rdt.View(loader)
        elif isinstance(loader, earth_networks.Loader):
            viewer = earth_networks.View(loader)
        elif isinstance(loader, data.GPM):
            viewer = view.GPMView(loader, color_mapper)
        elif isinstance(loader, satellite.EIDA50):
            viewer = view.EIDA50(loader, color_mapper)
        elif isinstance(loader, nearcast.NearCast):
            viewer = view.NearCast(loader, color_mapper)
            viewer.set_hover_properties(nearcast.NEARCAST_TOOLTIPS)
        elif isinstance(loader, intake_loader.IntakeLoader):
            viewer = view.UMView(loader, color_mapper)
            viewer.set_hover_properties(intake_loader.INTAKE_TOOLTIPS,
                                        intake_loader.INTAKE_FORMATTERS)
        else:
            viewer = view.UMView(loader, color_mapper)
        viewers[name] = viewer
        renderers[name] = [
                viewer.add_figure(f)
                for f in figures]

    image_sources = []
    for name, viewer in viewers.items():
        if isinstance(viewer, (view.UMView, view.GPMView, view.EIDA50)):
            image_sources.append(viewer.source)

    # Lakes
    for figure in figures:
        add_feature(figure, data.LAKES, color="lightblue")

    features = []
    for figure in figures:
        features += [
            add_feature(figure, data.COASTLINES),
            add_feature(figure, data.BORDERS)]

    # Disputed borders
    for figure in figures:
        add_feature(figure, data.DISPUTED, color="red")

    toggle = bokeh.models.CheckboxGroup(
            labels=["Coastlines"],
            active=[0],
            width=135)

    def on_change(attr, old, new):
        if len(new) == 1:
            for feature in features:
                feature.visible = True
        else:
            for feature in features:
                feature.visible = False

    toggle.on_change("active", on_change)

    dropdown = bokeh.models.Dropdown(
            label="Color",
            menu=[
                ("Black", "black"),
                ("White", "white")],
            width=50)
    autolabel(dropdown)

    def on_change(attr, old, new):
        for feature in features:
            feature.glyph.line_color = new

    dropdown.on_change("value", on_change)

    # Image opacity user interface (client-side)
    slider = bokeh.models.Slider(
        start=0,
        end=1,
        step=0.1,
        value=1.0,
        show_value=False)

    def is_image(renderer):
        return isinstance(getattr(renderer, 'glyph', None), bokeh.models.Image)

    renderers_list = []
    for _, r in renderers.items():
        renderers_list += r
    image_renderers = [r for r in renderers_list if is_image(r)]
    custom_js = bokeh.models.CustomJS(
            args=dict(renderers=image_renderers),
            code="""
            renderers.forEach(function (r) {
                r.glyph.global_alpha = cb_obj.value
            })
            """)
    slider.js_on_change("value", custom_js)

    menu = []
    for k, _ in config.patterns:
        menu.append((k, k))

    layers_ui = layers.LayersUI(menu)

    div = bokeh.models.Div(text="", width=10)
    border_row = bokeh.layouts.row(
        bokeh.layouts.column(toggle),
        bokeh.layouts.column(div),
        bokeh.layouts.column(dropdown))


    navigator = navigate.Navigator(config)

    # Pre-select menu choices (if any)
    initial_state = {}
    for _, pattern in config.patterns:
        initial_state = db.initial_state(navigator, pattern=pattern)
        break

    middlewares = [
        mws.echo,
        keys.navigate,
        db.InverseCoordinate("pressure"),
        db.next_previous,
        db.Controls(navigator),
        db.Converter({
            "valid_times": db.stamps,
            "inital_times": db.stamps
        }),
        colors.palettes,
        presets.Middleware(presets.proxy_storage(config.presets_file)),
        presets.middleware,
        layers.middleware,
    ]
    store = redux.Store(
        redux.combine_reducers(
            db.reducer,
            layers.reducer,
            series.reducer,
            colors.reducer,
            presets.reducer),
        initial_state=initial_state,
        middlewares=middlewares)

    # Connect renderer.visible states to store
    artist = layers.Artist(renderers)
    artist.connect(store)

    # Connect layers controls
    layers_ui.add_subscriber(store.dispatch)
    layers_ui.connect(store)

    # Connect tools controls
    tools_panel = series.ToolsPanel()
    tools_panel.connect(store)

    # Connect tap listener
    tap_listener = screen.TapListener()
    tap_listener.connect(store)

    # Connect figure controls/views
    figure_ui = layers.FigureUI()
    figure_ui.add_subscriber(store.dispatch)
    figure_row.connect(store)

    # Connect color palette controls
    color_palette = colors.ColorPalette(color_mapper).connect(store)

    # Connect limit controllers to store
    source_limits = colors.SourceLimits(image_sources)
    source_limits.add_subscriber(store.dispatch)

    user_limits = colors.UserLimits().connect(store)

    # Preset
    preset_ui = presets.PresetUI().connect(store)

    # Connect navigation controls
    controls = db.ControlView()
    controls.add_subscriber(store.dispatch)
    store.add_subscriber(controls.render)

    def old_world(state):
        kwargs = {k: state.get(k, None) for k in db.State._fields}
        return db.State(**kwargs)

    connector = layers.ViewerConnector(viewers, old_world).connect(store)

    # Set default time series visibility
    store.dispatch(series.on_toggle())

    # Set top-level navigation
    store.dispatch(db.set_value("patterns", config.patterns))

    # Pre-select first layer
    for name, _ in config.patterns:
        row_index = 0
        store.dispatch(layers.set_label(row_index, name))
        store.dispatch(layers.set_active(row_index, [0]))
        break

    tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                bokeh.models.Div(text="Layout:"),
                figure_ui.layout,
                bokeh.models.Div(text="Navigate:"),
                controls.layout,
                bokeh.models.Div(text="Compare:"),
                layers_ui.layout),
            title="Control"
        ),
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                tools_panel.buttons["toggle_time_series"],
                ),
            title="Tools"),
        bokeh.models.Panel(
            child=bokeh.layouts.column(
                border_row,
                bokeh.layouts.row(slider),
                preset_ui.layout,
                color_palette.layout,
                user_limits.layout
                ),
            title="Settings")
        ])

    # Series sub-figure widget
    series_figure = bokeh.plotting.figure(
                plot_width=400,
                plot_height=200,
                x_axis_type="datetime",
                toolbar_location=None,
                border_fill_alpha=0)
    series_figure.toolbar.logo = None

    tool_layout = series.ToolLayout(series_figure)
    tool_layout.connect(store)

    series_view = series.SeriesView.from_groups(
            series_figure,
            config.file_groups)
    series_view.add_subscriber(store.dispatch)
    series_args = (rx.Stream()
                .listen_to(store)
                .map(series.select_args)
                .filter(lambda x: x is not None)
                .distinct())
    series_args.map(lambda a: series_view.render(*a))
    series_args.map(print)  # Note: map(print) creates None stream
    for f in figures:
        f.on_event(bokeh.events.Tap, tap_listener.update_xy)
        marker = screen.MarkDraw(f).connect(store)


    # Minimise controls to ease navigation
    compact_button = bokeh.models.Button(
            label="Compact")
    compact_minus = bokeh.models.Button(label="-", width=50)
    compact_plus = bokeh.models.Button(label="+", width=50)
    compact_navigation = bokeh.layouts.column(
            compact_button,
            bokeh.layouts.row(
                compact_minus,
                compact_plus,
                width=100))
    control_root = bokeh.layouts.column(
            compact_button,
            tabs,
            name="controls")

    display = "large"
    def on_compact():
        nonlocal display
        if display == "large":
            control_root.height = 100
            control_root.width = 120
            compact_button.width = 100
            compact_button.label = "Expand"
            control_root.children = [
                    compact_navigation]
            display = "compact"
        else:
            control_root.height = 500
            control_root.width = 300
            compact_button.width = 300
            compact_button.label = "Compact"
            control_root.children = [compact_button, tabs]
            display = "large"

    compact_button.on_click(on_compact)

    # Add key press support
    key_press = keys.KeyPress()
    key_press.add_subscriber(store.dispatch)

    document = bokeh.plotting.curdoc()
    document.title = "FOREST"
    document.add_root(control_root)
    document.add_root(tool_layout.figures_row)
    document.add_root(figure_row.layout)
    document.add_root(key_press.hidden_button)


def any_none(obj, attrs):
    return any([getattr(obj, x) is None for x in attrs])


def add_feature(figure, data, color="black"):
    source = bokeh.models.ColumnDataSource(data)
    return figure.multi_line(
        xs="xs",
        ys="ys",
        source=source,
        color=color)


if __name__.startswith("bk"):
    main()
