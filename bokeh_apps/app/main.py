import yaml
import jinja2
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')
import bokeh.io
import bokeh.plotting
import bokeh.themes
import numpy
import matplotlib
matplotlib.use('agg')
import forest.util
import forest.plot
import forest.control
import forest.data
import forest.aws


class NameSpace():
    """Simple namespace turns attrs into properties"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_environment(env):
    """Load settings from environment variables"""
    if "FOREST_START" in env:
        start_date = dt.datetime.strptime(env["FOREST_START"], "%Y%m%d")
    else:
        start_date = days_ago(forest.data.NUM_DATA_DAYS)
    download_data = env.get("FOREST_DOWNLOAD_DATA",
                            "False").upper() == "TRUE"
    download_directory = parse_variable(env,
                                        ["FOREST_DOWNLOAD_DIR",
                                         "LOCAL_ROOT"],
                                        os.path.expanduser("~/SEA_data/"))
    if "FOREST_MOUNT_DIR" in env:
        mount_directory = os.path.expanduser(env["FOREST_MOUNT_DIR"])
    else:
        s3_root = env.get("S3_ROOT", os.path.expanduser("~/s3/"))
        mount_directory = os.path.join(s3_root, 'stephen-sea-public-london')
    if "FOREST_CONFIG_FILE" in env:
        config_file = env["FOREST_CONFIG_FILE" ]
    else:
        config_file = None
    return NameSpace(start_date=start_date,
                     download_data=download_data,
                     download_directory=download_directory,
                     mount_directory=mount_directory,
                     config_file=config_file)


def parse_variable(env, names, default):
    if isinstance(names, str):
        names = [names]
    for name in names:
        if name in env:
            return env[name]
    return default


def load_config(path):
    """Load Forest configuration from file"""
    with open(path) as stream:
        return yaml.load(stream)


def south_east_asia_config():
    return {
        "title": "Two model comparison",
        "regions": [
            {
                "name": "South east Asia",
                "longitude_range": [90.0, 153.96],
                "latitude_range": [-18.0, 29.96]
            },
            {
                "name": "Indonesia",
                "longitude_range": [99.875, 120.111],
                "latitude_range": [-15.1, 1.0865]
            },
            {
                "name": "Malaysia",
                "longitude_range": [95.25, 108.737],
                "latitude_range": [-2.75, 10.7365]
            },
            {
                "name": "Philippines",
                "longitude_range": [115.8, 131.987],
                "latitude_range": [3.1375, 21.349]
            },
        ],
        "models": [
            {
                "name": "N1280 GA6 LAM Model",
                "file": {
                    "pattern": "SEA_n1280_ga6_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ga6"
                }
            },
            {
                "name": "SE Asia 4.4KM RA1-T ",
                "file": {
                    "pattern": "SEA_km4p4_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Indonesia 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_indon2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Malaysia 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_mal2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            },
            {
                "name": "Philipines 1.5KM RA1-T",
                "file": {
                    "pattern": "SEA_phi2km1p5_ra1t_{:%Y%m%dT%H%MZ}.nc",
                    "format": "ra1t"
                }
            }
        ]
    }


def days_ago(days):
    """Helper method to select time N days ago"""
    return (dt.datetime.now() - dt.timedelta(days=days)).replace(second=0, microsecond=0)


@forest.util.timer
def main(bokeh_id):
    env = parse_environment(os.environ)
    if env.config_file is None:
        settings = south_east_asia_config()
    else:
        print("Loading: {}".format(env.config_file))
        settings = load_config(env.config_file)
    app(env, settings, bokeh.plotting.curdoc())


def load_app(config_file):
    env = parse_environment(os.environ)
    print("Loading: {}".format(config_file))
    settings = load_config(config_file)
    def wrapper(document):
        return app(env, settings, document,
                   custom_server=True)
    return wrapper


def app(env, settings, document, custom_server=False):
    '''Two-model bokeh application main program'''
    bokeh_id = ""
    if env.download_data:
        file_loader = forest.aws.S3Bucket(
                server_address='https://s3.eu-west-2.amazonaws.com',
                bucket_name='stephen-sea-public-london',
                download_directory=env.download_directory)
        user_feedback_directory = os.path.join(env.download_directory,
                'user_feedback')
    else:
        # FUSE mounted file system
        file_loader = forest.aws.S3Mount(env.mount_directory)
        user_feedback_directory = os.path.join(env.mount_directory,
                'user_feedback')

    models = settings['models']
    plot_descriptions = {
        model['name']: model['name']
            for model in models
    }
    file_patterns = {
        model['name']: model['file']['pattern']
            for model in models
    }
    file_formats = {
        model['name']: model['file']['format']
            for model in models
    }
    model_run_times = forest.data.get_model_run_times(env.start_date,
                                                      forest.data.NUM_DATA_DAYS,
                                                      forest.data.MODEL_RUN_PERIOD)
    datasets = forest.data.get_available_datasets(model_run_times,
                                                  file_patterns,
                                                  file_formats,
                                                  file_loader)
    try:
        init_fcast_time = list(datasets.keys())[-1]
    except IndexError:
        print("[WARNING] No forecast times found")
        layout1 = forest.util.load_error_page()
        document.add_root(layout1)
        return

    print('Most recent dataset available is {0}, forecast time selected for display.'.format(init_fcast_time))

    plot_type_time_lookups = {
        'precipitation': 'precipitation',
        'air_temperature': 'air_temperature',
        'wind_vectors': 'x_wind',
        'wind_mslp': 'x_wind',
        'wind_streams': 'x_wind',
        'mslp': 'mslp',
        'cloud_fraction': 'cloud_fraction',
    }
    for var in forest.data.PRECIP_ACCUM_VARS:
        plot_type_time_lookups[var] = var

    # Setup and display plots
    plot_opts = forest.util.create_colour_opts(list(plot_type_time_lookups.keys()))

    init_data_time_index = 1
    init_var = 'precipitation'

    init_model_left = models[0]['name']
    if len(models) == 1:
        init_model_right = models[0]['name']
    else:
        init_model_right = models[1]['name']
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    available_times = forest.data.get_available_times(datasets[init_fcast_time],
                                                      plot_type_time_lookups[init_var])
    init_data_time = available_times[init_data_time_index]
    num_times = available_times.shape[0]

    bokeh_figure = bokeh.plotting.figure(
        active_inspect=None,
        sizing_mode="stretch_both",
        match_aspect=True,
        name="map")
    forest.plot.add_x_axes(bokeh_figure, "above")
    forest.plot.add_y_axes(bokeh_figure, "right")

    # Add cartopy coastline to bokeh figure
    def _extent(region):
        x_start, x_end = region["longitude_range"]
        y_start, y_end = region["latitude_range"]
        # Note: this ordering should be removed from code base
        #       it is too confusing
        return y_start, y_end, x_start, x_end
    region_names = [region['name'] for region in settings["regions"]]
    region_dict = {region['name']: region['name'] for region in settings["regions"]}
    region_extents = {region['name']: _extent(region) for region in settings["regions"]}
    coords = region_extents[region_names[0]]
    y_start = coords[0]
    y_end = coords[1]
    x_start = coords[2]
    x_end = coords[3]
    extent = (x_start, x_end, y_start, y_end)
    forest.add_coastlines(bokeh_figure)
    forest.add_borders(bokeh_figure)

    # Set up plots
    forest_datasets = datasets[init_fcast_time]
    plot_obj_left = forest.plot.ForestPlot(forest_datasets,
                                           plot_descriptions,
                                           plot_opts,
                                           'plot_left' + bokeh_id,
                                           init_var,
                                           init_model_left,
                                           region_names[0],
                                           region_extents,
                                           app_path,
                                           init_data_time,
                                           bokeh_figure=bokeh_figure)
    plot_obj_left.render()

    plot_obj_right = forest.plot.ForestPlot(forest_datasets,
                                            plot_descriptions,
                                            plot_opts,
                                            'plot_right' + bokeh_id,
                                            init_var,
                                            init_model_right,
                                            region_names[0],
                                            region_extents,
                                            app_path,
                                            init_data_time,
                                            bokeh_figure=bokeh_figure,
                                            visible=False)

    colorbar_widget = plot_obj_left.create_colorbar_widget()

    # Set up GUI controller class
    # feedback_controller = forest.FeedbackController(user_feedback_directory,
    #                                                 bokeh_id)
    forest_controller = forest.ForestController(init_var,
                                                init_data_time_index,
                                                datasets,
                                                init_fcast_time,
                                                plot_type_time_lookups,
                                                [plot_obj_left, plot_obj_right],
                                                bokeh_figure,
                                                region_dict)

    # Attach bokeh layout to current document
    controls = bokeh.layouts.column(
             forest_controller.model_run_drop_down,
             forest_controller.time_previous_button,
             forest_controller.time_next_button,
             forest_controller.left_model_drop_down,
             forest_controller.right_model_drop_down,
             forest_controller.left_right_toggle,
             forest_controller.model_variable_drop_down,
             forest_controller.region_drop_down,
             name="btn")
    colorbar_widget.name = "colorbar"
    roots = [controls,
             bokeh_figure,
             colorbar_widget]
    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'
    if bokeh_mode == 'server':
        for root in roots:
            document.add_root(root)
    elif bokeh_mode == 'cli':
        root = bokeh.layouts.column(*roots)
        bokeh.io.show(root)
    document.title = settings["title"]

    if custom_server:
        apply_theme(document)
        apply_template(document)


def apply_theme(document):
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "theme.yaml")
    document.theme = bokeh.themes.Theme(filename=filename)


def apply_template(document):
    script_dir = os.path.dirname(__file__)
    loader = jinja2.FileSystemLoader(script_dir)
    env = jinja2.Environment(loader=loader)
    document.template = env.get_template("templates/index.html")


if __name__ == '__main__' or __name__.startswith("bk"):
    main(__name__)
