import yaml
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')
import bokeh.io
import bokeh.plotting
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
    download_directory = env.get("LOCAL_ROOT",
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


def load_config(path):
    """Load Forest configuration from file"""
    with open(path) as stream:
        return yaml.load(stream)


def south_east_asia_config():
    return {
        "models": [
            {
                "name": "N1280 GA6 LAM Model",
                "file_pattern": "SEA_n1280_ga6_{%Y%m%dT%H%MZ}.nc"
            },
            {
                "name": "SE Asia 4.4KM RA1-T ",
                "file_pattern": "SEA_km4p4_ra1t_{%Y%m%dT%H%MZ}.nc"
            },
            {
                "name": "Indonesia 1.5KM RA1-T",
                "file_pattern": "SEA_indon2km1p5_{%Y%m%dT%H%MZ}.nc"
            },
            {
                "name": "Malaysia 1.5KM RA1-T",
                "file_pattern": "SEA_mal2km1p5_{%Y%m%dT%H%MZ}.nc"
            },
            {
                "name": "Philipines 1.5KM RA1-T",
                "file_pattern": "SEA_phi2km1p5_{%Y%m%dT%H%MZ}.nc"
            }
        ]
    }


def days_ago(days):
    """Helper method to select time N days ago"""
    return (dt.datetime.now() - dt.timedelta(days=days)).replace(second=0, microsecond=0)


@forest.util.timer
def main(bokeh_id):
    '''Two-model bokeh application main program'''
    env = parse_environment(os.environ)
    if env.download_data:
        file_loader = forest.aws.S3Bucket(server_address='https://s3.eu-west-2.amazonaws.com',
                                          bucket_name='stephen-sea-public-london',
                                          download_directory=env.download_directory)
        user_feedback_directory = os.path.join(download_directory, 'user_feedback')
    else:
        # FUSE mounted file system
        file_loader = forest.aws.S3Mount(env.mount_directory)
        user_feedback_directory = os.path.join(env.mount_directory, 'user_feedback')

    plot_descriptions = {
        forest.data.N1280_GA6_KEY: 'N1280 GA6 LAM Model',
        forest.data.KM4P4_RA1T_KEY: 'SE Asia 4.4KM RA1-T ',
        forest.data.KM1P5_INDO_RA1T_KEY: 'Indonesia 1.5KM RA1-T',
        forest.data.KM1P5_MAL_RA1T_KEY: 'Malaysia 1.5KM RA1-T',
        forest.data.KM1P5_PHI_RA1T_KEY: 'Philipines 1.5KM RA1-T',
    }

    # Stash section and items for each variable
    var_lookups = {
        forest.data.N1280_GA6_KEY: forest.stash_codes("ga6"),
        forest.data.KM4P4_RA1T_KEY: forest.stash_codes("ra1t"),
        forest.data.KM1P5_INDO_RA1T_KEY: forest.stash_codes("ra1t"),
        forest.data.KM1P5_MAL_RA1T_KEY: forest.stash_codes("ra1t"),
        forest.data.KM1P5_PHI_RA1T_KEY: forest.stash_codes("ra1t")
    }
    model_run_times = forest.data.get_model_run_times(env.start_date,
                                                      forest.data.NUM_DATA_DAYS,
                                                      forest.data.MODEL_RUN_PERIOD)
    datasets = forest.data.get_available_datasets(file_loader,
                                                  model_run_times,
                                                  var_lookups)
    try:
        init_fcast_time = list(datasets.keys())[-1]
    except IndexError:
        init_fcast_time = None
    print("initial forecast time:", init_fcast_time)
    print(datasets)

    if init_fcast_time is None:
        layout1 = forest.util.load_error_page()
        bokeh.plotting.curdoc().add_root(layout1)
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

    # Create regions
    region_dict = forest.util.SEA_REGION_DICT

    # initial selected point is approximately Jakarta, Indonesia
    selected_point = (-6, 103)

    # Setup and display plots
    plot_opts = forest.util.create_colour_opts(list(plot_type_time_lookups.keys()))

    init_data_time_index = 1
    init_var = 'precipitation'

    south_east_asia_region = 'se_asia'
    init_model_left = forest.data.N1280_GA6_KEY
    init_model_right = forest.data.KM4P4_RA1T_KEY
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    available_times = forest.data.get_available_times(datasets[init_fcast_time],
                                                      plot_type_time_lookups[init_var])
    init_data_time = available_times[init_data_time_index]
    num_times = available_times.shape[0]

    bokeh_figure = bokeh.plotting.figure(active_inspect=None,
                                         match_aspect=True,
                                         title_location="below")
    forest.plot.add_x_axes(bokeh_figure, "above")
    forest.plot.add_y_axes(bokeh_figure, "right")
    bokeh_figure.toolbar.logo = None
    bokeh_figure.toolbar_location = "below"

    # Add cartopy coastline to bokeh figure
    region = region_dict[south_east_asia_region]
    y_start = region[0]
    y_end = region[1]
    x_start = region[2]
    x_end = region[3]
    extent = (x_start, x_end, y_start, y_end)
    forest.plot.add_coastlines(bokeh_figure, extent)
    forest.plot.add_borders(bokeh_figure, extent)

    # Set up plots
    forest_datasets = datasets[init_fcast_time]
    plot_obj_left = forest.plot.ForestPlot(forest_datasets,
                                           plot_descriptions,
                                           plot_opts,
                                           'plot_left' + bokeh_id,
                                           init_var,
                                           init_model_left,
                                           south_east_asia_region,
                                           region_dict,
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
                                            south_east_asia_region,
                                            region_dict,
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
    header = bokeh.layouts.column(
             bokeh.layouts.row(forest_controller.model_run_drop_down,
                               forest_controller.time_previous_button,
                               forest_controller.time_next_button),
             bokeh.layouts.row(forest_controller.left_model_drop_down,
                               forest_controller.right_model_drop_down,
                               forest_controller.left_right_toggle),
             bokeh.layouts.row(forest_controller.model_variable_drop_down,
                               forest_controller.region_drop_down),
             css_classes=["fst-head"])
    bokeh_figure.name = "figure"
    colorbar_widget.name = "colorbar"
    # footer = bokeh.layouts.column(
    #     feedback_controller.uf_vis_toggle,
    #     feedback_controller.uf_vis_layout,
    #     css_classes=["fst-foot"])
    roots = [header,
             bokeh_figure,
             colorbar_widget]
    #         footer]
    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'
    if bokeh_mode == 'server':
        document = bokeh.plotting.curdoc()
        for root in roots:
            document.add_root(root)
    elif bokeh_mode == 'cli':
        root = bokeh.layouts.column(*roots)
        bokeh.io.show(root)
    bokeh.plotting.curdoc().title = 'Two model comparison'


if __name__ == '__main__' or __name__.startswith("bk"):
    main(__name__)
