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


def main(bokeh_id):
    '''Two-model bokeh application main program'''

    download_data = os.getenv("FOREST_DOWNLOAD_DATA", default="False").upper() == "TRUE"

    if download_data:
        download_directory = os.environ.get('LOCAL_ROOT', os.path.expanduser("~/SEA_data/"))
        file_loader = forest.aws.S3Bucket(server_address='https://s3.eu-west-2.amazonaws.com',
                                          bucket_name='stephen-sea-public-london',
                                          download_directory=download_directory)
        user_feedback_directory = os.path.join(download_directory, 'user_feedback')
    else:
        # FUSE mounted file system
        s3_root = os.getenv("S3_ROOT", os.path.expanduser("~/s3/"))
        mount_directory = os.path.join(s3_root, 'stephen-sea-public-london')
        file_loader = forest.aws.S3Mount(mount_directory)
        user_feedback_directory = os.path.join(mount_directory, 'user_feedback')

    # Setup datasets. Data is not loaded until requested for plotting.
    dataset_template = {
        forest.data.N1280_GA6_KEY: {'data_type_name': 'N1280 GA6 LAM Model',
                                    'config_id': forest.data.GA6_CONF_ID},
        forest.data.KM4P4_RA1T_KEY: {'data_type_name': 'SE Asia 4.4KM RA1-T ',
                                     'config_id': forest.data.RA1T_CONF_ID},
        forest.data.KM1P5_INDO_RA1T_KEY: {'data_type_name': 'Indonesia 1.5KM RA1-T',
                                          'config_id': forest.data.RA1T_CONF_ID},
        forest.data.KM1P5_MAL_RA1T_KEY: {'data_type_name': 'Malaysia 1.5KM RA1-T',
                                         'config_id': forest.data.RA1T_CONF_ID},
        forest.data.KM1P5_PHI_RA1T_KEY: {'data_type_name': 'Philipines 1.5KM RA1-T',
                                         'config_id': forest.data.RA1T_CONF_ID},
    }
    for ds_name in dataset_template.keys():
        dataset_template[ds_name]['var_lookup'] = forest.data.get_var_lookup(dataset_template[ds_name]['config_id'])

    init_fcast_time, datasets = \
        forest.data.get_available_datasets(file_loader,
                                           dataset_template,
                                           forest.data.NUM_DATA_DAYS,
                                           forest.data.NUM_DATA_DAYS,
                                           forest.data.MODEL_RUN_PERIOD,
                                           )

    print("initial forecast time:", init_fcast_time)
    if init_fcast_time is None:
        layout1 = forest.util.load_error_page()
        bokeh.plotting.curdoc().add_root(layout1)
        return

    print('Most recent dataset available is {0}, forecast time selected for display.'.format(init_fcast_time))

    plot_type_time_lookups = \
        {'precipitation': 'precipitation',
         'air_temperature': 'air_temperature',
         'wind_vectors': 'x_wind',
         'wind_mslp': 'x_wind',
         'wind_streams': 'x_wind',
         'mslp': 'mslp',
         'cloud_fraction': 'cloud_fraction',
         }

    for var1 in forest.data.PRECIP_ACCUM_VARS:
        plot_type_time_lookups.update({var1: var1})

    # Create regions
    region_dict = forest.util.SEA_REGION_DICT

    # initial selected point is approximately Jakarta, Indonesia
    selected_point = (-6, 103)

    # Setup and display plots
    plot_opts = forest.util.create_colour_opts(list(plot_type_time_lookups.keys()))

    init_data_time_index = 1
    init_var = 'precipitation'

    south_east_asia_region = 'se_asia'
    init_model_left = forest.data.N1280_GA6_KEY  # KM4P4_RA1T_KEY
    init_model_right = forest.data.KM4P4_RA1T_KEY  # N1280_GA6_KEY
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    available_times = forest.data.get_available_times(datasets[init_fcast_time],
                                                      plot_type_time_lookups[init_var])
    init_data_time = available_times[init_data_time_index]
    num_times = available_times.shape[0]

    bokeh_figure = bokeh.plotting.figure(toolbar_location="above",
                                         active_inspect=None,
                                         match_aspect=True)
    forest.plot.add_x_axes(bokeh_figure, "above")
    forest.plot.add_y_axes(bokeh_figure, "right")
    bokeh_figure.toolbar.logo = None
    bokeh_figure.toolbar_location = None

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
    plot_obj_left = forest.plot.ForestPlot(datasets[init_fcast_time],
                                           init_fcast_time,
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

    plot_obj_right = forest.plot.ForestPlot(datasets[init_fcast_time],
                                            init_fcast_time,
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
    feedback_controller = forest.FeedbackController(user_feedback_directory,
                                                    bokeh_id)
    forest_controller = forest.ForestController(init_var,
                                                init_data_time_index,
                                                datasets,
                                                init_fcast_time,
                                                plot_type_time_lookups,
                                                [plot_obj_left, plot_obj_right],
                                                bokeh_figure,
                                                region_dict)

    # Attach bokeh layout to current document
    navbar = bokeh.layouts.layout([
        [forest_controller.model_run_drop_down,
         forest_controller.time_previous_button,
         forest_controller.time_next_button],
        [forest_controller.left_model_drop_down,
         forest_controller.right_model_drop_down,
         forest_controller.model_variable_drop_down,
         forest_controller.region_drop_down],
        [forest_controller.left_right_toggle]],
        css_classes=["forest-nav"])
    footer = bokeh.layouts.column(
        colorbar_widget,
        feedback_controller.uf_vis_toggle,
        feedback_controller.uf_vis_layout
    )
    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'
    if bokeh_mode == 'server':
        print("server mode")
        document = bokeh.plotting.curdoc()
        document.add_root(navbar)
        document.add_root(bokeh_figure)
        document.add_root(footer)
    elif bokeh_mode == 'cli':
        root = bokeh.layouts.column(navbar,
                                    bokeh_figure,
                                    footer)
        bokeh.io.show(root)
    bokeh.plotting.curdoc().title = 'Two model comparison'


if __name__ == '__main__' or __name__.startswith("bk"):
    main(__name__)
