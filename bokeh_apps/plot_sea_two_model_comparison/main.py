import os

import warnings
warnings.filterwarnings('ignore')

import bokeh.io
import bokeh.plotting

import matplotlib
matplotlib.use('agg')

import forest.util
import forest.plot
import forest.control
import forest.data

def add_main_plot(main_layout, bokeh_doc):
    
    '''
    
    '''
    
    print('finished creating, executing document add callback')

    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'    
        
    if bokeh_mode == 'server':
        bokeh_doc.add_root(main_layout)
    elif bokeh_mode == 'cli':
        bokeh.io.show(main_layout)


# Extract and Load
bucket_name = 'stephen-sea-public-london'
server_address = 'https://s3.eu-west-2.amazonaws.com'



@forest.util.timer
def main(bokeh_id):

    '''
    
    '''

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

    s3_base = '{server}/{bucket}/model_data/'.format(server=server_address,
                                                    bucket=bucket_name)

    try:
        s3_root = os.environ['S3_ROOT']
    except KeyError:
        s3_root = os.path.expanduser('~/s3')
    s3_local_base = os.path.join(s3_root,
                                 bucket_name,
                                 'model_data')
    try:
        local_root = os.environ['LOCAL_ROOT']
    except KeyError:
        local_root = os.path.expanduser('~/SEA_data')
    base_path_local = os.path.join(local_root, 'model_data')


    use_s3_mount = True
    do_download = False

    init_fcast_time, datasets = \
        forest.data.get_available_datasets(s3_base,
                                           s3_local_base,
                                           use_s3_mount,
                                           base_path_local,
                                           do_download,
                                           dataset_template,
                                           forest.data.NUM_DATA_DAYS,
                                           forest.data.NUM_DATA_DAYS,
                                           forest.data.MODEL_RUN_PERIOD,
                                           )

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
        plot_type_time_lookups.update({var1:var1})

    bokeh_doc = bokeh.plotting.curdoc()

    #Create regions
    region_dict = forest.util.SEA_REGION_DICT

    #Setup and display plots
    plot_opts = forest.util.create_colour_opts(list(plot_type_time_lookups.keys()))

    init_data_time_index = 1
    init_var = 'precipitation'

    init_region = 'se_asia'
    init_model_left = forest.data.N1280_GA6_KEY # KM4P4_RA1T_KEY
    init_model_right = forest.data.KM4P4_RA1T_KEY # N1280_GA6_KEY
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    available_times = \
        forest.data.get_available_times(datasets[init_fcast_time],
                                        plot_type_time_lookups[init_var])
    init_data_time = available_times[init_data_time_index]
    num_times = available_times.shape[0]

    # Set up plots
    plot_obj_left = forest.plot.ForestPlot(datasets[init_fcast_time],
                                           init_fcast_time,
                                           plot_opts,
                                           'plot_left' + bokeh_id,
                                           init_var,
                                           init_model_left,
                                           init_region,
                                           region_dict,
                                           forest.data.UNIT_DICT,
                                           forest.data.UNIT_DICT_DISPLAY,
                                           app_path,
                                           init_data_time,
                                           )

    bokeh_img_left = plot_obj_left.create_plot()
    stats_left = plot_obj_left.create_stats_widget()

    plot_obj_right = forest.plot.ForestPlot(datasets[init_fcast_time],
                                            init_fcast_time,
                                            plot_opts,
                                            'plot_right' + bokeh_id,
                                            init_var,
                                            init_model_right,
                                            init_region,
                                            region_dict,
                                            forest.data.UNIT_DICT,
                                            forest.data.UNIT_DICT_DISPLAY,
                                            app_path,
                                            init_data_time,
                                            )

    bokeh_img_right = plot_obj_right.create_plot()
    stats_right = plot_obj_right.create_stats_widget()

    colorbar_widget = plot_obj_left.create_colorbar_widget()

    plot_obj_right.link_axes_to_other_plot(plot_obj_left)

    s3_local_base_feedback = \
        os.path.join(s3_root,
                     bucket_name,
                     'user_feedback')



    # Set up GUI controller class
    control1 = forest.control.ForestController(init_var,
                                               init_data_time_index,
                                               datasets,
                                               init_fcast_time,
                                               plot_type_time_lookups,
                                               [plot_obj_left, plot_obj_right],
                                               [bokeh_img_left, bokeh_img_right],
                                               colorbar_widget,
                                               [stats_left, stats_right],
                                               region_dict,
                                               bokeh_doc,
                                               s3_local_base_feedback,
                                               bokeh_id,
                                               )

    add_main_plot(control1.main_layout, bokeh_doc)

    bokeh_doc.title = 'Two model comparison'    

main(__name__)