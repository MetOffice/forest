''' SE Asia Model and GPM IMERG Matplotlib example Bokeh app script

# This script demonstrates creating plots of model rainfall data and GPM IMERG
#  data for SE Asia using the Matplotlib plotting library to provide images to
#  a Bokeh Server App.

'''

import os
import datetime
import math
import numpy
import dateutil.parser
import copy

import matplotlib
matplotlib.use('agg')
import iris
iris.FUTURE.netcdf_promote = True
import bokeh.plotting

import forest.util
import forest.plot
import forest.data

import model_gpm_control
import model_gpm_data



def main(bokeh_id):

    '''Main app function
     
    '''



    
    # Extract data from S3. The data can either be downloaded in full before 
    #  loading, or downloaded on demand using the /s3 filemount. This is 
    #  controlled by the do_download flag.
    
    bucket_name = 'stephen-sea-public-london'
    server_address = 'https://s3.eu-west-2.amazonaws.com'
    
    GPM_IMERG_EARLY_KEY = 'gpm_imerg_early'
    GPM_IMERG_LATE_KEY = 'gpm_imerg_late'

    # The datasets dictionary is the main container for the forecast data and
    #  associated meta data. It is stored as a dictionary of dictionaries.
    # The first layer of indexing selects the region/config, for example N1280 GA6
    #  global or 1.5km Indonesia RA1-T. Each of these will be populated with a
    #  cube for each of the available variables as well as asociated metadata such
    #  as model name, file paths etc.

    datasets_template = {
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
        GPM_IMERG_EARLY_KEY: {'data_type_name': 'GPM IMERG Early'},
        GPM_IMERG_LATE_KEY: {'data_type_name': 'GPM IMERG Late'},
    }

    model_datasets_template = {forest.data.N1280_GA6_KEY: datasets_template[forest.data.N1280_GA6_KEY],
                      forest.data.KM4P4_RA1T_KEY: datasets_template[forest.data.KM4P4_RA1T_KEY],
                      forest.data.KM1P5_INDO_RA1T_KEY: datasets_template[forest.data.KM1P5_INDO_RA1T_KEY],
                      forest.data.KM1P5_MAL_RA1T_KEY: datasets_template[forest.data.KM1P5_MAL_RA1T_KEY],
                      forest.data.KM1P5_PHI_RA1T_KEY: datasets_template[forest.data.KM1P5_PHI_RA1T_KEY],
                      }

    gpm_datasets_template = {GPM_IMERG_EARLY_KEY: datasets_template[GPM_IMERG_EARLY_KEY],
                    GPM_IMERG_LATE_KEY: datasets_template[GPM_IMERG_LATE_KEY],
                    }
    
    GPM_TYPE_KEY = 'gpm_type'
    datasets_template[GPM_IMERG_EARLY_KEY][GPM_TYPE_KEY] = 'early'
    datasets_template[GPM_IMERG_LATE_KEY][GPM_TYPE_KEY] = 'late'

    for ds_name in model_datasets_template.keys():
        model_datasets_template[ds_name]['var_lookup'] = forest.data.get_var_lookup(model_datasets_template[ds_name]['config_id'])

    use_s3_mount = True
    do_download = False
    use_jh_paths = True

    try:
        s3_root = os.environ['S3_ROOT']
    except KeyError:
        s3_root = os.path.expanduser('~/s3')
    s3_local_mnt = os.path.join(s3_root,
                                bucket_name,
                                )
    try:
        base_dir = os.environ['LOCAL_ROOT']
    except KeyError:
        base_dir = os.path.expanduser('~/SEA_data')

    base_path_local_model = os.path.join(base_dir, 'model_data')
    base_path_local_gpm = os.path.join(base_dir, 'gpm_imerg') + '/'

    s3_base_str_model = '{server}/{bucket}/model_data/'
    s3_base_model = s3_base_str_model.format(server=server_address, bucket=bucket_name)
    s3_local_base_model = os.path.join(s3_local_mnt,  'model_data')

    # when comparing gpm and model data, we want to compare model runs for which
    # gpm observations are available. Currently only model runs with all corresponding
    # obs data are avaiable for display. Since model runs are 5 days (120 hours) long
    # models run from 8 to 5 days ago are included.
    days_since_gpm_model_period_start=8
    days_in_gpm_model_dataset=3

    init_fcast_time, datasets = \
        forest.data.get_available_datasets(s3_base_model,
                                           s3_local_base_model,
                                           use_s3_mount,
                                           base_path_local_model,
                                           do_download,
                                           model_datasets_template,
                                           days_since_gpm_model_period_start,
                                           days_in_gpm_model_dataset,
                                           forest.data.MODEL_RUN_PERIOD,
                                           )




    s3_base_str_gpm = '{server}/{bucket}/gpm_imerg/'
    s3_base_gpm = s3_base_str_gpm.format(server=server_address, bucket=bucket_name)
    s3_local_base_gpm = os.path.join(s3_local_mnt, 'gpm_imerg')

    # Set datetime objects and string for controlling data download
    # now_time_obj = datetime.datetime.utcnow()
    # data_period_start = now_time_obj - datetime.timedelta(days = forest.data.NUM_DATA_DAYS)
    # fcast_hour = 12*int(now_time_obj.hour/12)
    # fcast_time_obj = data_period_start.replace(hour=fcast_hour, minute=0)
    # fcast_time =  fcast_time_obj.strftime('%Y%m%dT%H%MZ')
    for fcast_time1 in datasets.keys():
        datasets[fcast_time1].update(copy.deepcopy(gpm_datasets_template))
        for ds_name in gpm_datasets_template.keys():
            fcast_time_obj = dateutil.parser.parse(fcast_time1)
            fcast_hour = fcast_time_obj.hour
            imerg_type = gpm_datasets_template[ds_name][GPM_TYPE_KEY]
            fname_fmt = 'gpm_imerg_NRT{im}_V05B_{datetime}_sea_only.nc'
            times_list = [(fcast_time_obj + datetime.timedelta(days=dd)).strftime('%Y%m%d') for dd in range(forest.data.MODEL_RUN_DAYS)]
            fnames_list = [fname_fmt.format(im=imerg_type, datetime=dt_str) for dt_str in times_list]
            datasets[fcast_time1][ds_name]['data'] = \
                model_gpm_data.GpmDataset(ds_name,
                                          fnames_list,
                                          s3_base_gpm,
                                          s3_local_base_gpm,
                                          use_s3_mount,
                                          base_path_local_gpm,
                                          do_download,
                                          times_list,
                                          fcast_hour,
                                          )

    ## Setup plots
    # Set up plot colours and geoviews datasets before creating and showing plots

    # create regions dict, for selecting which map region to display
    region_dict = forest.util.SEA_REGION_DICT

    plot_opts = forest.util.create_colour_opts(['precipitation'])

    # Set the initial values to be plotted
    init_var = 'accum_precip_3hr'
    init_region = 'se_asia'
    init_model_left = forest.data.N1280_GA6_KEY
    init_model_right = GPM_IMERG_EARLY_KEY
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    available_times = forest.data.get_available_times(datasets[init_fcast_time],
                                                      init_var)

    init_time_ix = 4
    init_time = available_times[init_time_ix]


    ## Display plots

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
                                           init_time,
                                           )

    # Create a plot object for the left model display

    plot_obj_left.current_time = init_time
    bokeh_img_left = plot_obj_left.create_plot()
    stats_left = plot_obj_left.create_stats_widget()

    # Create a plot object for the right model display
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
                                            init_time,
                                            )

    plot_obj_right.current_time = init_time
    bokeh_img_right = plot_obj_right.create_plot()
    stats_right = plot_obj_right.create_stats_widget()

    stats_list = [stats_left, stats_right]

    plot_obj_right.link_axes_to_other_plot(plot_obj_left)

    control1 = model_gpm_control.ModelGpmControl(init_var,
                                                 datasets,
                                                 init_time_ix,
                                                 init_fcast_time,
                                                 [plot_obj_left, plot_obj_right],
                                                 [bokeh_img_left, bokeh_img_right],
                                                 stats_list)

    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'

    if bokeh_mode == 'server':
        bokeh.plotting.curdoc().add_root(control1.main_layout)
        bokeh.plotting.curdoc().title = 'Model rainfall vs GPM app'

    elif bokeh_mode == 'cli':
        bokeh.io.show(control1.main_layout)

main(__name__)
