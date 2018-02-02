### SE Asia Model and GPM IMERG Matplotlib example Bokeh app script

# This script demonstrates creating plots of model rainfall data and GPM IMERG
#  data for SE Asia using the Matplotlib plotting library to provide images to
#  a Bokeh Server App.

## Setup notebook
# Do module imports

import os
import datetime
import math

import numpy

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



## Extract
# Extract data from S3. The data can either be downloaded in full before 
#  loading, or downloaded on demand using the /s3 filemount. This is 
#  controlled by the do_download flag.

def main(bokeh_id):

    bucket_name = 'stephen-sea-public-london'
    server_address = 'https://s3.eu-west-2.amazonaws.com'

    fcast_time = '20180109T0000Z'
    fcast_time_obj =  datetime.datetime.strptime(fcast_time, '%Y%m%dT%H%MZ')

    N1280_GA6_KEY = 'n1280_ga6'
    KM4P4_RA1T_KEY = 'km4p4_ra1t'
    KM1P5_INDO_RA1T_KEY = 'indon2km1p5_ra1t'
    KM1P5_MAL_RA1T_KEY = 'mal2km1p5_ra1t'
    KM1P5_PHI_RA1T_KEY = 'phi2km1p5_ra1t'
    GPM_IMERG_EARLY_KEY = 'gpm_imerg_early'
    GPM_IMERG_LATE_KEY = 'gpm_imerg_late'

    # The datasets dictionary is the main container for the forecast data and
    #  associated meta data. It is stored as a dictionary of dictionaries.
    # The first layer of indexing selects the region/config, for example N1280 GA6
    #  global or 1.5km Indonesia RA1-T. Each of these will be populated with a
    #  cube for each of the available variables as well as asociated metadata such
    #  as model name, file paths etc.

    datasets = {N1280_GA6_KEY: {'data_type_name': 'N1280 GA6 LAM Model'},
                KM4P4_RA1T_KEY: {'data_type_name': 'SE Asia 4.4KM RA1-T '},
                KM1P5_INDO_RA1T_KEY: {'data_type_name': 'Indonesia 1.5KM RA1-T'},
                KM1P5_MAL_RA1T_KEY: {'data_type_name': 'Malaysia 1.5KM RA1-T'},
                KM1P5_PHI_RA1T_KEY: {'data_type_name': 'Philipines 1.5KM RA1-T'},
                GPM_IMERG_EARLY_KEY: {'data_type_name': 'GPM IMERG Early'},
                GPM_IMERG_LATE_KEY: {'data_type_name': 'GPM IMERG Late'},
                }
    model_datasets = {N1280_GA6_KEY: datasets[N1280_GA6_KEY],
                      KM4P4_RA1T_KEY: datasets[KM4P4_RA1T_KEY],
                      KM1P5_INDO_RA1T_KEY: datasets[KM1P5_INDO_RA1T_KEY],
                      KM1P5_MAL_RA1T_KEY: datasets[KM1P5_MAL_RA1T_KEY],
                      KM1P5_PHI_RA1T_KEY: datasets[KM1P5_PHI_RA1T_KEY],
                      }

    gpm_datasets = {GPM_IMERG_EARLY_KEY: datasets[GPM_IMERG_EARLY_KEY],
                    GPM_IMERG_LATE_KEY: datasets[GPM_IMERG_LATE_KEY],
                    }
    GPM_TYPE_KEY = 'gpm_type'
    gpm_datasets[GPM_IMERG_EARLY_KEY][GPM_TYPE_KEY] = 'early'
    gpm_datasets[GPM_IMERG_LATE_KEY][GPM_TYPE_KEY] = 'late'

    datasets[N1280_GA6_KEY]['var_lookup'] = {'precipitation': 'precipitation_flux'}
    datasets[N1280_GA6_KEY]['units'] = forest.data.UNIT_DICT
    datasets[KM4P4_RA1T_KEY]['var_lookup'] = {'precipitation': 'stratiform_rainfall_rate'}
    datasets[KM4P4_RA1T_KEY]['units'] = {'precipitation': 'kg-m-2-hour^-1'}

    datasets[KM1P5_INDO_RA1T_KEY]['units'] = forest.data.UNIT_DICT
    datasets[KM1P5_MAL_RA1T_KEY]['units'] = forest.data.UNIT_DICT
    datasets[KM1P5_PHI_RA1T_KEY]['units'] = forest.data.UNIT_DICT

    datasets[KM1P5_INDO_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
    datasets[KM1P5_MAL_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])
    datasets[KM1P5_PHI_RA1T_KEY]['var_lookup'] = dict(datasets[KM4P4_RA1T_KEY]['var_lookup'])

    use_s3_mount = False
    do_download = True
    use_jh_paths = True
    base_dir = os.path.expanduser('~/SEA_data')

    base_path_local_model = os.path.join(base_dir, 'model_data')
    base_path_local_gpm = os.path.join(base_dir, 'gpm_imerg') + '/'

    s3_base_str_model = '{server}/{bucket}/model_data/'
    s3_base_model = s3_base_str_model.format(server=server_address, bucket=bucket_name)
    s3_local_base_model = os.path.join(os.sep,'s3',bucket_name, 'model_data')

    for ds_name in model_datasets.keys():
        fname1 = 'SEA_{conf}_{fct}.nc'.format(conf=ds_name,
                                              fct=fcast_time)
        datasets[ds_name]['data'] = forest.data.ForestDataset(ds_name,
                                                              fname1,
                                                              s3_base_model,
                                                              s3_local_base_model,
                                                              use_s3_mount,
                                                              base_path_local_model,
                                                              do_download,
                                                              datasets[ds_name]['var_lookup']
                                                              )

    s3_base_str_gpm = '{server}/{bucket}/gpm_imerg/'
    s3_base_gpm = s3_base_str_gpm.format(server=server_address, bucket=bucket_name)
    s3_local_base_gpm = os.path.join(os.sep,'s3',bucket_name, 'gpm_imerg')

    for ds_name in gpm_datasets.keys():
        imerg_type = gpm_datasets[ds_name][GPM_TYPE_KEY]
        fname_fmt = 'gpm_imerg_NRT{im}_V05B_{datetime}_sea_only.nc'
        times_list = [(fcast_time_obj + datetime.timedelta(days=dd)).strftime('%Y%m%d') for dd in range(4)]
        fnames_list = [fname_fmt.format(im=imerg_type, datetime=dt_str) for dt_str in times_list]


        datasets[ds_name]['data'] = model_gpm_data.GpmDataset(ds_name,
                                                              fnames_list,
                                                              s3_base_gpm,
                                                              s3_local_base_gpm,
                                                              use_s3_mount,
                                                              base_path_local_gpm,
                                                              do_download,
                                                              times_list,
                                                              )

    plot_names = ['precipitation']

    var_names = ['precipitation']



    ## Setup plots
    # Set up plot colours and geoviews datasets before creating and showing plots

    # create regions dict, for selecting which map region to display
    region_dict = forest.util.SEA_REGION_DICT

    plot_opts = forest.util.create_colour_opts(plot_names)

    # Set the initial values to be plotted
    init_time = 12
    init_var = 'precipitation'
    init_region = 'se_asia'
    init_model_left = KM4P4_RA1T_KEY
    init_model_right = GPM_IMERG_EARLY_KEY


    ## Display plots

    plot_obj_left = forest.plot.ForestPlot(datasets,
                                           plot_opts,
                                           'plot_left' + bokeh_id,
                                           init_var,
                                           init_model_left,
                                           init_region,
                                           region_dict,
                                           forest.data.UNIT_DICT,
                                           )
    # Create a plot object for the left model display

    plot_obj_left.current_time = init_time
    bokeh_img_left = plot_obj_left.create_plot()

    # Create a plot object for the right model display
    plot_obj_right = forest.plot.ForestPlot(datasets,
                                            plot_opts,
                                            'plot_right' + bokeh_id,
                                            init_var,
                                            init_model_right,
                                            init_region,
                                            region_dict,
                                            forest.data.UNIT_DICT,
                                            )

    plot_obj_right.current_time = init_time
    bokeh_img_right = plot_obj_right.create_plot()

    plot_obj_right.link_axes_to_other_plot(plot_obj_left)


    num_times = 3 * datasets[GPM_IMERG_LATE_KEY]['data'].get_data('precipitation').shape[0]

    control1 = model_gpm_control.ModelGpmControl(datasets,
                                                 init_time,
                                                 num_times,
                                                 [plot_obj_left, plot_obj_right],
                                                 [bokeh_img_left, bokeh_img_right], )
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
