'''SE Asia Sim. Im. and Himawari-8 Matplotlib app

 This script creates plots of simulated satellite imagery and 
  Himawari-8 imagery for SE Asia using the Matplotlib plotting 
  library to provide images to a Bokeh Server app.

'''

import os
import datetime
import copy

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('agg')
import iris
iris.FUTURE.netcdf_promote = True
import bokeh.plotting

import forest.util
import forest.plot

import simim_sat_control
import simim_sat_data

def main(bokeh_id):
    
    '''Main function of app
    
    '''



    # Extract data from S3. The data can either be downloaded in full before
    #  loading, or downloaded on demand using the /s3 filemount. This is 
    #  controlled by the do_download flag.

    bucket_name = 'stephen-sea-public-london'
    server_address = 'https://s3.eu-west-2.amazonaws.com'
    
    do_download = False
    use_s3_mount = True
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

    SIMIM_KEY = simim_sat_data.SIMIM_KEY
    HIMAWARI8_KEY = 'himawari-8'

    # The datasets dictionary is the main container for the Sim. Im./Himawari-8
    #  data and associated meta data. It is stored as a dictionary of dictionaries.
    # The first layer of indexing selects the data type, for example Simulated 
    #  Imagery or Himawari-8 imagery. Each of these will be populated with a cube
    #  or data image array for each of the available wavelengths as well as 
    #  asociated metadata such as file paths etc.

    datasets_template = {simim_sat_data.SIMIM_KEY:{'data_type_name': 'Simulated Imagery'},
                simim_sat_data.HIMAWARI8_KEY:{'data_type_name': 'Himawari-8 Imagery'},
               }

    s3_base_str = '{server}/{bucket}/{data_type}/'

    # Himawari-8 imagery dict population
    s3_base_sat = s3_base_str.format(server=server_address,
                                     bucket=bucket_name,
                                     data_type=simim_sat_data.HIMAWARI8_KEY)
    s3_local_base_sat = os.path.join(s3_local_mnt, simim_sat_data.HIMAWARI8_KEY)
    base_path_local_sat = os.path.join(base_dir, simim_sat_data.HIMAWARI8_KEY)

    # Simulated Imagery dict population
    s3_base_simim = s3_base_str.format(server=server_address,
                                       bucket=bucket_name,
                                       data_type = simim_sat_data.SIMIM_KEY)
    s3_local_base_simim = os.path.join(s3_local_mnt, simim_sat_data.SIMIM_KEY)
    base_path_local_simim= os.path.join(base_dir,simim_sat_data.SIMIM_KEY)

    days_since_period_start = forest.data.MODEL_RUN_DAYS + forest.data.MODEL_OBS_COMP_DAYS
    model_run_times, mrt_strings = \
        forest.data.get_model_run_times(days_since_period_start,
                                        forest.data.MODEL_OBS_COMP_DAYS,
                                        forest.data.MODEL_RUN_PERIOD)

    datasets = {}
    for fcast_time_obj in model_run_times:

        # Set datetime objects and string for controlling data download
        fcast_hour = fcast_time_obj.hour
        fcast_time =  fcast_time_obj.strftime('%Y%m%dT%H%MZ')

        datasets[fcast_time] = copy.deepcopy(datasets_template)
        fnames_list_sat = {}
        for im_type in simim_sat_data.HIMAWARI_KEYS.keys():
            fnames_list_sat[im_type] = \
                ['{im}_{datetime}.jpg'.format(im = simim_sat_data.HIMAWARI_KEYS[im_type],
                                              datetime = (fcast_time_obj + datetime.timedelta(hours = int(vt))).strftime('%Y%m%d%H%M'))
                 for vt in simim_sat_data.DATA_TIMESTEPS[im_type][fcast_hour]]

        datasets[fcast_time][simim_sat_data.HIMAWARI8_KEY]['data'] = \
            simim_sat_data.SatelliteDataset(simim_sat_data.HIMAWARI8_KEY,
                                            fnames_list_sat,
                                            s3_base_sat,
                                            s3_local_base_sat,
                                            use_s3_mount,
                                            base_path_local_sat,
                                            do_download,
                                            )



        simim_fmt_str = 'sea4-{it}_HIM8_{date}_s4{run}_T{time}.nc'
        bt_fnames_list_simim = [simim_fmt_str.format(it='simbt',
                                                     date=fcast_time[:8],
                                                     run=fcast_time[9:11],
                                                     time=vt)
                                for vt in simim_sat_data.DATA_TIMESTEPS['I'][fcast_hour]]
        vis_fnames_list_simim = [simim_fmt_str.format(it='simvis',
                                                      date=fcast_time[:8],
                                                      run=fcast_time[9:11],
                                                      time=vt)
                                 for vt in simim_sat_data.DATA_TIMESTEPS['V'][fcast_hour]]
        fnames_list_simim = bt_fnames_list_simim + vis_fnames_list_simim
        time_list_simim = [None]

        datasets[fcast_time][simim_sat_data.SIMIM_KEY]['data'] = \
            simim_sat_data.SimimDataset(simim_sat_data.SIMIM_KEY,
                                        fnames_list_simim,
                                        s3_base_simim,
                                        s3_local_base_simim,
                                        use_s3_mount,
                                        base_path_local_simim,
                                        do_download,
                                        time_list_simim,
                                        fcast_time_obj)

    ## Setup plots

    plot_options = forest.util.create_satellite_simim_plot_opts()

    # Set the initial values to be plotted
    init_fcast_time_obj = model_run_times[-1]
    init_fcast_time = init_fcast_time_obj.strftime('%Y%m%dT%H%MZ')
    init_time = (init_fcast_time_obj + datetime.timedelta(hours=12)).strftime('%Y%m%d%H%M')
    init_var = 'I'
    init_region = 'se_asia'
    region_dict = forest.util.SEA_REGION_DICT
    app_path = os.path.join(*os.path.dirname(__file__).split('/')[-1:])

    ## Display plots

    # Create a plot object for the left model display
    plot_obj_left = forest.plot.ForestPlot(datasets[init_fcast_time],
                                           init_fcast_time,
                                           plot_options,
                                           'plot_left' + bokeh_id,
                                           init_var,
                                           'simim',
                                           init_region,
                                           region_dict,
                                           simim_sat_data.UNIT_DICT,
                                           simim_sat_data.UNIT_DICT,
                                           app_path,
                                           init_time,
                                           )

    plot_obj_left.current_time = init_time
    colorbar = plot_obj_left.create_colorbar_widget()
    bokeh_img_left = plot_obj_left.create_plot()

    # Create a plot object for the right model display
    plot_obj_right = forest.plot.ForestPlot(datasets[init_fcast_time],
                                            None,
                                            plot_options,
                                            'plot_right' + bokeh_id,
                                            init_var,
                                            simim_sat_data.HIMAWARI8_KEY,
                                            init_region,
                                            region_dict,
                                            simim_sat_data.UNIT_DICT,
                                            simim_sat_data.UNIT_DICT,
                                            app_path,
                                            init_time,
                                            )

    plot_obj_right.current_time = init_time
    bokeh_img_right = plot_obj_right.create_plot()

    plot_obj_right.link_axes_to_other_plot(plot_obj_left)

    plot_list1 = [plot_obj_left, plot_obj_right]
    bokeh_imgs1 = [bokeh_img_left, bokeh_img_right]
    control1 = simim_sat_control.SimimSatControl(datasets, 
                                                 init_time, 
                                                 init_fcast_time,
                                                 init_var,
                                                 plot_list1, 
                                                 bokeh_imgs1,
                                                 colorbar,
                                                )

    try:
        bokeh_mode = os.environ['BOKEH_MODE']
    except:
        bokeh_mode = 'server'    

    if bokeh_mode == 'server':
        bokeh.plotting.curdoc().add_root(control1.main_layout)
    elif bokeh_mode == 'cli':
        bokeh.io.show(control1.main_layout)

    bokeh.plotting.curdoc().title = 'Model simulated imagery vs Himawari-8'    

main(__name__)