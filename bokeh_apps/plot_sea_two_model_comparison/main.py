import os
import copy
import numpy
import bokeh.io
import bokeh.layouts 
import bokeh.models.widgets
import bokeh.plotting
import matplotlib
matplotlib.use('agg')

import iris

import forest.util
import forest.plot
import forest.control
import forest.data

iris.FUTURE.netcdf_promote = True

try:
    get_ipython
    is_notbook = True
except:
    is_notebook = False

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


def get_available_datasets(s3_base,
                           s3_local_base,
                           use_s3_mount,
                           base_path_local,
                           do_download,
                           dataset_template):
    
    '''
    
    '''
    
    fcast_dt_list, fcast_dt_str_list = forest.util.get_model_run_times(7)

    fcast_time_list = []
    datasets = {}
    for fct,fct_str in zip(fcast_dt_list, fcast_dt_str_list):
        
        fct_data_dict = copy.deepcopy(dict(dataset_template))
        model_run_data_present = True
        for ds_name in dataset_template.keys():
            fname1 = 'SEA_{conf}_{fct}.nc'.format(conf=ds_name, fct=fct_str)
            fct_data_dict[ds_name]['data'] = forest.data.ForestDataset(ds_name,
                                                                       fname1,
                                                                       s3_base,
                                                                       s3_local_base,
                                                                       use_s3_mount,
                                                                       base_path_local,
                                                                       do_download,
                                                                       dataset_template[ds_name]['var_lookup'],
                                                                       )
            model_run_data_present = model_run_data_present and fct_data_dict[ds_name]['data'].check_data()
        # include forecast if all configs are present
        #TODO: reconsider data structure to allow for some model configs at different times to be present
        if model_run_data_present:
            datasets[fct_str] = fct_data_dict
            fcast_time_list += [fct_str]
            
    # select most recent available forecast
    fcast_time = fcast_time_list[-1]
    return fcast_time, datasets


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
    s3_local_base = os.path.join(os.sep,'s3',bucket_name, 'model_data')
    base_path_local = os.path.expanduser('~/SEA_data/model_data/')
    use_s3_mount = False
    do_download = True

    init_fcast_time, datasets = get_available_datasets(s3_base,
                                                       s3_local_base,
                                                       use_s3_mount,
                                                       base_path_local,
                                                       do_download,
                                                       dataset_template)

    print('Most recent dataset available is {0}, forecast time selected for display.'.format(init_fcast_time))

    # import pdb
    # pdb.set_trace()

    #set up datasets dictionary

    plot_names = ['precipitation',
                'air_temperature',
                'wind_vectors',
                    'wind_mslp',
                'wind_streams',
                'mslp',
                'cloud_fraction',
                #'blank',
                ]



    # create regions
    region_dict = forest.util.SEA_REGION_DICT

    #Setup and display plots
    plot_opts = forest.util.create_colour_opts(plot_names)

    init_data_time = 4
    init_var = plot_names[0] #blank
    init_region = 'se_asia'
    init_model_left = forest.data.N1280_GA6_KEY # KM4P4_RA1T_KEY
    init_model_right = forest.data.KM4P4_RA1T_KEY # N1280_GA6_KEY

    #Set up plots
    plot_obj_left = forest.plot.ForestPlot(datasets[init_fcast_time],
                            plot_opts,
                            'plot_left' + bokeh_id,
                            init_var,
                            init_model_left,
                            init_region,
                            region_dict,
                            forest.data.UNIT_DICT,
                            )

    plot_obj_left.current_time = init_data_time
    bokeh_img_left = plot_obj_left.create_plot()
    colorbar_left = plot_obj_left.create_colorbar_widget()
    stats_left = plot_obj_left.create_stats_widget()

    plot_obj_right = forest.plot.ForestPlot(datasets[init_fcast_time],
                        plot_opts,
                        'plot_right' + bokeh_id,
                        init_var,
                        init_model_right,
                        init_region,
                        region_dict,
                        forest.data.UNIT_DICT,
                        )


    plot_obj_right.current_time = init_data_time
    bokeh_img_right = plot_obj_right.create_plot()
    colorbar_right = plot_obj_right.create_colorbar_widget()
    stats_right = plot_obj_right.create_stats_widget()

    plot_obj_right.link_axes_to_other_plot(plot_obj_left)

    num_times = datasets[init_fcast_time][forest.data.N1280_GA6_KEY]['data'].get_data('precipitation').shape[0]
    for ds_name in datasets[init_fcast_time]:
        num_times = min(num_times, datasets[init_fcast_time][ds_name]['data'].get_data('precipitation').shape[0])
    bokeh_doc = bokeh.plotting.curdoc()

    # Set up GUI controller class
    control1 = forest.control.ForestController(init_data_time,
                                               num_times,
                                               datasets[init_fcast_time],
                                               plot_names,
                                               [plot_obj_left, plot_obj_right],
                                               [bokeh_img_left, bokeh_img_right],
                                               [colorbar_left, colorbar_right],
                                               [stats_left, stats_right],
                                               region_dict,
                                               bokeh_doc,
                                               )

    add_main_plot(control1.main_layout, bokeh_doc)

    bokeh_doc.title = 'Two model comparison'    

main(__name__)