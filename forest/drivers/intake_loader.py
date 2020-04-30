'''
Module for loading data from the Pangeo CMIP6n intake catalogue (usng the intake-esm
loader) created by NCAR.
'''

import functools

try:
    import iris
except ModuleNotFoundError:
    iris = None
    # ReadTheDocs can't import iris

import numpy

try:
    import intake
except ModuleNotFoundError:
    intake = None

import forest.view
from forest import geo, util
from forest.drivers import gridded_forecast

# Location of the Pangeo-CMIP6 intake catalogue file.
URL = 'https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json'


class Dataset:
    def __init__(self, pattern=None, **kwargs):
        self.pattern = pattern

    def navigator(self):
        return Navigator()

    def map_view(self, color_mapper):
        loader = IntakeLoader(self.pattern)
        view = forest.view.UMView(loader, color_mapper)
        view.set_hover_properties(INTAKE_TOOLTIPS, INTAKE_FORMATTERS)
        return view


@functools.lru_cache(maxsize=64)
def _get_intake_vars(
        experiment_id,
        table_id,
        grid_label,
        institution_id,
        member_id):
    """
    Helper function to get a list of variables for this particular combination
    of parameters. Function is cahced to reduce remote queries.
    """
    collection = intake.open_esm_datastore(URL)
    cat = collection.search(
        experiment_id=experiment_id,
        table_id=table_id,
        grid_label=grid_label,
        institution_id=institution_id,
        member_id=member_id,
    )
    var_list = cat.unique('variable_id')['variable_id']['values']
    return var_list


@functools.lru_cache(maxsize=16)
def _load_from_intake(
        experiment_id,
        table_id,
        grid_label,
        variable_id,
        institution_id,
        activity_id,
        member_id):
    """
    Load data from the pangeo CMIP6 intake catalogue.The arguments relate to
    the CMIP6 parameters of a dataset. The CMIP6 reference is the ESGF servers
    which can be accessed here:
    https://esgf-index1.ceda.ac.uk/search/cmip6-ceda/
    Function is cahced to reduce remote queries.
    """
    collection = intake.open_esm_datastore(URL)
    cat = collection.search(
        experiment_id=experiment_id,
        table_id=table_id,
        grid_label=grid_label,
        institution_id=institution_id,
        member_id=member_id,
        variable_id=variable_id)
    dset_dict = cat.to_dataset_dict(
        zarr_kwargs={'consolidated': True, 'decode_times': False},
        cdf_kwargs={'chunks': {}, 'decode_times': False})

    # The search should have produced a dictionary with only 1 item, so
    # get that item and get a cube from it.
    ds_label, xr = dset_dict.popitem()
    cube = xr[variable_id].to_iris()
    coord_names = [c1.name() for c1 in cube.coords()]
    if 'air_pressure' in coord_names:
        cube.coord('air_pressure').convert_units('hPa')
    return iris.util.squeeze(cube)  # drop member dimension

INTAKE_TOOLTIPS = [
    ("Name", "@name"),
    ("Value", "@image @units"),
    ('Valid', '@valid{%F %H:%M}'),
    ("Level", "@level"),
    ("Experiment", "@experiment"),
    ("Institution", "@institution"),
    ("Member", "@memberid"),
    ('Variable', "@variableid"), ]
INTAKE_FORMATTERS = {
    '@valid': 'datetime',
}


@functools.lru_cache(maxsize=16)
def _get_bokeh_image(cube,
                     experiment_id,
                     variable_id,
                     institution_id,
                     initial_time,
                     member_id,
                     selected_time,
                     pressure,
                     ):
    """
    A helper function to do  the creation of the image dict required by bokeh.
    This includes downloading the actual data required for the current view, so
    this function is cached to reduce remote queries.
    """

    def time_comp(select_time, time_cell):  #
        data_time = util.to_datetime(time_cell.point)
        if abs((select_time - data_time).days) < 2:
            return True
        return False

    def lat_filter(lat):
        """
        Due to the way the current projection of gridded data works, the poles are
        not well handled, resulting in NaNs if we use the full range of latitudes.
        The current hack is to chop off latitude greater than 85 degrees north and
        south. Given the importance of data at the poles in climate change research,
        we will need to fix this in future.
        """
        return -85.0 < lat < 85.0

    def pressure_select(select_pressure, data_pressure):
        return abs(select_pressure - data_pressure.point) < 1.0

    if cube is None or initial_time is None:
        data = gridded_forecast.empty_image()
    else:
        constraint_dict = {'time': functools.partial(time_comp,
                                                     selected_time),
                           'latitude': lat_filter,
                           }
        coord_names = [c1.name() for c1 in cube.coords()]
        if 'air_pressure' in coord_names:
            constraint_dict['air_pressure'] = functools.partial(
                pressure_select,
                pressure,
            )
        cube_cropped = cube.extract(iris.Constraint(**constraint_dict))
        lat_pts = cube_cropped.coord('latitude').points
        long_pts = cube_cropped.coord('longitude').points - 180.0
        cube_data_cropped = cube_cropped.data
        cube_width = int(cube_data_cropped.shape[1] / 2)
        cube_data_cropped = numpy.concatenate(
            [cube_data_cropped[:, cube_width:],
             cube_data_cropped[:, :cube_width]], axis=1)

        data = geo.stretch_image(long_pts, lat_pts, cube_data_cropped)
        data['image'] = [numpy.ma.masked_array(data['image'][0],
                                               mask=numpy.isnan(
                                                   data['image'][0]))]
        return data

class IntakeLoader:
    """
    Loader class for the CMIP6 intake dataset.
    """
    def __init__(self, pattern):
        institution_id, experiment_id,member_id, grid, table_id,activity_id = pattern.split('_')
        self.experiment_id = experiment_id
        self.table_id = table_id
        self.grid_label = grid
        self.variable_id = ''
        self.institution_id = institution_id
        self.activity_id = activity_id
        self.member_id = member_id
        self._label = f'{self.experiment_id}_{self.institution_id}_{self.member_id}'
        self._cube = None

    @property
    def cube(self):
        """
        The purpose of this property is to delay loading of the cube until the
        point where all relevant parameters are defined and data and metadata
        can be downloaded.
        """
        if not self._cube:
            self._load_cube()
        return self._cube

    def _load_cube(self):
        self._cube = _load_from_intake(experiment_id=self.experiment_id,
                                       table_id=self.table_id,
                                       grid_label=self.grid_label,
                                       variable_id=self.variable_id,
                                       institution_id=self.institution_id,
                                       activity_id=self.activity_id,
                                       member_id=self.member_id)

    def image(self, state):
        """
        Main image loading function. This function will actually realise the
        data,
        """
        if self.variable_id != state.variable:
            self.variable_id = state.variable
            self._cube = None

        valid_time = state.valid_time
        pressure = state.pressure

        selected_time = util.to_datetime(valid_time)

        # the guts of creating the bokeh object has been put into a separate
        # function so that it can be cached, so if image is called multiple
        # time the calculations are only done once (hopefully).
        cube = self.cube
        coord_names = [c1.name() for c1 in cube.coords()]
        if 'air_pressure' in coord_names and pressure is None:
            data = gridded_forecast.empty_image()
            return data

        data = _get_bokeh_image(cube, self.experiment_id,
                                self.variable_id,
                                self.institution_id, state.initial_time,
                                self.member_id, selected_time, pressure)

        data.update(gridded_forecast.coordinates(str(selected_time),
                                                 state.initial_time,
                                                 state.pressures,
                                                 pressure))
        data.update({
            'name': [self._label],
            'units': [str(cube.units)],
            'experiment': [self.experiment_id],
            'institution': [self.institution_id],
            'memberid': [self.member_id],
            'variableid': [self.variable_id]
        })

        return data


class Navigator:
    """
    Navigator class for CMIP6 intake dataset.
    """
    def __init__(self):
        self.experiment_id = ''
        self.table_id = ''
        self.grid_label = ''
        self.variable_id = ''
        self.institution_id = ''
        self.activity_id = ''
        self.parent_source_id = ''
        self.member_id = ''
        self._cube = None

    def _parse_pattern(self, pattern):
        """
        The pattern contains the important CMIP6 parameters needed to get the
        correct data and metadata through the intake catalogue.
        """
        institution_id, experiment_id,member_id, grid, table_id,activity_id = pattern.split('_')
        self.experiment_id = experiment_id
        self.table_id = table_id
        self.grid_label = grid
        self.institution_id = institution_id
        self.activity_id = activity_id
        self.member_id = member_id
        self._label = f'{self.experiment_id}_{self.institution_id}_{self.member_id}'

    @property
    def cube(self):
        """
        The purpose of this property is to delay loading of the cube until the
        point where all relevant parameters are defined and data and metadata
        can be downloaded.
        """
        if not self._cube:
            self._load_cube()
        return self._cube

    def _load_cube(self):
        if not self.variable_id:
            self.variable_id = self._get_vars()[0]
        self._cube = _load_from_intake(experiment_id=self.experiment_id,
                                       table_id=self.table_id,
                                       grid_label=self.grid_label,
                                       variable_id=self.variable_id,
                                       institution_id=self.institution_id,
                                       activity_id=self.activity_id,
                                       member_id=self.member_id)


    def variables(self, pattern):
        self._parse_pattern(pattern)
        return self._get_vars()

    def _get_vars(self):
        var_list = _get_intake_vars(experiment_id=self.experiment_id,
                                table_id=self.table_id,
                                grid_label=self.grid_label,
                                institution_id=self.institution_id,
                                member_id=self.member_id)
        # make air temperature at surface the first variable so it shows as
        # default
        if 'tas' in var_list:
            var_list = ['tas'] + [v1 for v1 in var_list if v1 != 'tas']
        return var_list


    def initial_times(self, pattern, variable=None):
        self._parse_pattern(pattern)
        cube = self.cube
        for cell in cube.coord('time').cells():
            init_time = util.to_datetime(cell.point)
            return [init_time]

    def valid_times(self, pattern, variable, initial_time):
        if self.variable_id != variable:
            self.variable_id = variable
            self._cube = None
        self._parse_pattern(pattern)
        cube = self.cube
        valid_times = [util.to_datetime(cell.point) for cell in
                       cube.coord('time').cells()]
        return valid_times

    def pressures(self, pattern, variable, initial_time):
        print(f'retrieving pressures for variable {variable}')
        if self.variable_id != variable:
            self.variable_id = variable
            self._cube = None
        self._parse_pattern(pattern)
        cube = self.cube
        print(pattern)
        print(variable)
        try:
            # get pressures and sorted from largest to smallest, so that
            # closer to the surface shows higher up the list.
            pressures = sorted((cell.point for cell in
                         cube.coord('air_pressure').cells()), reverse=True)
        except iris.exceptions.CoordinateNotFoundError:
            pressures = []
        return pressures
