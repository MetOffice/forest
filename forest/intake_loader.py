import intake

URL = 'https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json'

def _load_from_intake(
    experiment_id='ssp585',
    table_id='Amon',
    grid_label='gn',
    variable_id='ta',
    institution_id='NCAR',
    activity_id='ScenarioMIP',
    parent_source_id='CESM2',
    member_id='r2i1p1f1'):
    collection = intake.open_esm_datastore(URL)
    cat = collection.search(
        experiment_id=experiment_id,
        table_id=table_id,
        grid_label=grid_label,
        institution_id=institution_id,
        member_id=member_id)
    print(cat)
    dset_dict = cat.to_dataset_dict(
        zarr_kwargs={'consolidated': True, 'decode_times': False},
        cdf_kwargs={'chunks': {}, 'decode_times': False})
    print(dset_dict.keys())
    ds_label = f'{activity_id}.{institution_id}.{parent_source_id}.{experiment_id}.{table_id}.{grid_label}'
    xr = dset_dict[ds_label]
    cube = xr[variable_id].to_iris()
    return cube


class IntakeLoader:
    def __init__(self):
        self._label = 'something'
        self._cubes = _load(pattern)

    def image(self, state):
        # TODO: cube = ?
        if cube is None:
            data = empty_image()
        else:
            data = geo.stretch_image(cube.coord('longitude').points,
                                     cube.coord('latitude').points, cube.data)
            data.update(coordinates(state.valid_time, state.initial_time,
                                    state.pressures, state.pressure))
            data.update({
                'name': [self._label],
                'units': [str(cube.units)]
            })
        return data


class IntakeNavigator:
    def __init__(self):
        pass

    def variables(self):
        pass

    def initial_times(self):
        pass

    def valid_times(self):
        pass

    def pressures(self):
        pass


if __name__ == '__main__':
    cube = _load_from_intake()
    print(cube)