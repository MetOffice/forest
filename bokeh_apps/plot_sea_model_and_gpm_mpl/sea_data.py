import os
class ForestDataset(object):
    def __init__(self,
                 config,
                 file_name,
                 s3_base,
                 s3_local_base,
                 use_s3_mount,
                 base_local_path,
                 do_download,
                 var_lookup):
        self.config_name = config
        self.var_lookup = var_lookup
        self.file_name = file_name
        self.s3_base_url = s3_base
        self.s3_url = os.path.join(self.s3_base_url, self.file_name)
        self.s3_local_base = s3_local_base
        self.s3_local_path = os.path.join(self.s3_local_base, self.file_name)
        self.use_s3_local_mount = use_s3_mount
        self.base_local_path = base_local_path
        self.do_download = do_download
        self.local_path = os.path.join(self.base_local_path,
                                       self.file_name)

        # set up data loader functions
        self.loaders = dict([(v1, self.basic_cube_load) for v1 in VAR_NAMES])
        self.loaders[WIND_SPEED_NAME] = self.wind_speed_loader
        self.loaders[WIND_VECTOR_NAME] = self.wind_vector_loader
        for wv_var in WIND_VECTOR_VARS:
            self.loaders[wv_var] = self.wind_vector_loader

        self.data = dict([(v1, None) for v1 in self.loaders.keys()])
        if self.use_s3_local_mount:
            self.path_to_load = self.s3_local_path
        else:
            self.path_to_load = self.local_path

    def __str__(self):
        return 'FOREST dataset'

    def get_data(self, var_name):
        if self.data[var_name] is None:
            # get data from aws s3 storage
            self.retrieve_data()

            # load the data into memory from file (will only load meta data initially)
            self.load_data(var_name)

        return self.data[var_name]

    def load_data(self, var_name):
        self.loaders[var_name](var_name)

    def retrieve_data(self):
        '''
        '''
        if self.do_download:
            if not (os.path.isdir(self.base_local_path)):
                print('creating directory {0}'.format(self.base_local_path))
                os.makedirs(self.base_local_path)

            lib_sea.download_from_s3(self.s3_url, self.local_path)