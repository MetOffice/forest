"""Amazon Web Services infrastructure"""
import os
import numpy as np
import iris
from . import util


class SyntheticBucket(object):
    """Synthetic file system

    Useful for developing Forest prototypes since there
    is no communication with AWS or a real file system
    """
    def file_exists(self, file_name):
        return True

    def path_to_load(self, file_name):
        return file_name

    def load_cube(self, file_name, constraint):
        n = 5
        time = iris.coords.DimCoord([0, 1],
                standard_name="time",
                units="seconds since 1981-01-01 00:00:00 utc")
        latitude = iris.coords.DimCoord(np.linspace(-90, 90, n),
                                        standard_name="latitude",
                                        units="degrees")
        longitude = iris.coords.DimCoord(np.linspace(-180, 180, n),
                                         standard_name="longitude",
                                         units="degrees")
        class Code(object):
            section = "section"
        attributes = {
            "STASH": Code()
        }
        return iris.cube.Cube(
            np.zeros((2, n, n)),
            dim_coords_and_dims=[
                (time, 0),
                (latitude, 1),
                (longitude, 2)
            ],
            attributes=attributes
        ).extract(constraint)


class S3Bucket(object):
    """S3 bucket infrastructure"""
    def __init__(self, server_address, bucket_name,
                 use_s3_mount=True,
                 do_download=False):
        self.server_address = server_address
        self.bucket_name = bucket_name
        self.use_s3_mount = use_s3_mount
        self.do_download = do_download

    @property
    def s3_base(self):
        return '{server}/{bucket}/model_data/'.format(server=self.server_address,
                                                      bucket=self.bucket_name)

    @property
    def s3_local_base(self):
        return os.path.join(self.s3_root, self.bucket_name, "model_data")

    @property
    def s3_root(self):
        try:
            return os.environ['S3_ROOT']
        except KeyError:
            return os.path.expanduser('~/s3')

    @property
    def base_path_local(self):
        try:
            local_root = os.environ['LOCAL_ROOT']
        except KeyError:
            local_root = os.path.expanduser('~/SEA_data')
        return os.path.join(local_root, 'model_data')

    def s3_url(self, file_name):
        return os.path.join(self.s3_base, file_name)

    def s3_local_path(self, file_name):
        return os.path.join(self.s3_local_base, file_name)

    def local_path(self, file_name):
        return os.path.join(self.base_path_local, file_name)

    def path_to_load(self, file_name):
        if self.use_s3_mount:
            return self.s3_local_path(file_name)
        else:
            return self.local_path(file_name)

    def file_exists(self, file_name):
        """AWS file exists or downloaded file exists"""
        if self.do_download:
            return util.check_remote_file_exists(self.s3_url(file_name))
        else:
            return os.path.isfile(self.path_to_load(file_name))

    def retrieve_file(self, file_name):
        directory = self.base_path_local
        if not os.path.isdir(directory):
            print("creating directory {0}".format(directory))
            os.makedirs(directory)
        util.download_from_s3(self.s3_url(file_name),
                              self.local_path(file_name))

    def load_cube(self, file_name, constraint):
        """Helper method to perform iris.Cube I/O

        .. note:: This method makes ForestDataset file system
                  independent
        """
        if self.do_download and self.file_exists(file_name):
            self.retrieve_file(file_name)
        return iris.load_cube(self.path_to_load(file_name), constraint)
