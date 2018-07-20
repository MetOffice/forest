"""Amazon Web Services infrastructure"""
import os
from . import util


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
