"""Amazon Web Services infrastructure"""
import os


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
