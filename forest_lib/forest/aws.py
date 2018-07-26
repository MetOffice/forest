"""Amazon Web Services infrastructure"""
import os
import urllib.request


# expose print to be patched
print = print


class S3Bucket(object):
    """S3 bucket infrastructure"""
    def __init__(self,
                 server_address,
                 bucket_name,
                 download_directory):
        self.server_address = server_address
        self.bucket_name = bucket_name
        self.download_directory = download_directory

    def file_exists(self, file_name):
        """AWS file exists or downloaded file exists"""
        return self.remote_file_exists(file_name)

    def remote_file_exists(self, file_name):
        return check_remote_file_exists(self.s3_url(file_name))

    def local_file_exists(self, file_name):
        return os.path.isfile(self.path_to_load(file_name))

    def path_to_load(self, file_name):
        return os.path.join(self.download_directory, file_name)

    def load_file(self, file_name):
        if not os.path.isdir(self.download_directory):
            print("creating directory {0}".format(self.download_directory))
            os.makedirs(self.download_directory)
        if not self.local_file_exists(file_name) and self.remote_file_exists(file_name):
            self.s3_download(self.s3_url(file_name),
                             self.path_to_load(file_name))
        return self.path_to_load(file_name)

    def s3_url(self, file_name):
        return os.path.join(self.server_address,
                            self.bucket_name,
                            "model_data",
                            file_name)

    @staticmethod
    def s3_download(url, path):
        """port of forest.util.download_from_s3"""
        if not os.path.isfile(path):
            print('retrieving file from {0}'.format(url))
            urllib.request.urlretrieve(url, path)
            print('file {0} downloaded'.format(path))
        else:
            print(path, ' - File already downloaded')


class S3Mount(object):
    """Local file system mount point access to AWS"""
    def __init__(self, directory):
        self.directory = directory

    def file_exists(self, path):
        return os.path.isfile(path)

    def path_to_load(self, file_name):
        return os.path.join(self.directory, file_name)

    def load_file(self, file_name, constraint):
        return self.path_to_load(file_name)


def check_remote_file_exists(remote_path):
    """Check whether a remote file exists; return Bool.

    Check whether a file at the remote location specified by remore
    path exists by trying to open a url request.

    Arguments
    ---------
    - remote_path -- Str; Path to check for file at.
    """
    file_exists = False
    try:
        _ = urllib.request.urlopen(remote_path)
        print('file {0} found at remote location.'.format(remote_path))
        file_exists = True
    except urllib.error.HTTPError:
        warning_msg1 = 'warning: file {0} NOT found at remote location.'
        warning_msg1 = warning_msg1.format(remote_path)
        print(warning_msg1)
    return file_exists
