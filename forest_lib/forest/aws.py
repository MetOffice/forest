"""
Amazon Web Services
===================

Forest uses Amazon Web Service (AWS) S3 buckets that contain
the data to drive the dynamic web visualisations

Download
--------

Without AWS credentials it is possible to access Forest data
for development purposes using Python's standard library
module :py:mod:`urllib`. A :class:`.S3Bucket` should be
used, the bucket loads a file on demand for use in Forest

>>> file_loader = forest.aws.S3Bucket(
...                   server_address='https://...',
...                   bucket_name='my_bucket',
...                   download_directory='/download',
...               )

A file_loader can then be used to query remote and local file
existence, and if necessary trigger a download on demand.

>>> file_loader.load_file('file.nc')

Mount file system
-----------------

If a file system has been mounted using a POSIX-ish wrapper
like **Goofys** then a file system interface can be used

>>> file_loader = forest.aws.S3Mount('/mount/dir')
>>> file_loader.load_file('file.nc')

A mounted file system makes it quicker and easier for Forest
to find and load data on demand as it only accesses the
individual values needed to plot the data

Application programming interface (API)
---------------------------------------

Forest can access data using Amazon credentials or
download data from AWS using urllib

"""
import os
import urllib.request


# expose print to be patched
print = print


class S3Bucket(object):
    """S3 bucket urllib interface

    Loads files from **AWS** S3 bucket using :py:mod:`urllib`
    module from the Python standard library, specifically
    ``urllib.requests``

    :param server_address: Amazon server URL
    :param bucket_name: name of bucket
    :param download_directory: directory on local file system to
                               store file(s)
    """
    def __init__(self,
                 server_address,
                 bucket_name,
                 download_directory):
        self.server_address = server_address
        self.bucket_name = bucket_name
        self.download_directory = download_directory

    def file_exists(self, file_name):
        """AWS file exists or downloaded file exists

        .. note:: Only checks remote file existence
                  to support peculiarities of :mod:`forest.data`

        :param file_name: base name of file to be queried
        :returns: logical indicating file existence
        """
        return self.remote_file_exists(file_name)

    def remote_file_exists(self, file_name):
        """Check file present in S3 bucket

        :param file_name: base name of file to be queried
        :returns: logical indicating file existence
        """
        return self.s3_file_exists(self.s3_url(file_name))

    @staticmethod
    def s3_file_exists(url):
        """Check whether a remote file exists; return Bool.

        Check whether a file at the remote location specified by remore
        path exists by trying to open a url request.

        :param remote_path: Path to check for file at.
        :returns: logical indicating file existence
        """
        try:
            _ = urllib.request.urlopen(remote_path)
            print('file {0} found at remote location.'.format(remote_path))
            return True
        except urllib.error.HTTPError:
            warning_msg1 = 'warning: file {0} NOT found at remote location.'
            warning_msg1 = warning_msg1.format(remote_path)
            print(warning_msg1)
        return False

    def local_file_exists(self, file_name):
        """Check file present on disk

        :param file_name: base name of file
        :returns: logical indicating file existence
        """
        return os.path.isfile(self.path_to_load(file_name))

    def path_to_load(self, file_name):
        """Compute string representation of downloaded file

        :param file_name: base name of file
        :returns: full path of file
        """
        return os.path.join(self.download_directory, file_name)

    def load_file(self, file_name):
        """Do download from S3 bucket if file not already on disk

        :param file_name: base name of file
        :returns: path_to_file
        """
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
    """Local file system mount point access to AWS

    :param directory: directory where AWS file system is mounted
    """
    def __init__(self, directory):
        self.directory = directory

    def file_exists(self, path):
        """Check file exists on file system

        :param file_name: base name of file to be queried
        :returns: logical indicating file existence
        """
        return os.path.isfile(path)

    def path_to_load(self, file_name):
        """Compute string representation of mounted file

        :param file_name: base name of file
        :returns: full path of file
        """
        return os.path.join(self.directory, file_name)

    def load_file(self, file_name):
        """Full path to file on file system

        .. note:: Method does not perform i/o, it exists to
                  fulfill API requirements

        :param file_name: base name of file
        :returns: full path of file
        """
        return self.path_to_load(file_name)
