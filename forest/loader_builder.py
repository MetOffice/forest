import os
from forest import (
        data,
        db,
        unified_model)


class LoaderBuilder(object):
    """Encapsulates complex Loader construction logic"""
    def __init__(self,
            label,
            file_type,
            file_pattern=None,
            files=None,
            prefix_dir=None,
            leaf_dir=None,
            locator_type="file_system",
            use_files=False):
        self.label = label
        self.locator_type = locator_type
        self.file_type = file_type
        self.file_pattern = file_pattern
        self.files = files
        self.prefix_dir = prefix_dir
        self.leaf_dir = leaf_dir
        self.use_files = use_files

    def add_database(self, database):
        self.database = database

    @classmethod
    def group_args(cls, group, args):
        """Construct builder from FileGroup and argparse.Namespace

        :param group: FileGroup instance
        :param args: argparse.Namespace instance
        """
        return cls(
                group.label,
                group.file_type,
                file_pattern=group.pattern,
                files=args.files,
                prefix_dir=args.directory,
                leaf_dir=group.directory,
                locator_type=group.locator,
                use_files=args.config_file is None)

    def loader(self):
        """Construct Loader related to file_type"""
        return data.file_loader(
                    self.file_type,
                    self.pattern(),
                    label=self.label,
                    locator=self.locator())

    def locator(self):
        """Helper method to construct locator"""
        if self.locator_type == "database":
            # Search using SQL database
            return db.Locator(
                self.database.connection,
                directory=self.replace_dir(self.prefix_dir, self.leaf_dir))
        elif self.locator_type == "file_system":
            if self.use_files:
                # Search using list of files
                locator = None  # RDT, EIDA50 etc. have built-in locators
                if self.file_type == 'unified_model':
                    locator = unified_model.Locator(self.files)
                return locator
            else:
                # Search using prefix, leaf and pattern
                locator = None  # RDT, EIDA50 etc. have built-in locators
                if self.file_type == 'unified_model':
                    locator = unified_model.Locator.pattern(self.pattern())
                return locator
        else:
            raise Exception("Unknown locator: {}".format(self.locator_type))

    def pattern(self):
        """Helper method to construct Loader"""
        if self.locator_type == "database":
            return self.file_pattern
        elif self.locator_type == "file_system":
            if self.use_files:
                return self.file_pattern
            else:
                return os.path.expanduser(
                        self.full_pattern(
                            self.file_pattern,
                            self.leaf_dir,
                            self.prefix_dir))
        else:
            raise Exception("Unknown locator: {}".format(self.locator_type))

    @staticmethod
    def full_pattern(pattern, leaf_dir, prefix_dir):
        """Combine user specified patterns to files on disk

        .. note:: absolute path leaf directory takes precedence over prefix
                  directory

        :param pattern: str representing file name wildcard pattern
        :param leaf_dir: leaf directory to add after prefix directory
        :param prefix_dir: directory to place before leaf and pattern
        """
        dirs = [d for d in [prefix_dir, leaf_dir] if d is not None]
        return os.path.join(*dirs, pattern)

    @staticmethod
    def replace_dir(prefix_dir, leaf_dir):
        """Replacement directory for SQL queries

        Combine two user defined directories to allow flexible
        approach to directory specification

        :param prefix_dir: directory to put before relative leaf directory
        :param leaf_dir: directory to append to prefix
        """
        dirs = [d for d in [prefix_dir, leaf_dir] if d is not None]
        if len(dirs) == 0:
            return
        return os.path.join(*dirs)
