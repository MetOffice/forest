"""
Configure application
---------------------

This module implements parsers and data structures
needed to configure the application. It supports
richer settings than those that can be easily
represented on the command line by leveraging file formats
such as YAML and JSON that are widely used to configure
applications.


.. code-block:: yaml
    :caption: Example config.yaml

    datasets:
        - label: Trial
          driver:
            name: gridded_forecast
            settings:
              pattern: "*.nc"
              directory: trial/output
        - label: Control
          driver:
            name: gridded_forecast
            settings:
              pattern: "*.nc"
              directory: control/output
        - label: RDT
          driver:
            name: rdt
            settings:
              pattern: "*.json"
              directory: /satellite/rdt/json


.. autofunction:: dataset_specs

.. autofunction:: to_data

.. autofunction:: parse_datasets

.. autoclass:: DatasetSpec
   :members:

.. autoclass:: DriverSpec
   :members:

"""
import os
import typing
import yaml
from forest.export import export


__all__ = []


class DriverSpec(typing.NamedTuple):
    """Specification to instansiate a driver

    :param name: name of driver in `forest.drivers`
    :type name: str
    :param settings: keyword args passed to Dataset constructor
    :type settings: dict
    """
    name: str
    settings: dict


class DatasetSpec(typing.NamedTuple):
    """Specification to instantiate a dataset

    Contains information needed to build a
    dataset. Every dataset needs a label and
    a driver to power visualisations

    :param label: text description of dataset
    :type label: str
    :param driver: specification of driver
    :type driver: DriverSpec
    """
    label: str
    driver: DriverSpec


def dataset_specs(config_file, files, file_type, directory, database):
    """Generate specifications from settings

    :returns: list of :class:`DatasetSpec`
    """
    return parse_datasets(to_data(config_file, files, file_type, directory))


def parse_datasets(data):
    """Parse configuration data into convenient namedtuples

    It takes a application representation and returns named tuples

    >>> data = {
    ...     "datasets": [{
    ...         "label": "Hello",
    ...         "driver": {
    ...             "name": "world",
    ...             "settings": {"x": 1}
    ...         }}]
    ... }
    >>> datasets = parse_datasets(data)
    >>> datasets
    ... [DatasetSpec(label="Hello",
    ...              driver=DriverSpec(name="world",
    ...                                settings={"x": 1}))]
    >>> datasets[0].driver.settings
    ... {"x": 1}

    Named tuples provide syntactic sugar to ease attribute access

    :param data: data structure representing application configuration
    :type data: dict
    :returns: list of :class:`DatasetSpec`
    """
    labels = [ds["label"]
            for ds in data["datasets"]]
    drivers = [DriverSpec(ds["driver"]["name"], ds["driver"].get("settings", {}))
            for ds in data["datasets"]]
    return [DatasetSpec(label, driver) for label, driver in zip(labels, drivers)]


def to_data(config_file, files, file_type, directory):
    """Convert command line args to intermediate data structure

    It takes a parsed command line and returns a data structure
    containing a list of configured datasets

    >>> to_data(None, ["a.nc"], "unified_model", "/prefix")
    ... {
    ... "datasets": [
    ...     {
    ...         "label": "a.nc",
    ...         "driver": {
    ...             "name": "unified_model",
    ...             "settings": {
    ...                 "pattern": "/prefix/a.nc"
    ...             }
    ...         }
    ...     }
    ... }

    .. note:: If ``config_file`` is not ``None`` its contents are
              loaded and merged with other command line settings

    :param config_file: path to config file
    :param files: list of file names
    :param file_type: keyword to select loader
    :param directory: prefix directory
    :returns: nested structure representing application configuration
    """
    datasets = []

    # Parse config file to datasets
    if config_file is not None:
        with open(config_file) as stream:
            try:
                # PyYaml 5.1 onwards
                data = yaml.full_load(stream)
            except AttributeError:
                data = yaml.load(stream)
        datasets += data["datasets"]

    # Append command line files
    for path in files:
        datasets.append({
            "label": path,
            "driver": {
                "name": file_type,
                "settings": {
                    "pattern": path
                }
            }
        })

    # Update datasets with prefix directory
    if directory is not None:
        for dataset in datasets:
            settings = dataset["driver"]["settings"]
            pattern = settings["pattern"]
            settings["pattern"] = os.path.join(directory, pattern)

    return {
        "datasets": datasets
    }


class Config(object):
    """Configuration data structure

    This high-level object represents the application configuration.
    It is file format agnostic but has helper methods to initialise
    itself from disk or memory.

    .. note:: This class is intended to provide the top-level
              configuration with low-level details implemented
              by specialist classes, e.g. :class:`FileGroup`
              which contains meta-data for files

    :param data: native Python data structure representing application
                 settings
    """
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return "{}({})".format(
                self.__class__.__name__,
                self.data)

    @property
    def patterns(self):
        if "files" in self.data:
            return [(f["label"], f["pattern"])
                    for f in self.data["files"]]
        return []

    @classmethod
    def load(cls, path):
        """Parse settings from either YAML or JSON file on disk

        The configuration can be controlled elegantly
        through a text file. Groups of files can
        be specified in a list.

        .. note:: Relative or absolute directories are
                  declared through the use of a leading /

        .. code-block:: yaml

            files:
                - label: Trial
                  pattern: "*.nc"
                  directory: trial/output
                - label: Control
                  pattern: "*.nc"
                  directory: control/output
                - label: RDT
                  pattern: "*.json"
                  directory: /satellite/rdt/json
                  file_type: rdt

        :param path: JSON/YAML file to load
        :returns: instance of :class:`Config`
        """
        with open(path) as stream:
            try:
                # PyYaml 5.1 onwards
                data = yaml.full_load(stream)
            except AttributeError:
                data = yaml.load(stream)
        return cls(data)

    @classmethod
    def from_files(cls, files, file_type="unified_model"):
        """Configure using list of file names and a file type

        :param files: list of file names
        :param file_type: keyword to apply to all files
        :returns: instance of :class:`Config`
        """
        return cls({
            "files": [dict(pattern=f, label=f, file_type=file_type)
                for f in files]})

    @property
    def file_groups(self):
        return [FileGroup(**data)
                for data in self.data["files"]]


class FileGroup(object):
    """Meta-data needed to describe group of files

    To describe a collection of related files extra
    meta-data is needed. For example, the type of data
    contained within the files or how data is catalogued
    and searched.

    :param label: decription used by buttons and tooltips
    :param pattern: wildcard pattern used by either SQL or glob
    :param locator: keyword describing search method (default: 'file_system')
    :param file_type: keyword describing file contents (default: 'unified_model')
    :param directory: leaf/absolute directory where file(s) are stored (default: None)
    """
    def __init__(self,
            label,
            pattern,
            locator="file_system",
            file_type="unified_model",
            directory=None):
        self.label = label
        self.pattern = pattern
        self.locator = locator
        self.file_type = file_type
        self.directory = directory

    @property
    def full_pattern(self):
        if self.directory is None:
            return self.pattern
        return os.path.join(self.directory, self.pattern)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise Exception("Can not compare")
        attrs = ("label", "pattern", "locator", "file_type", "directory")
        return all(
                getattr(self, attr) == getattr(other, attr)
                for attr in attrs)

    def __repr__(self):
        arg_attrs = [
            "label",
            "pattern"]
        args = [self._str(getattr(self, attr))
                for attr in arg_attrs]
        kwarg_attrs = [
            "locator",
            "file_type",
            "directory"]
        kwargs = [
            "{}={}".format(attr, self._str(getattr(self, attr)))
                for attr in kwarg_attrs]
        return "{}({})".format(
                self.__class__.__name__,
                ", ".join(args + kwargs))
    @staticmethod
    def _str(value):
        if isinstance(value, str):
            return "'{}'".format(value)
        else:
            return str(value)


@export
def load_config(path):
    """Load configuration from a file"""
    return Config.load(path)


@export
def from_files(files, file_type):
    """Define configuration with a list of files"""
    return Config.from_files(files, file_type)
