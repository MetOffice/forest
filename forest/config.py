"""
Configure application
---------------------

This module implements parsers and data structures
needed to configure the application. It supports
richer settings than those that can be easily
represented on the command line by leveraging file formats
such as YAML and JSON that are widely used to configure
applications.

.. autoclass:: Config
   :members:

.. autoclass:: FileGroup
   :members:

.. autofunction:: load_config

.. autofunction:: from_files

"""
import os
import string
import yaml
import forest.drivers
import forest.state
from dataclasses import dataclass, field
from collections import defaultdict
from collections.abc import Mapping
from forest.export import export


__all__ = []


@dataclass
class Figures:
    ui: bool = True
    maximum: int = 3


@dataclass
class Defaults:
    figures: Figures = field(default_factory=Figures)
    timeui: bool = True
    presetui: bool = True
    viewport: dict = field(default_factory=dict)

    def __post_init__(self):
        self._assign("figures", Figures)

    def _assign(self, att_name, cls):
        if isinstance(getattr(self, att_name), dict):
            obj = cls(**getattr(self, att_name))
            setattr(self, att_name, obj)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)


@dataclass
class PluginSpec:
    """Data representation of plugin"""
    entry_point: str


class Plugins(Mapping):
    """Specialist mapping between allowed keys and specs"""
    def __init__(self, data):
        allowed = ("feature",)
        self.data = {}
        for key, value in data.items():
            if key in allowed:
                self.data[key] = PluginSpec(**value)
            else:
                msg = f"{key} not in {allowed}"
                raise Exception(msg)

    def __getitem__(self, *args, **kwargs):
        return self.data.__getitem__(*args, **kwargs)

    def __len__(self, *args, **kwargs):
        return self.data.__len__(*args, **kwargs)

    def __iter__(self, *args, **kwargs):
        return self.data.__iter__(*args, **kwargs)


class Viewport:
    def __init__(self, lon_range, lat_range):
        self.lon_range = lon_range
        self.lat_range = lat_range


def combine_variables(os_environ, args_variables):
    """Utility function to update environment with user-specified variables

    .. note: When there is a key clash the user-specified args take precedence

    :param os_environ: os.environ dict
    :param args_variables: variables parsed from command line
    :returns: merged dict
    """
    variables = dict(os_environ)
    if args_variables is not None:
        variables.update(dict(args_variables))
    return variables


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
        self.plugins = Plugins(self.data.get("plugins", {}))
        self.state = forest.state.State.from_dict(self.data.get("state", {}))

    def __repr__(self):
        return "{}({})".format(
                self.__class__.__name__,
                self.data)

    @property
    def features(self):
        """Dict of user-defined feature toggles"""
        d = defaultdict(lambda: False)
        d.update(self.data.get("features", {}))
        return d

    @property
    def use_web_map_tiles(self):
        """Turns web map tiling backgrounds on/off

        .. code-block:: yaml

            use_web_map_tiles: false

        .. note:: This is best used during development if an internet
                  connection is not available
        """
        return self.data.get("use_web_map_tiles", True)

    @property
    def defaults(self):
        return Defaults.from_dict(self.data.get("defaults", {}))

    @property
    def default_viewport(self):
        defaults = self.data.get('defaults', {})
        viewport = defaults.get('viewport', {})
        lon_range = viewport.get('lon_range', (-180, 180))
        lat_range = viewport.get('lat_range', (-80, 80))
        return Viewport(lon_range, lat_range)

    @property
    def presets_file(self):
        """Colorbar presets JSON file

        A location on disk where colorbar settings can be saved/loaded. If
        the file does not exist it will be created by the application.

        Use the following syntax to declare the presets file location

        .. code-block:: yaml

            presets:
              file: ${HOME}/example/preset-save.json

        :returns: location on disk to save colorbar presets
        """
        return self.data.get("presets", {}).get("file", None)

    @property
    def patterns(self):
        if "files" in self.data:
            return [(f["label"], f["pattern"])
                    for f in self.data["files"]]
        return []

    @classmethod
    def load(cls, path, variables=None):
        """Parse settings from either YAML or JSON file on disk

        The configuration can be controlled elegantly
        through a text file. Groups of files can
        be specified in a list.

        .. note:: Relative or absolute directories are
                  declared through the use of a leading /

        .. code-block:: yaml

            files:
                - label: Trial
                  pattern: "${TRIAL_DIR}/*.nc"
                - label: Control
                  pattern: "${CONTROL_DIR}/*.nc"
                - label: RDT
                  pattern: "${RDT_DIR}/*.json"
                  file_type: rdt

        :param path: JSON/YAML file to load
        :param variables: dict of key/value pairs used by :py:class:`string.Template`
        :returns: instance of :class:`Config`
        """
        with open(path) as stream:
            text = stream.read()

        if variables is not None:
            template = string.Template(text)
            text = template.substitute(**variables)

        try:
            # PyYaml 5.1 onwards
            data = yaml.safe_load(text)
        except AttributeError:
            data = yaml.load(text)
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

    @property
    def datasets(self):
        for group in self.file_groups:
            settings = {
                "label": group.label,
                "pattern": group.pattern,
                "locator": group.locator,
                "database_path": group.database_path,
                "directory": group.directory
            }
            yield forest.drivers.get_dataset(group.file_type, settings)


class FileGroup(object):
    """Meta-data needed to describe group of files

    To describe a collection of related files extra
    meta-data is needed. For example, the type of data
    contained within the files or how data is catalogued
    and searched.

    .. note:: This class violates the integration separation principle (ISP)
              all driver settings are included in the same constructor

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
            directory=None,
            database_path=None):
        self.label = label
        self.pattern = pattern
        self.locator = locator
        self.file_type = file_type
        self.directory = directory
        self.database_path = database_path

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
            "directory",
            "database_path"]
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
