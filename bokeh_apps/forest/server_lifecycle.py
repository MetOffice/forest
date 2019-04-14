from collections import OrderedDict
import os
import yaml
import data


class Namespace():
    """Namespace turns attrs into properties"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def parse_env():
    directory = os.getenv('FOREST_DIR', None)
    config_file = os.getenv('FOREST_CONFIG_FILE', None)
    return Namespace(
            directory=directory,
            config_file=config_file)


def on_server_loaded(server_context):
    env = parse_env()
    if env.config_file is not None:
        with open(env.config_file) as stream:
            config = yaml.load(stream)
    else:
        config = {
            "patterns": []
        }

    patterns = OrderedDict({})
    for item in config["patterns"]:
        patterns[item["name"]] = item["directory"]
    data.on_server_loaded(patterns)
