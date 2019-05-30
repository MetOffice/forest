import bokeh.plotting
import argparse
import yaml
import _control as control
import _view as view
import _database as db


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        required=True,
        help="SQL database to optimise menu system")
    parser.add_argument(
        "--config-file",
        required=True, metavar="YAML_FILE",
        help="YAML file to configure application")
    return parser.parse_args(args=argv)


def main():
    args = parse_args()
    with open(args.config_file) as stream:
        config = load_config(stream)
    database = db.Database.connect(args.database)
    controls = control.Controls(database, patterns=config.patterns)
    controls.subscribe(print)

    locator = db.Locator.connect(args.database)
    text = view.View(text="Hello, world!", locator=locator)
    controls.subscribe(text.on_state)

    document = bokeh.plotting.curdoc()
    document.add_root(controls.layout)
    document.add_root(text.div)


class Namespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def load_config(stream):
    data = yaml.load(stream)
    patterns = [(m["name"], m["pattern"]) for m in data["models"]]
    return Namespace(patterns=patterns)


if __name__.startswith('bk'):
    main()
