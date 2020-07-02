import sys
import os
import forest.main
import forest.cli.main
import forest.data as data


class DatasetSyncCallback:
    """Process to synchronize datasets"""
    def __init__(self, datasets):
        self.datasets = datasets

    def __call__(self):
        for dataset in self.datasets:
            if hasattr(dataset, "sync"):
                dataset.sync()


def on_server_loaded(server_context):
    data.on_server_loaded()

    # Add periodic callback to keep database(s) up to date
    config = forest.main.configure(parse_forest_args())
    interval_ms = 15 * 60 * 1000  # 15 minutes in miliseconds
    callback = DatasetSyncCallback(list(config.datasets))
    server_context.add_periodic_callback(callback, interval_ms)


def parse_forest_args(argv=None):
    """Find arguments suitable for forest.parse_args.parse_args

    e.g. bokeh serve --args [ARGS] or forest [ARGS]
    """
    if argv is None:
        argv = sys.argv
    if "bokeh" in os.path.basename(argv[0]):
        i = argv.index("--args")
        return argv[i + 1:]
    else:
        _, argv = forest.cli.main.parse_args(argv)
        return argv[1:]


def on_session_destroyed(session_context):
    '''
    Function called when a session is closed 
    (e.g. tab closed or time out)
    '''
    if data.AUTO_SHUTDOWN:
        import sys 
        sys.exit('\033[1;31mThe session has ended - tab closed or timeout. \n\n --- Terminating the Forest progam and relinquishing control of port. ---\033[1;00m')
