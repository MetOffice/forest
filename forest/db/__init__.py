import functools
import os

from .control import *
from .database import *
from .locate import *
from .util import *


@functools.lru_cache(maxsize=None)
def get_database(database_path):
    if database_path != ':memory:' and not os.path.exists(database_path):
        raise ValueError(f'Database file {database_path!r} must exist')
    return Database.connect(database_path)
