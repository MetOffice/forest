import pytest
import json
import numpy as np
from forest import encode


@pytest.mark.parametrize("value,expect", [
    (0, "0"),
    (np.int32(0), "0"),
    (np.float32(0.), "0.0"),
])
def test_json_serialize_float32(value, expect):
    assert expect == json.dumps(value, cls=encode.NumpyEncoder)
