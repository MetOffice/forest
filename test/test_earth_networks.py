import datetime as dt
import numpy as np
import glob
from forest import earth_networks


LINES = [
    "1,20190417T000001.440,+02.7514400,+031.9206400,-000001778,000,15635,007,001",
    "1,20190417T000001.093,+02.6388400,+031.9008800,+000002524,000,14931,007,012"
]


def test_earth_networks(tmpdir):
    path = str(tmpdir / "sample.txt")
    with open(path, "w") as stream:
        stream.writelines(LINES)

    loader = earth_networks.Loader([path])
    frame = loader.load_date(dt.datetime(2019, 4, 17))
    result = frame.iloc[0]
    atol = 0.000001
    if isinstance(result["date"], dt.datetime):
        # Pandas <0.25.x
        assert result["date"] == dt.datetime(2019, 4, 17, 0, 0, 1, 440000)
    else:
        # Pandas 1.0.x
        assert result["date"] == np.datetime64('2019-04-17T00:00:01.440000000')
    assert result["flash_type"] == "IC"
    assert abs(result["latitude"] - 2.75144) < atol
    assert abs(result["longitude"] - 31.92064) < atol
