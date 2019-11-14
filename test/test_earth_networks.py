import datetime as dt
import glob
from forest import earth_networks
import pandas as pd
import pandas.testing as pdt


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
    expect = pd.Series([
        dt.datetime(2019, 4, 17, 0, 0, 1, 440000),
        "IC",
        2.75144,
        31.92064], index=[
            "date",
            "flash_type",
            "latitude",
            "longitude"
        ],
        name=0)
    pdt.assert_series_equal(expect, result)
