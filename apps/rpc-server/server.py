#!/usr/bin/env python
import uvicorn
import glob
import datetime as dt
from fastapi import FastAPI


app = FastAPI()


@app.get("/navigator/variables")
def variables():
    return {"result": ["air_temperature", "relative_humidity"]}


@app.get("/navigator/initial_times")
def initial_times():
    return {"result": [dt.datetime(2022, 1, 1)]}


@app.get("/navigator/valid_times")
def valid_times():
    return {"result": [dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 1, 3)]}


@app.get("/navigator/pressures")
def pressures():
    return {"result": []}


@app.get("/map_view/image")
def map_view(
    valid_time: dt.datetime,
    initial_time: dt.datetime,
    variable: str,
    pressure: float,
):
    print(f"{initial_time=}")
    print(f"{valid_time=}")
    print(f"{variable=}")
    print(f"{pressure=}")
    if valid_time.hour == 0:
        return {
            "result": {
                "x": [-2e6],
                "y": [-2e6],
                "dw": [4e6],
                "dh": [4e6],
                "image": [[[2, 1, 0], [3, 4, 8], [6, 7, 5]]],
            }
        }
    else:
        return {
            "result": {
                "x": [-2e6],
                "y": [-2e6],
                "dw": [4e6],
                "dh": [4e6],
                "image": [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]],
            }
        }


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
