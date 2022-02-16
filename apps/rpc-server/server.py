#!/usr/bin/env python
import uvicorn
import glob
import datetime as dt
from fastapi import FastAPI


app = FastAPI()


@app.get("/navigator/variables")
def variables():
    return {"result": ["air_temperature"]}


@app.get("/navigator/initial_times")
def initial_times():
    return {"result": [dt.datetime(2022, 1, 1)]}


@app.get("/navigator/valid_times")
def valid_times():
    return {"result": [dt.datetime(2022, 1, 1), dt.datetime(2022, 1, 1, 3)]}


@app.get("/navigator/pressures")
def pressures():
    return {"result": [1000, 100, 1]}


@app.get("/map_view/image")
def map_view(valid_time: dt.datetime):
    print(f"{valid_time=}")
    return {
        "result": {
            "x": [-2e6],
            "y": [-2e6],
            "dw": [4e6],
            "dh": [4e6],
            "image": [[[valid_time.hour, 1, 2], [3, 4, 5], [6, 7, 8]]],
        }
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")
