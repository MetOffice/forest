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
    return {"result": [dt.datetime(2022, 1, 1)]}


@app.get("/navigator/pressures")
def pressures():
    return {"result": [1000, 100, 1]}


@app.get("/map_view/image")
def map_view():
    return {
        "result": {
            "x": [-2e6],
            "y": [-2e6],
            "dw": [4e6],
            "dh": [4e6],
            "image": [[[0, 1, 2], [3, 4, 5], [6, 7, 8]]],
        }
    }
