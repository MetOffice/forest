#!/usr/bin/env bash

conda install bokeh=0.13.0 -y
export S3_ROOT=$1
bokeh serve --allow-websocket-origin ${FOREST_URL} --port=$2 $3/plot_sea_two_model_comparison