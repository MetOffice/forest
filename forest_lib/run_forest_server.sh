#!/usr/bin/env bash

conda install bokeh=0.13.0 -y

export S3_ROOT=$1
PORT=$2
BOKEH_APP_DIR=$3

# Tornado server for HIGHWAY and WCSSP
${BOKEH_APP_DIR}/server.py \
    --port=${PORT} \
    --allow-websocket-origin ${FOREST_URL}
