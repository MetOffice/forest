#!/usr/bin/env bash

conda install bokeh=0.13.0 -y
conda install tornado=5.0.1 -y

export S3_ROOT=$1
PORT=$2
BOKEH_APP_DIR=$3
DAY_IN_MILLISECONDS=86400000

# Tornado server for HIGHWAY and WCSSP
${BOKEH_APP_DIR}/server.py \
    --port=${PORT} \
    --allow-websocket-origin ${FOREST_URL} \
    --unused-session-lifetime ${DAY_IN_MILLISECONDS}
