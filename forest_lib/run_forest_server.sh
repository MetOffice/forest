#!/usr/bin/env bash

export S3_ROOT=$1
bokeh serve --allow-websocket-origin ${FOREST_URL} --port=$2 $3/*