#!/usr/bin/env bash

export S3_ROOT=$1
bokeh serve --allow-websocket-origin forestkube.informaicslab.co.uk:$2 --port=$2 $3/*