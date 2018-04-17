#! /bin/bash

ln -s /opt/s3 ~/s3

export BOKEH_APPS_DIR=/opt/bokeh_apps
export S3_ROOT=/opt/s3/stephen-sea-public-london/

if [ -z "$PUBLIC_IP" ]
then
    bokeh serve --port=80 --num-procs=0 ${BOKEH_APPS_DIR}/*
else
    bokeh serve --allow-websocket-origin $PUBLIC_IP:80 --port=80 --num-procs=0 ${BOKEH_APPS_DIR}/*
fi