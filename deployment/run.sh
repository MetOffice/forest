#! /bin/bash

ln -s /opt/s3 ~/s3

export BOKEH_APPS_DIR=/opt/bokeh_apps

if [ -z "$PUBLIC_IP" ]
then
    bokeh serve --port=8888 --num-procs=0 ${BOKEH_APPS_DIR}/*
else
    bokeh serve --allow-websocket-origin $PUBLIC_IP:8888 --port=8888 --num-procs=0 ${BOKEH_APPS_DIR}/*
fi