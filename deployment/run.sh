#! /bin/bash

# TODO this should be in a loop
mkdir -p /s3/mogreps-uk
goofys mogreps-uk /s3/mogreps-uk

mkdir -p /s3/mogreps-g
goofys mogreps-g /s3/mogreps-g

mkdir -p /s3/graeme-misc-london
goofys graeme-misc-london /s3/graeme-misc-london

mkdir -p /s3/stephen-sea-public-london
goofys stephen-sea-public-london /s3/stephen-sea-public-london

export BOKEH_APPS_DIR=/opt/bokeh_apps
if [ -z "$PUBLIC_IP" ]
then
    bokeh serve --port 8888 ${BOKEH_APPS_DIR}/*
else
    bokeh serve --allow-websocket-origin $PUBLIC_IP:8888 --port 8888 ${BOKEH_APPS_DIR}/*
fi