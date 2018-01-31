#! /bin/bash
pysssix /s3 >/dev/null 2>&1 &

if [ -z "$PUBLIC_IP" ]
then
    bokeh serve --port 8888 /opt/documents/*
else
    bokeh serve --allow-websocket-origin $PUBLIC_IP:8888 --port 8888 /opt/documents/*
fi