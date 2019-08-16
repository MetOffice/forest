#!/bin/bash
REPO_DIR=$1
BUCKET_DIR=$2
bokeh serve \
    --port 8080 \
    --allow-websocket-origin forest.informaticslab.co.uk \
     ${REPO_DIR}/apps/forest
