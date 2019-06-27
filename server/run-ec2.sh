#!/bin/bash
# EC2 instance bokeh serve command
REPO_DIR=$1
WEBSOCKET_ORIGIN=$2
bokeh serve \
    --port 8080 \
    --allow-websocket-origin ${WEBSOCKET_ORIGIN} \
    ${REPO_DIR}/forest \
    --args \
        --database ${REPO_DIR}/forest/empty.db \
        --config-file ${REPO_DIR}/forest/config.yaml
