#!/bin/bash
# EC2 instance bokeh serve command
WEBSOCKET_ORIGIN=$1
bokeh serve \
    --port 8080 \
    --allow-websocket-origin ${WEBSOCKET_ORIGIN} \
    forest \
    --args \
        --database forest/empty.db \
        --config-file forest/config.yaml
