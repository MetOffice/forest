#!/bin/bash
# EC2 instance bokeh serve command
REPO_DIR=$1
BUCKET_DIR=$2
bokeh serve \
    --port 8080 \
    --allow-websocket-origin forest-future.informaticslab.co.uk \
    ${REPO_DIR}/forest \
    --args \
        --database ${BUCKET_DIR}/met-office-rmed-forest/forest-informaticslab.db \
        --config-file ${REPO_DIR}/forest/config.yaml
