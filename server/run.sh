#!/usr/bin/env bash
export S3_ROOT=$1
PORT=$2
REPO_DIR=$3
DAY_IN_MILLISECONDS=86400000
PING_MILLISECONDS=10000

conda install -y --file ${REPO_DIR}/conda-spec.txt

USE_CUSTOM=False
if [[ "$USE_CUSTOM" == "True" ]] ; then
    # Tornado server for HIGHWAY and WCSSP
    ${BOKEH_APP_DIR}/server.py \
        --port=${PORT} \
        --allow-websocket-origin ${FOREST_URL} \
        --unused-session-lifetime ${DAY_IN_MILLISECONDS} \
        --keep-alive ${PING_MILLISECONDS}
else
    # Bokeh server for HIGHWAY only
    FOREST_DIR=${S3_ROOT}/stephen-sea-public-london \
    FOREST_CONFIG_FILE=${REPO_DIR}/forest/highway.yaml \
    bokeh serve ${REPO_DIR}/forest \
        --port ${PORT} \
        --allow-websocket-origin ${FOREST_URL} \
        --unused-session-lifetime ${DAY_IN_MILLISECONDS} \
        --keep-alive ${PING_MILLISECONDS}
fi
