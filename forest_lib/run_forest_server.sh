#!/usr/bin/env bash

conda install bokeh=1.0.4 -y
conda install -c conda-forge nodejs -y

export S3_ROOT=$1
PORT=$2
BOKEH_APP_DIR=$3
DAY_IN_MILLISECONDS=86400000
PING_MILLISECONDS=10000

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
    # Install custom Typescript
    # cd ${BOKEH_APP_DIR}/forest/wind
    # npm install > ${FOREST_DIR}/npm_install.log 2>&1
    # npm run-script build > ${FOREST_DIR}/npm_run_script.log 2>&1
    # cd -
    touch ${S3_ROOT}/stephen-sea-public-london/file.txt

    FOREST_DIR=${S3_ROOT}/stephen-sea-public-london \
    FOREST_CONFIG_FILE=${BOKEH_APP_DIR}/forest/highway.yaml \
    bokeh serve ${BOKEH_APP_DIR}/forest \
        --port ${PORT} \
        --allow-websocket-origin ${FOREST_URL} \
        --unused-session-lifetime ${DAY_IN_MILLISECONDS} \
        --keep-alive ${PING_MILLISECONDS}
fi
