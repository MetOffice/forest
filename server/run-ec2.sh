#!/bin/bash
REPO_DIR=$1
BUCKET_DIR=$2
bokeh serve \
    --use-xheaders \
    --port 8080 \
    --allow-websocket-origin forest.informaticslab.co.uk \
     ${REPO_DIR}/forest \
         --args \
         --config ${REPO_DIR}/forest/config.yaml \
         --directory ${BUCKET_DIR}/met-office-rmed-forest \
         --database /database/forest-informaticslab.db
         --survey-db /survey/survey.json
