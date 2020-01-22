#!/bin/bash
# Script to keep local database in-sync with S3 contents
set -x
docker run \
    --rm \
    --name forest-cron \
    -v /home/ec2-user/forest:/repo/forest \
    -v /home/ec2-user/database:/database \
    -v /s3/met-office-rmed-forest:/s3/met-office-rmed-forest \
    informaticslab/forest bash -c '. /repo/forest/server/housekeeping/update-database-container.sh'
