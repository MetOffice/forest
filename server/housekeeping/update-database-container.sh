#!/bin/bash
set -x

# Environment variables
PREFIX=/home/custom
export PYTHONPATH=${PREFIX}/lib/python3.6/site-packages:${PYTHONPATH}
export PATH=${PREFIX}/bin:${PATH}

# Install forestdb to custom directory needed by update script
cd /repo/forest
mkdir -p ${PREFIX}/lib/python3.6/site-packages
python setup.py install --prefix ${PREFIX}
cd -

# Check forestdb available inside container
forestdb -h

# Run custom update Python script
/repo/forest/server/housekeeping/update-database.py \
    /s3/met-office-rmed-forest \
    /database/philippines.db