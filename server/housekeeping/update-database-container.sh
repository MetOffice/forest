#!/bin/bash
set -x

# Make a custom directory inside container
PREFIX=/home/custom
mkdir -p ${PREFIX}

# Install forestdb to custom directory needed by custom Python
cd /repo/forest
PYTHONPATH=${PREFIX}/lib/python3.6/site-packages \
    python setup.py install --prefix ${PREFIX}
cd -

# Check forestdb available inside container
PYTHONPATH=${PREFIX}/lib/python3.6/site-packages ${PREFIX}/bin/forestdb -h

# Run custom update Python script
ls -l /database
ls -l /repo/forest
ls -l /s3
