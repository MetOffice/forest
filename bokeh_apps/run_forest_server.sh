#!/usr/bin/env bash

export S3_ROOT=$1
bokeh serve --port=$2 $3