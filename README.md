# FOREST meteorology visualisation tool

This repository hosts the code to visualise forecast model output and observation data in a web portal, as well as the scripts and configuration files to deploy the server infrastructure.

## Background Info
TODO

## Installation
The forest tool requires three things to run:
* access to the relevant data
* the correct dependencies (python  and python libraries) installed
* access to the forest source code

### Data access

The data for the forest tool is stored in the AWS S3 storage bucket located at
s3://stephen-sea-public-london/. To make the data accessible to the
application, the recommended approach is to mount the bucket using a FUSE
tool such as [Goofys](https://github.com/kahing/goofys).

### Dependencies
Forest is written in the Python language. Python 3.6 is the version currently
supported. Bokeh server is used to serve the apps.
It requires the following third party tools to be installed:
* matplotlib
* bokeh
* iris
* cartopy

The recommended way to install python and the dependencies is to use
[anaconda](https://www.anaconda.com/download/#linux). The libraries can then
be installed using conda install.

### Source code
The source code should be made available to run Forest by cloning the
repository. When you run the server using the bokeh serve command, the
forest_lib directory will need to be available on the PYTHONPATH so that
the relevant modules can be imported.

## Running locally

## Deploying
TODO

