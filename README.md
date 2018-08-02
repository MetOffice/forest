# FOREST meteorology visualisation tool 

[![Build Status](https://travis-ci.com/informatics-lab/forest.svg?branch=master)](https://travis-ci.com/informatics-lab/forest)

This repository hosts the code to visualise forecast model output and observation data in a web portal, as well as the scripts and configuration files to deploy the server infrastructure.

## Background Info
See [wiki pages](https://github.com/met-office-lab/SEAsia/wiki) for more info.

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
`forest_lib` directory will need to be available on the PYTHONPATH so that
the relevant modules can be imported.

## Running locally

To run you must ensure that the forest_lib directory is in you python PATH.

`export PYTHONPATH=forest_lib/:$PYHTONPATH`

If S3 is mounted run:
`bokeh serve bokeh_apps/plot_sea_two_model_comparison`

If S3 isn't mounted run to download the data as required:

`FOREST_DOWNLOAD_DATA=True bokeh serve bokeh_apps/plot_sea_two_model_comparison`



## Deploying
TODO

## Documentation

The documentation for the forest library uses the Python package Sphinx
and can be built from inside the doc directory

```sh
> cd doc/
> make html
```

Please see [Sphinx official documentation](http://www.sphinx-doc.org/en/master/) for
further details

## Testing

The test suite uses Python's builtin unittest module to test the Python
source code and node package manager [https://www.npmjs.com/](npm) to unit test the
JavaScript callback code

```sh
> python -m unittest discover
```

**Note:** To test **forest** it must be available to the Python interpreter
          invoking the tests. A simple way to achieve this is to include
          the forest directory in your `PYTHONPATH`

### Node package manager

The Python unit test suite has a test that calls `npm test` to run the
JavaScript unit tests. If `npm` is not available this test will fail. If
the **node_modules** directory is missing inside the **forestjs** directory, run
`npm install` inside **forestjs** to install `mocha` and `chai` libraries
needed by `npm test` to run the JS unit tests

