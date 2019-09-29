
Getting started
===============

Welcome to the complete guide to using FOREST. Learn how
to integrate FOREST into your existing work flow, build a
web portal or quickly view your model diagnostics alongside
observations.

Installation
============

FOREST is distributed via conda through the `conda-forge` channel

.. code-block:: sh

  :> conda install -c conda-forge forest -y

Full documentation for conda can be found here: https://docs.conda.io/en/latest/

Who is FOREST for?
==================

FOREST is intended to provide a step change to exploration and
routine monitoring of forecasting systems, technical and non-technical
users should be able to easily compare, interrogate and report on the
quality of forecasts.

While the primary intention of FOREST is to support research-mode activities
it should be trivial to use in an operational context.

Example - Unified model output
==============================

FOREST comes with example cases intended to get users off the ground
quickly, reading about a tool is all well and good but nothing compares
to hands on experience.

.. code-block:: python

   >>> import forest.example
   ... forest.example.build_all()

The above snippet can be used to populate the current working directory with
all of the inputs needed to run the `forest` command line interface

To display the unified model without any additional configuration simply
run the following command inside a shell prompt

.. code-block:: sh

  :> forest --show sample-um.nc


The above example shows how `forest` can be used in a similar mode to well-known
utilities, e.g. `xconv`, `ncview` etc. However, given we have a full Tornado
server running and the power of Python at our finger tips it would be
criminal to curtail our application. To go beyond vanilla `ncview` behaviour
try the following command:

.. code-block:: sh

  :> forest --show --file-type rdt sample-rdt.json

This should bring up a novel polygon geojson visualisation of satellite
RDT (rapidly developing thunderstorms). But wait, without the underlying
OLR (outgoing longwave radiation) layer the polygons by themselves are
of little value

.. code-block:: sh

  :> forest --show --file-type eida50 sample-eida50.nc

It seems we are beginning to outgrow the command line, wouldn't it be
nice if we could store our settings and use them in a reproducible way!

Open up `sample-config.yml` for an example of the settings that can be adjusted
to suit your particular use case.

.. code-block:: yaml

  files:
     - label: UM
       pattern: sample-um.nc
       locator: file_system
     - label: RDT
       pattern: sample-rdt.json
       locator: file_system
     - label: EIDA50
       pattern: sample-eida50.nc
       locator: file_system

Running the following command should load FOREST with a model diagnostic, satellite image and derived polygon product at the same time that can be simultaneously compared
