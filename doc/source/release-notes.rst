Release notes
=============

Features and bugs that have been added/removed are listed
here to aid future users and maintainers.

0.5.0
-----

- Add template substitution using environment
  variables or ``--var 'key:value'``
  when using ``--config-file`` flag. E.g.
  using ``forest --var prefix:/some/dir --config-file some.yaml``

.. code-block:: yaml

   # some.yaml
   files:
     - label: UM
       pattern: ${HOME}/file.nc
     - label: RDT
       pattern: ${prefix}/file.json

Would be equivalent to the following file

.. code-block:: yaml

   # some.yaml.processed
   files:
     - label: UM
       pattern: /Users/Bob/file.nc
     - label: RDT
       pattern: /some/dir/file.json

The intention is to remove the implicit nature of ``--directory`` flag
in favour of explicit syntax inside the config file
