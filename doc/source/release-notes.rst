Release notes
=============

Features and bugs that have been added/removed are listed
here to aid future users and maintainers.

0.5.0
-----

- Add Python 3.8 support by modifying sqlite3 usage
- Add template substitution using environment
  variables or ``--var key value``
  when using ``--config-file`` flag. E.g.
  using ``forest --var prefix /some/dir --config-file some.yaml``

  Users can now specify their configuration files explicitly

.. code-block:: yaml
   :caption: example.yaml

   files:
     - label: UM
       pattern: ${HOME}/file.nc
     - label: RDT
       pattern: ${prefix}/file.json
