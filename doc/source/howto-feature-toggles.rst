
Add a feature toggle
--------------------

It often happens that a feature is mostly done but not quite
ready for a wider audience. Feature toggles give developers and
user experience wizards the ability to test features early in
the cycle and give them insight into performance and usability.

**Static feature flags**

.. code-block:: yaml

   features:
     foo: true
     bar: false

To access these settings in main.py use the following syntax.

.. code-block:: python

   if forest.data.FEATURE_FLAGS['foo']:
       # Do foo feature


As easy as that.

**Dynamic feature flags**

To add more sophisticated dynamic feature toggles it is possible to
specify an ``entry_point`` that runs general purpose Python code to
determine the feature flags dictionary.


.. code-block:: yaml

   plugins:
     feature:
       entry_point: lib.mod.func

The string ``lib.mod.func`` is parsed into an import statement to
import ``lib.mod`` and a call of the ``func`` method. This is very
similar to how setup.py wires up commands.


.. warning:: Since the entry_point could point to arbitrary Python code
             make sure this feature is only used with trusted source code
