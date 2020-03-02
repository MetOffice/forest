
Add a feature toggle
--------------------

It often happens that a feature is mostly done but not quite
ready for a wider audience. Feature toggles give developers and
user experience wizards the ability to test features early in
the cycle and give them insight into performance and usability.


.. code-block:: yaml

   features:
     foo: true
     bar: false

To access these settings in main.py use the following syntax.

.. code-block:: python

   if config.features['foo']:
       # Do foo feature


As easy as that.
