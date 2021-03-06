

Write a reducer
---------------

A reducer is a pure function that given a state and an action produces a
new state.

.. image:: reducer-signature.png
   :width: 400
   :align: center

Or expressed in code, it has the following signature.

.. code:: python

   def reducer(state: State, action: Action) -> State:
       # Reduction logic goes here
       return state

It's only responsibility is to incorporate information described by an action
into the state.

For example, say we wanted to store a list of datasets in the state. The list
could then be used to populate dropdowns in the user interface. The first thing
to do is to create an action that describes an instruction to change the
state

.. code:: python

   action = {'kind': 'ADD_DATASET', 'payload': 'Lightning'}

Writing a dict literal every time we want to use this action is a bit
long-winded. How about we wrap it into a tiny function to save our fingers?

.. code:: python

   def add_dataset(label):
       # 'kind' and 'payload' are used by convention throughout the codebase
       return {'kind': 'ADD_DATASET', 'payload': label}

Nice touch. Now we need to think about where this information belongs in the state, for
example, we could store a list of strings under the dictionary key ``'datasets'``.

.. code:: python

   state = {'datasets': ['Lightning', 'Satellite', etc.]}

Great, we have encoded our action and have a good idea where to find our information inside
the state. All that is missing is a way of going from a state without our
information to a state with our information. We need to combine our action
and our state in some way. Enter the reducer.

.. code:: python

   def list_reducer(state, action):
       # A simple example of a reducer
       state = copy.deepcopy(state)
       if action['kind'] == 'ADD_DATASET':
           dataset = action['payload']
           datasets = state.get('datasets', []) + [dataset]
           state['datasets'] = datasets
       return state

.. warning:: A reducer must always return a new state. Modifying a reference to
          a state in-place introduces side-effects that generate hard to
          diagnose bugs. States are considered to be immutable

We now have all of the ingredients needed to continually update our state. For
example repeated application of our reducer builds a list of datasets.

.. code:: python

   >>> # illustrate repeat reducer application
   >>> state_0 = {}
   >>> state_1 = list_reducer(state_0, add_dataset('A'))
   >>> state_2 = list_reducer(state_1, add_dataset('B'))
   >>> state_3 = list_reducer(state_2, add_dataset('C'))
   >>> state_3
   {'datasets': ['A', 'B', 'C']}


If you are used to object-oriented designs this approach may seem a bit long
winded. It is. Luckily for us, the :class:`forest.redux.Store` takes care
of the boilerplate. Repeated application of the reducer and usage of
middleware is abstracted away from us so we only need to implement
the methods.

.. code:: python

   >>> # Using Store
   >>> from forest.redux import Store
   >>> store = Store(list_reducer)
   >>> for letter in ['A', 'B', 'C']:
   ...     store.dispatch(add_dataset(letter))
   ...
   >>> store.state
   {'datasets': ['A', 'B', 'C']}

That said, it does take a little more time to decompose our
thoughts into actions, states and reducers. However after going through
that effort we gain many nice features

   - Easy to unit test, no side-effects or complicated mocking needed
   - Behaviour and state separated simpler mental model
   - Single source of truth, state represents full information needed
     to configure application
   - Ability to undo, replay and reload application at will
   - Decoupled components, a view does not care how the state came to be
     it simply reacts to the data presented to it


.. note:: :meth:`forest.redux.combine_reducers` provides a simple way
   to compose multiple reducers into a single function
