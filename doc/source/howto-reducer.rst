

Write a reducer
---------------

A reducer is a pure function that given a state and an action produces a
new state.

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
       {'kind': 'ADD_DATASET', 'payload': label}

Nice touch. Now we need to think about where this information belongs in the state, for
example, we could store a list of strings under the key ``dataset``.

.. code:: python

   state = {'datasets': ['Lightning', 'Satellite', etc.]}

Great, we have encoded our action and have a good idea where to find our information inside
the state. All that is missing is a way of going from a state without our
information to a state with our information. We need to combine our action
and our state in some way. Enter the reducer.

.. code:: python

   def list_reducer(state, action):
       state = copy.deepcopy(state)
       if action['kind'] == 'ADD_DATASET':
           dataset = action['payload']
           datasets = state.get('datasets', []) + [dataset]
           state['datasets'] = datasets
       return state

.. note:: A reducer must always return a new state, modifying states by
          reference introduces a side-effect that violates purity and
          leads to hard to track down bugs

We now have all of the ingredients needed to continually update our state. For
example repeated application of our reducer builds a list of datasets.

.. code:: python

   >>> state_0 = {}
   >>> state_1 = list_reducer(state_0, add_dataset('A'))
   >>> state_2 = list_reducer(state_1, add_dataset('B'))
   >>> state_3 = list_reducer(state_2, add_dataset('C'))
   {'datasets': ['A', 'B', 'C']}


If you are used to object-oriented designs this approach may seem a bit long
winded. However, although it takes a little more time to decompose our
intentions we gain an awful lot of nice features.

   - Easy to unit test, no side-effects or complicated mocking needed
   - Behaviour and state separated simpler mental model
   - Single source of truth, state represents full information needed
     to configure application
   - Ability to undo, replay and reload application at will
   - Decoupled components, a view does not care how the state came to be
     it simply reacts to the data presented to it

