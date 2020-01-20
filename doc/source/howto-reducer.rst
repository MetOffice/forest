

Write a reducer
---------------

A reducer is a pure function that takes the current state and an action
and produces a new state.

.. code-block::

   def reducer(state: State, action: Action) -> State:
       return state

It's job is to incorporate the information described by the action into the
application state.

For example, say we wanted to store a list of datasets in the state. The list
can be used to populate dropdowns in the user interface. The first thing
to do is to create an action that encapsulates an instruction to change the
state

.. code-block::

   action = {'kind': 'ADD_DATASET', 'payload': 'Lightning'}

Then we need to think about where in the state this information belongs, for
example, we could store datasets as a list of strings under the key ``dataset``.

.. code-block::

   state = {'datasets': ['Lightning', 'Satellite', etc.]}

Now we have encoded our message and know where to find our information inside
the state. All that is missing is a way of going from one state to the next.
Introducing, the reducer, a function that can do that one-trick.

.. code-block::

   def datasets_reducer(state, action):
       state = copy.deepcopy(state)
       if action['kind'] == 'ADD_DATASET':
           dataset = action['payload']
           datasets = state.get('datasets', []) + [dataset]
           state['datasets'] = datasets
       return state

.. note:: A reducer must always return a new state, modifying states by
          reference introduces a side-effect that violates purity and
          leads to hard to track down bugs

We now have all of the ingredients needed to continually update our state.

.. code-block::

   state = {}
   state = datasets_reducer(state, add_dataset('A'))
   state = datasets_reducer(state, add_dataset('B'))
   state = datasets_reducer(state, add_dataset('C'))
   # state -> {'datasets': ['A', 'B', 'C']}


This may feel like a long winded approach with plenty of boilerplate and you
would be correct. However, the benefits far outweigh the additional typing

   - Easy to unit test, no side-effects or complicated mocking needed
   - Behaviour and state separated simpler mental model
   - Single source of truth, state represents full information needed
     to configure application
   - Ability to undo, replay and reload application at will
   - Decoupled components, a view does not care how the state came to be
     it simply reacts to the data presented to it

