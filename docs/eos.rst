
Equations of State
=========================================================

Here we list list the available equations of state and the function used to access them. 

.. autofunction:: despasito.equations_of_state.initiate_eos


.. _eos-types:

Available EOS
-------------

.. toctree::
   :maxdepth: 1

   saft
   peng-robinson

Supporting Modules
---------------------

.. autosummary::
   :toctree: _autosummary

   despasito.equations_of_state.eos_toolbox
   despasito.equations_of_state.combining_rule_types
   despasito.equations_of_state.constants


Adding an EOS
-------------

Adding an EOS family is easily implemented by adding a new directory to the ``equations_of_state`` directory. A new EOS is then added by adding a module with the desired EOS inside that is derived from our EOS interface (shown below).

.. automodule::
   despasito.equations_of_state.interface
   :members:


