
Equations of State
=========================================================

Here we list list the available equations of state and the function used to access them. 

.. currentmodule:: despasito

.. autosummary::

   :toctree: _autosummary

   equations_of_state

Available EOS
-------------

.. autosummary::

   :toctree: _autosummary

   equations_of_state.saft.gamma_mie
   equations_of_state.saft.gamma_mie_funcs
   equations_of_state.saft.nojit_exts
   equations_of_state.saft.jit_exts

Adding an EOS
-------------

Adding an EOS family is easily implemented by adding a new directory to the ``equations_of_state`` directory. A new EOS is then added by adding a module with the desired EOS inside that is derived from our EOS interface (shown below).

.. note:: In the future, a USER folder will be included in our program so that users can collect all of their personal additions and modifications to DESPASITO in one location.

.. autosummary::

   :toctree: _autosummary

   equations_of_state.interface



