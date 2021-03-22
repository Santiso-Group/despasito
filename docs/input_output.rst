
Input/Output
============

Package Inputs
--------------
Inputs are contained in .json files with instructions for thermodynamic calculations and paths to the .json parameter files for the Eos object. See documentation below, or :ref:`input-schema` for more information.

.. currentmodule:: despasito.input_output
.. autosummary::
   :toctree: _autosummary

   read_input

Package Outputs
---------------
Calculation outputs are saved within the current working directory. An alternative output file name can be defined in the input.json file for command line use or thermodynamics dictionary for imported use.

.. autosummary::
   :toctree: _autosummary

   write_output

