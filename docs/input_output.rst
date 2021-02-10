
Input/Output
============

Package Inputs
--------------
Inputs are .json files containing instructions for thermodynamic calculations and paths to the .json files of parameters for the Eos object. 

.. currentmodule:: despasito.input_output
.. autosummary::
   :toctree: _autosummary

   read_input

Package Outputs
---------------
Outputs are files saved to the current working directory containing thermodynamic calculations. Default file names are used, but can be defined in the input.json file for command line use or thermodynamics dictionary for imported use.

.. autosummary::
   :toctree: _autosummary

   write_output

