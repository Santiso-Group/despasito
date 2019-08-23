API Documentation
=================

DESPASITO has been primarily designed as a command line tool but can be used as an imported package.

See examples directory or Input/Output documentation for input file structures.

Command Line
------------
In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

Imported Package
----------------
Once installed, DESPASITO can be easily imported with ``import despasito``.
Each module will then need to replicate our main command line function and call each modulein succession.

#. Provide ``input.json`` file name to ``readwrite_input.extract_calc_data`` to obtain the appropriate dictionaries for the equation of state (eos) object, and the thermodynamic calculation.
#. Use eos dictionary of bead types and parameters to generate eos object used in thermodynamic calculations
#. Choose calculation type and use eos object and thermodynamic dictionary to start calculation.

Some steps could be skipped if your script already contains the appropriate output of a given step.

