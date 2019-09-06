"""
Handles the primary functions

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

"""

import sys
import os
from .input_output import readwrite_input
from .equations_of_state import eos
from .thermodynamics import thermo
from .fit_parameters import fit

# This file is intended to be run in the desired file to run the calculation. All inputs and settings should be in those files.

# Settings that should be replaced
meth = "brent"

#read input file (need to add command line specification)
eos_dict, thermo_dict = readwrite_input.process_commandline(sys.argv)

eos = eos("saft.gamma_mie",**eos_dict)

# Run either parameterization or thermodynamic calculation
if "opt_params" in list(thermo_dict.keys()):
    fit(eos, thermo_dict)
else:
    thermo(eos, thermo_dict)

