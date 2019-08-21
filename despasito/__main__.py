"""
Handles the primary functions

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

"""

import sys
import os
from .input_output import readwrite_input
from .equations_of_state import eos
from .thermodynamics import thermo
#from . import fit_parameters

# This file is intended to be run in the desired file to run the calculation. All inputs and settings should be in those files.

# Settings that should be replaced
meth = "brent"

#read input file (need to add command line specification)
calctype, eos_dict, thermo_dict = readwrite_input.process_commandline(sys.argv)

eos = eos("saft.gamma_mie",**eos_dict)
thermo(calctype, eos, thermo_dict)
print("hello!")

# Try to implement this
#try:
#    calctype=input_dict['parameterization_type']
#    thermo()
#except:
#    calctype='none'
#    print('No calculation type specified')

