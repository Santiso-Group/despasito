"""
run.py
DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output

Handles the primary functions
"""

import sys
import os
from .eos import eos
from .thermo import thermo
from . import param_fit
from .input_output import readwrite_input

# This file is intended to be run in the desired file to run the calculation. All inputs and settings should be in those files.

# Settings that should be replaced
meth = "brent"

#read input file (need to add command line specification)
calctype, eos_dict, thermo_dict = readwrite_input.process_commandline(sys.argv)

eos = eos("saft.gamma_mie",**eos_dict)
thermo(calctype, eos, thermo_dict)

# Try to implement this
#try:
#    calctype=input_dict['parameterization_type']
#    thermo()
#except:
#    calctype='none'
#    print('No calculation type specified')


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
