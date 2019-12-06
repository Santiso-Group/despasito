"""
Handles the primary functions

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

"""

import logging
import argparse

from .input_output import readwrite_input
from .equations_of_state import eos as eos_mod
from .thermodynamics import thermo
from .fit_parameters import fit

def commandline_parser():
    ## Define parser functions and arguments
    parser = argparse.ArgumentParser(description="DESPASITO: Determining Equilibrium State and Parametrization: Application for SAFT, Intended for Thermodynamic Output.  This is an open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) EOS and SAFT-𝛾-Mie coarse-grained simulations.")
    parser.add_argument("-i", "--input", dest="input", help="Input .json file with calculation instructions and path(s) to equation of state parameters. See documentation for explicit explanation. Compile docs or visit https://despasito.readthedocs.io")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose level: repeat up to three times for Warning, Info, or Debug levels.")
    parser.add_argument("--log", nargs='?', dest="logFile", default="despasito.log", help="Output a log file. The default name is despasito.log.")
    parser.add_argument("-t", "--threads", dest="threads", type=int, help="**This hasn't been implemented yet.** Set the number of threads used.",default=1)
    parser.add_argument("-p", "--path", default=".", help="Set the location of the data/library files (e.g. SAFTcross, etc.) for despasito to look for")
    parser.add_argument("--jit", action='store_true', default=0, help="Turn on Numba's JIT compilation for accelerated computation")

    return parser

def run(filename="input.json", path=".", **args):

    """ Main function for running despasito calculations. All inputs and settings should be in the supplied JSON file(s).
    """

    logger = logging.getLogger(__name__)
    
    #read input file (need to add command line specification)
    logger.info("Begin processing input file: %s" % filename)
    eos_dict, thermo_dict, output_file = readwrite_input.extract_calc_data(filename, path, **args)
    eos_dict['jit'] = args['jit']

    if output_file:
        file_dict = {"output_file":output_file}
    else:
        file_dict = {"output_file": "despasito_out.txt"}

    logger.debug("EOS dict:", eos_dict)
    logger.debug("Thermo dict:", thermo_dict)
    logger.info("Finish processing input file: {}".format(filename))
    
    eos = eos_mod(**eos_dict)
    
    # Run either parametrization or thermodynamic calculation
    if "opt_params" in list(thermo_dict.keys()):
        logger.info("Initializing parametrization procedure")
        fit(eos, thermo_dict)
        #output = fit(eos, thermo_dict)
        logger.info("Finished parametrization")
        # readwrite_input.writeout_dict(output_dict,**file_dict)
    else:
        logger.info("Initializing thermodynamic calculation")
        output_dict = thermo(eos, thermo_dict)
        logger.info("Finished thermodynamic calculation")
        readwrite_input.writeout_dict(output_dict,thermo_dict["calculation_type"],**file_dict)
    
