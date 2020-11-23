# -- coding: utf8 --

"""
Handles the primary functions

In any directory with the appropriate .json input files, run DESPASITO with ``python -m despasito input.json``

"""

import logging
import argparse

from .input_output import read_input
from .input_output import write_output
from .equations_of_state import eos as eos_mod
from .thermodynamics import thermo
from .parameter_fitting import fit

logger = logging.getLogger(__name__)

def get_parser():

    ## Define parser functions and arguments
    parser = argparse.ArgumentParser(description="DESPASITO: Determining Equilibrium State and Parametrization Application for SAFT, Intended for Thermodynamic Output.  This is an open-source application for thermodynamic calculations and parameter fitting for the Statistical Associating Fluid Theory (SAFT) EOS and SAFT-:math:`\gamma`-Mie coarse-grained simulations.")
    parser.add_argument("-i", "--input", dest="input", help="Input .json file with calculation instructions and path(s) to equation of state parameters. See documentation for explicit explanation. Compile docs or visit https://despasito.readthedocs.io")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Verbose level: repeat up to three times for Warning, Info, or Debug levels.")
    parser.add_argument("--log", nargs='?', dest="logFile", default="despasito.log", help="Output a log file. The default name is despasito.log.")
    parser.add_argument("-n", "--ncores", dest="ncores", type=int, help="Set the number of cores used. A value of -1 will request all possible resources.",default=1)
    parser.add_argument("-p", "--path", default=".", help="Set the location of the data/library files (e.g. SAFTcross, etc.) for despasito to look for")
    parser.add_argument("--numba", action='store_true', help="Turn on Numba's JIT compilation for accelerated computation")
    parser.add_argument("--python", action='store_true', help="Remove default fortran module for association site calculations.")
    parser.add_argument("--cython", action='store_true', help="Turn on Cython for accelerated computation")

    return parser

def run(filename="input.json", path=".", numba=False, cython=False, python=False, **kwargs):
    """ Main function for running despasito calculations. All inputs and settings should be in the supplied JSON file(s).
    """

    #read input file (need to add command line specification)
    logger.info("Begin processing input file: %s" % filename)
    eos_dict, thermo_dict, output_file = read_input.extract_calc_data(filename, path, **kwargs)

    thermo_dict['mpObj'] = kwargs['mpObj']
    eos_dict["numba"] = numba
    eos_dict["cython"] = cython
    eos_dict["python"] = python

    if output_file:
        file_dict = {"output_file":output_file}
    else:
        file_dict = {"output_file": "despasito_out.txt"}

    logger.debug("EOS dict:")
    for key, value in eos_dict.items():
        logger.debug("    {}: {}".format(key, value))
    logger.debug("Thermo dict:")
    for key, value in thermo_dict.items():
        logger.debug("    {}: {}".format(key, value))
    logger.info("Finish processing input file: {}".format(filename))

    # Run either parametrization or thermodynamic calculation
    fitting_opts = ["objective_method", "nan_number", "nan_ratio"]

    if "opt_params" in thermo_dict:
        for key,exp_dict in thermo_dict["exp_data"].items():
            eos_dict = exp_dict["eos_dict"]
            thermo_dict["exp_data"][key].pop("eos_dict", None)
            thermo_dict["exp_data"][key]["eos_obj"] = eos_mod(**eos_dict)
            for key2 in fitting_opts:
                if key2 in thermo_dict:
                    thermo_dict["exp_data"][key][key2] = thermo_dict[key2]
        logger.info("Initializing parametrization procedure")

        output_dict = fit(**thermo_dict.copy())
        logger.info("Finished parametrization")
        write_output.writeout_fit_dict(output_dict,**file_dict)
    else:
        eos = eos_mod(**eos_dict)
        logger.info("Initializing thermodynamic calculation")
        output_dict = thermo(eos, **thermo_dict.copy())
        logger.info("Finished thermodynamic calculation")
        try:
            write_output.writeout_thermo_dict(output_dict,thermo_dict["calculation_type"],**file_dict)
        except:
            logger.info("Final Output: {}".format(output_dict))

    if thermo_dict['mpObj'].flag_use_mp:
        thermo_dict['mpObj'].end_pool()
    
