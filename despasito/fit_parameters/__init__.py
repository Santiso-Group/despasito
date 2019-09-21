"""
Fit Parameters
--------------

This package uses functions from input_output, equations_of_state, and thermodynamics to fit parameters to experimental data.

"""

import os
import numpy as np
from importlib import import_module
import scipy.optimize as spo
import logging

from . import fit_funcs as ff
from . import data_classes

def fit(eos, thermo_dict):
    """
    Fit defined parameters for equation of state object with given experimental data. Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    eos : obj
        Equation of state output that writes pressure, max density, and chemical potential
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting.
  
    Returns
    -------
        Output file saved in current working directory
    """

    logger = logging.getLogger(__name__)

    # Extract relevent quantities from thermo_dict
    keys_del = []
    for key, value in thermo_dict.items():
        # Extract inputs
        if key == "opt_params":
            opt_params  = thermo_dict["opt_params"]
        elif key == "exp_data":
            exp_data = thermo_dict["exp_data"]
        # Optional inputs
        elif key == "beadparams0":
            beadparams0 = thermo_dict["beadparams0"]
        elif key == "output_file":
            output_file = thermo_dict["output_file"]
        else:
            continue
        keys_del.append(key)

    for key in keys_del:
        thermo_dict.pop(key,None)

    if list(thermo_dict.keys()):
        print("Note: thermo_dict keys: %s, were not used." % ", ".join(list(thermo_dict.keys())))    

    # Reformat exp. data into formatted dictionary
    exp_dict = {}
    pkgpath = os.path.dirname(data_classes.__file__)
    type_list = [f for f in os.listdir(pkgpath) if ".py" in f]
    type_list = type_list.remove("__init__.py")

    for key, data_dict in exp_data.items():
        fittype = data_dict["name"]
        try:
            exp_module = import_module("."+fittype,package="despasito.fit_parameters.data_classes")
            data_class = getattr(exp_module, "Data")
        except:
            if not type_list: "No fit types"
            elif len(type_list) == 1: tmp = type_list[0]
            else: tmp = ", ".join(type_list)
            raise ImportError("The experimental data type, '"+fittype+"', was not found\nThe following calculation types are supported: "+tmp)

        try:
            instance = data_class(data_dict)
            exp_dict[key] = instance
        except:
            raise AttributeError("Data set, %s, did not properly initiate object" % (key))

    # Generate initial guess for parameters if none was given
    if "beadparams0" not in thermo_dict:
        beadparams0 = eos.param_guess(opt_params["fit_params"])
  
    # NoteHere: how is this array generated?
#    stepmag = np.array([550.0, 26.0, 4.0e-10, 0.45, 500.0, 150.0e-30, 550.0])

    # Run Parameter Fitting
    try:
#        custombasinstep = ff.BasinStep(stepmag, stepsize=0.1)
#        res = spo.basinhopping(ff.compute_SAFT_obj,
#                       beadparams0,
#                       niter=500,
#                       T=0.2,
#                       stepsize=0.1,
#                       minimizer_kwargs={
#                           "args": (opt_params, eos, exp_dict),"method": 'nelder-mead', "options": {'maxiter': 200}},
#                       take_step=custombasinstep,
#                       disp=True)
        # Why doesn't this work???
        # "args": (opt_params, eos, exp_dict, output_file),"method": 'nelder-mead', "options": {'maxiter': 200}},
        res = spo.basinhopping(ff.compute_SAFT_obj,
                       beadparams0,
                       niter=500,
                       T=0.2,
                       stepsize=0.1,
                       minimizer_kwargs={
                           "args": (opt_params, eos, exp_dict),"method": 'nelder-mead', "options": {'maxiter': 200}},
                       disp=True)
    except:
        raise TypeError("The parameter fitting failed")

