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
    r"""
    Fit defined parameters for equation of state object with given experimental data. Each set of experimental data is converted to an object with the built in ability to evaluate its part of objective function.
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations and parameter fitting.

        - opt_params (dict) - Parameters used in basin fitting algorithm.

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
            - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).

        - bounds (numpy.ndarray) - List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
        - exp_data (dict) - This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each dictionary is converted into an object and saved back to this structure before parameter fitting begins. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - name (str) - One of the supported data type objects to fit parameters

        - beadparams0 (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from eos object.
        - output_file (str), Optional - Output file name
        - basin_dict (dict), Optional - kwargs used in scipy.optimize.basinhopping

            - niter (int) - Number of basin hopping iterations
            - T (float) - Temperature parameter, should be comparable to separation between local minima (i.e. the “height” of the walls separating values).
            - take_step (obj) - Callable object that includes custom routine that produces attribute stepsize
            - niter_success (int) - Stop run if minimum stays the same for this many iterations

        - minimizer_dict (dict), Optional - Dictionary used to define minimization type and the associated options.

            - method (str) - Method available to scipy.optimize.minimize
            - options (dict) - This dictionary contains the kwargs available to the chosen method
  
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
        elif key == "bounds":
            bounds = thermo_dict["bounds"]
        else:
            continue
        keys_del.append(key)

    for key in keys_del:
        thermo_dict.pop(key,None)

    if list(thermo_dict.keys()):
        print("Note: thermo_dict keys: {}, were not used.".format(", ".join(list(thermo_dict.keys()))))    

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
            if not type_list: raise ImportError("No fit types")
            elif len(type_list) == 1: tmp = type_list[0]
            else: tmp = ", ".join(type_list)
            raise ImportError("The experimental data type, '"+fittype+"', was not found\nThe following calculation types are supported: "+tmp)

        try:
            instance = data_class(data_dict)
            exp_dict[key] = instance
        except:
            raise AttributeError("Data set, {}, did not properly initiate object".format(key))

    # Generate initial guess for parameters if none was given
    if "beadparams0" not in thermo_dict:
        beadparams0 = eos.param_guess(opt_params["fit_params"])

    # Options for basin hopping
    basin_dict = {"niter": 10, "T": 0.5, "niter_success": 3}
    if "basin_dict" in thermo_dict:
        for key, value in thermo_dict["basin_dict"].items():
            basin_dict[key] = value

    # Set up options for minimizer in basin hopping
    minimizer_dict = {"method": 'nelder-mead', "options": {'maxiter': 50}}
    if "minimizer_dict" in thermo_dict:
        for key, value in thermo_dict["minimizer_dict"].items():
            if key == "method":
                minimizer_dict[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    minimizer_dict[key][key2] = value2
  
    # NoteHere: how is this array generated? stepmag = np.array([550.0, 26.0, 4.0e-10, 0.45, 500.0, 150.0e-30, 550.0])
    try:
        if "stepsize" in basin_dict:
        	stepsize = basin_dict["stepsize"]
        else:
        	stepsize = 0.1
        stepmag = np.transpose(np.array(bounds))[1]
        basin_dict["take_step"] = ff.BasinStep(stepmag, stepsize=stepsize)
        custombounds = ff.BasinBounds(bounds)
    except:
    	raise TypeError("Could not initialize BasinStep and/or BasinBounds")

    # Run Parameter Fitting
    try:
        result = spo.basinhopping(ff.compute_SAFT_obj, beadparams0, **basin_dict, accept_test=custombounds, disp=True,
                       minimizer_kwargs={"args": (opt_params, eos, exp_dict),**minimizer_dict})

        logger.info("Fitting terminated:\n{}".format(result.message))
        logger.info("{}".format(result.message))
        logger.info("Best Fit Parameters")
        logger.info("    Obj. Value: {}".format(result.fun))
        for i in range(len(opt_params["fit_params"])):
            logger.info("    {} {}: {}".format(opt_params["fit_bead"],opt_params["fit_params"][i],result.x[i]))

    except:
        raise TypeError("The parameter fitting failed")

