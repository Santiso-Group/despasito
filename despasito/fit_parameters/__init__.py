"""
Fit Parameters
--------------

This package uses functions from input_output, equations_of_state, and thermodynamics to fit parameters to experimental data.

Input.json files have a different dictionary structure that is processed by :func:`~despasito.input_output.read_input.process_param_fit_inputs`

"""

import sys
import os
import numpy as np
from importlib import import_module
import logging

from . import fit_funcs as ff
from . import data_classes


def fit(eos, thermo_dict):
    r"""
    Fit defined parameters for equation of state object with given experimental data. 

    Each set of experimental data is converted to an object with the built in ability to evaluate its part of objective function.
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations and parameter fitting.

        - opt_params (dict) - Parameters used in global fitting algorithm.

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
            - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
            - beadparams0 (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from eos object.
            - global_method (str), Optional - default: 'basinhopping', Global optimization method used to fit parameters. See :func:`~despasito.fit_parameters.fit_funcs.global_minimization`.

        - bounds (numpy.ndarray) - List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
        - exp_data (dict) - This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each dictionary is converted into an object and saved back to this structure before parameter fitting begins. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - name (str) - One of the supported data type objects to fit parameters

        - global_dict (dict), Optional - kwargs used in global optimization method. See :func:`~despasito.fit_parameters.fit_funcs.global_minimization`.

            - niter (int) - default: 10, Number of basin hopping iterations
            - T (float) - default: 0.5, Temperature parameter, should be comparable to separation between local minima (i.e. the “height” of the walls separating values).
            - niter_success (int) - default: 3, Stop run if minimum stays the same for this many iterations
            - stepsize (float) - default: 0.1, Maximum step size for use in the random displacement. We use this value to define an object for the `take_step` option that includes a custom routine that produces attribute stepsizes for each parameter.

        - minimizer_dict (dict), Optional - Dictionary used to define minimization type and the associated options.

            - method (str) - Method available to scipy.optimize.minimize
            - options (dict) - This dictionary contains the kwargs available to the chosen method
  
    Returns
    -------
    Output file saved in current working directory
    """

    logger = logging.getLogger(__name__)

    # Extract relevant quantities from thermo_dict
    dicts = {}

    keys_del = []
    for key, value in thermo_dict.items():
        # Extract inputs
        if key == "opt_params":
            opt_params = value
        elif key == "exp_data":
            exp_data = value
        # Optional inputs
        elif key == "bounds":
            bounds = value
        elif key == "minimizer_dict":
            dicts["minimizer_dict"] = value
        elif key == "global_dict":
            dicts["global_dict"] = value
        else:
            continue
        keys_del.append(key)

    for key in keys_del:
        thermo_dict.pop(key, None)

    if list(thermo_dict.keys()):
        logger.info(
            "Note: thermo_dict keys: {}, were not used.".format(
                ", ".join(list(thermo_dict.keys()))
            )
        )

    if "bounds" not in thermo_dict:
        bounds = np.empty((len(opt_params["fit_params"]), 2))
    bounds = ff.check_parameter_bounds(opt_params, eos, bounds)

    # Reformat exp. data into formatted dictionary
    exp_dict = {}
    pkgpath = os.path.dirname(data_classes.__file__)
    type_list = [f for f in os.listdir(pkgpath) if ".py" in f]
    type_list = type_list.remove("__init__.py")

    for key, data_dict in exp_data.items():
        fittype = data_dict["name"]
        try:
            exp_module = import_module(
                "." + fittype, package="despasito.fit_parameters.data_classes"
            )
            data_class = getattr(exp_module, "Data")
        except:
            if not type_list:
                raise ImportError("No fit types")
            elif len(type_list) == 1:
                tmp = type_list[0]
            else:
                tmp = ", ".join(type_list)
            raise ImportError(
                "The experimental data type, '"
                + fittype
                + "', was not found\nThe following calculation types are supported: "
                + tmp
            )

        try:
            instance = data_class(data_dict)
            exp_dict[key] = instance
            logger.info("Initiated exp. data object: {}".format(instance.name))
        except:
            raise AttributeError(
                "Data set, {}, did not properly initiate object".format(key)
            )

    # Generate initial guess for parameters if none was given
    if "beadparams0" in opt_params:
        beadparams0 = opt_params["beadparams0"]
    else:
        beadparams0 = ff.initial_guess(opt_params, eos)
    logger.info("Initial guess in parameters: {}".format(beadparams0))

    # Check global optimization method
    if "global_method" in opt_params:
        global_method = opt_params["global_method"]
    else:
        global_method = "basinhopping"

    # Run Parameter Fitting
    try:
        result = ff.global_minimization(
            global_method,
            beadparams0,
            bounds,
            opt_params["fit_bead"],
            opt_params["fit_params"],
            eos,
            exp_dict,
            **dicts
        )

        print(result.keys())
        logger.info("Fitting terminated:\n{}".format(result.message))
        logger.info("Best Fit Parameters")
        logger.info("    Obj. Value: {}".format(result.fun))
        for i in range(len(opt_params["fit_params"])):
            logger.info(
                "    {} {}: {}".format(
                    opt_params["fit_bead"], opt_params["fit_params"][i], result.x[i]
                )
            )

    except:
        raise TypeError("The parameter fitting failed")

    return {
        "fit_bead": opt_params["fit_bead"],
        "fit_parameters": opt_params["fit_params"],
        "final_parameters": result.x,
        "objective_value": result.fun,
    }
