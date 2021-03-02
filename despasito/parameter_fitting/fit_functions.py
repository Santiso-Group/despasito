import os
import numpy as np
import logging
from inspect import getmembers, isfunction
from scipy.optimize import NonlinearConstraint, LinearConstraint

from . import constraint_types as constraints_mod
from . import global_methods as global_methods_mod
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


def initial_guess(optimization_parameters, Eos):
    r"""
    Extract initial guess in fit parameters from EOS object.

    These values were taken from the EOSgroup file.

    Parameters
    ----------
    optimization_parameters : dict
        Parameters used in basin fitting algorithm

        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of bead_configuration
        - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).

    Eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.

    Returns
    -------
    parameters_guess : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
        
    """

    # Update bead_library with test parameters

    parameters_guess = np.ones(len(optimization_parameters["fit_parameter_names"]))
    for i, param in enumerate(optimization_parameters["fit_parameter_names"]):
        fit_parameter_names_list = param.split("_")
        if len(fit_parameter_names_list) == 1:
            parameters_guess[i] = Eos.guess_parameters(
                fit_parameter_names_list[0], [optimization_parameters["fit_bead"]]
            )
        elif len(fit_parameter_names_list) == 2:
            parameters_guess[i] = Eos.guess_parameters(
                fit_parameter_names_list[0],
                [optimization_parameters["fit_bead"], fit_parameter_names_list[1]],
            )
        else:
            raise ValueError(
                "Parameters for only one bead are allowed to be fit. Multiple underscores in a parameter name suggest more than one bead type in your fit parameter name, {}".format(
                    param
                )
            )

    return parameters_guess


def check_parameter_bounds(optimization_parameters, Eos, bounds):
    r"""
    Check that provided parameter bounds are within Eos reasonable limits
    
    Parameters
    ----------
    optimization_parameters : dict
        Parameters used in basin fitting algorithm
    
        - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    
    Eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    bounds : numpy.ndarray
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    
    Returns
    -------
    new_bounds : list[tuple]
        Checked with Eos object method, this list has a length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    
    """

    new_bounds = [
        (0, 0) for x in range(len(optimization_parameters["fit_parameter_names"]))
    ]
    # Check boundary parameters to be sure they're in a reasonable range
    for i, param in enumerate(optimization_parameters["fit_parameter_names"]):
        fit_parameter_names_list = param.split("_")
        new_bounds[i] = tuple(
            Eos.check_bounds(fit_parameter_names_list[0], param, bounds[i])
        )

    return new_bounds


def consolidate_bounds(optimization_parameters):
    r"""
    Parse parameter bounds in the optimization_parameters dictionary.

    The resulting bounds form a 2D numpy array with a length equal to the number of parameters being fit.
    
    Parameters
    ----------
    optimization_parameters : dict
        Parameters used in basin fitting algorithm
    
        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of bead_configuration
        - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        - \*_bounds (list[float]), Optional - This list contains the minimum and maximum of the parameter from a parameter listed in fit_parameter_names, represented in place of the asterisk. See input file instructions for more information.
    
    Returns
    -------
    new_optimization_parameters : dict
        Parameters used in basin fitting algorithm
    
        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of bead_configuration
        - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        - bounds (numpy.ndarray) - List of lists of length two, of length equal to fit_parameter_names. If no bounds were given then the default parameter boundaries are [0,1e+4], else bounds given as \*_bounds in input dictionary are used.

    """

    if "fit_bead" not in optimization_parameters:
        raise ValueError(
            "optimization_parameters dictionary should include keyword, fit_bead, defining the name of the bead whose parameters are to be fit."
        )
    if "fit_parameter_names" not in optimization_parameters:
        raise ValueError(
            "optimization_parameters dictionary should include keyword, fit_parameter_names, defining the parameters to be fit."
        )

    new_optimization_parameters = {}
    new_optimization_parameters["bounds"] = [
        [0, 1e4] for x in range(len(optimization_parameters["fit_parameter_names"]))
    ]
    for key2, value2 in optimization_parameters.items():
        if "bounds" in key2:
            tmp = key2.replace("_bounds", "")
            if tmp in optimization_parameters["fit_parameter_names"]:
                logger.info(
                    "Accepted bounds for parameter, '{}': {}".format(tmp, value2)
                )
                ind = optimization_parameters["fit_parameter_names"].index(tmp)
                new_optimization_parameters["bounds"][ind] = value2
            else:
                logger.error(
                    "Bounds for parameter type '{}' were given, but this parameter is not defined to be fit.".format(
                        tmp
                    )
                )
        else:
            new_optimization_parameters[key2] = value2
            continue

    return new_optimization_parameters


def reformat_output(cluster):
    r"""
    Takes a list of lists that contain thermo output of lists and floats and reformats it into a 2D numpy array.
 
    Parameters
    ----------
    cluster : list[list[list/floats]]
        A list of lists, where the inner list is made up of lists and floats

    Returns
    -------
    matrix : numpy.ndarray
        A 2D matrix
    len_cluster : list
        a list of lengths for each of the columns (whether 1 for float, or len(list))
        
    """

    # if input is a list or array
    if len(cluster) == 1:
        matrix = np.transpose(np.array(cluster[0]))
        if not gtb.isiterable(cluster[0]):
            len_cluster = [1]
        else:
            len_cluster = [len(cluster[0])]
    # If list of lists or arrays
    else:

        # Obtain dimensions of final matrix
        len_cluster = []
        for i, tmp_cluster in enumerate(cluster):
            if gtb.isiterable(tmp_cluster[0]):
                len_cluster.append(len(tmp_cluster[0]))
            else:
                len_cluster.append(1)
        matrix_tmp = np.zeros([len(cluster[0]), sum(len_cluster)])

        # Transfer information to final matrix
        ind = 0
        for i, val in enumerate(cluster):
            try:
                matrix = np.zeros([len(val[0]), len(val)])
            except Exception:
                matrix = np.zeros([1, len(val)])
            for j, tmp in enumerate(
                val
            ):  # yes, this is a simple transpose, but for some reason a numpy array of np arrays wouldn't transpose
                matrix[:, j] = tmp
            l = len_cluster[i]
            if l == 1:
                matrix_tmp[:, ind] = np.array(matrix)
                ind += 1
            else:
                if len(matrix) == 1:
                    matrix = matrix[0]

                for j in range(l):
                    matrix_tmp[:, ind] = matrix[j]
                    ind += 1

        matrix = np.array(matrix_tmp)

    return matrix, len_cluster


def global_minimization(global_method, *args, **kwargs):
    r"""
    Fit defined parameters for equation of state object with given experimental data. 

    Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    global_method : str
        Global optimization method used to fit parameters. See supported :mod:`~despasito.parameter_fitting.global_methods`.
    parameters_guess : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional
        Kwargs of global optimization algorithm. See specific option in `global_methods.py` 
    minimizer_opts : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    constraints : dict, Optional
        This dictionary of constraint types and their arguments will be converted into the appropriate form for the chosen optimization method.

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    logger.info("Using global optimization method: {}".format(global_method))
    calc_list = [
        o[0]
        for o in getmembers(global_methods_mod)
        if (isfunction(o[1]) and o[0][0] != "_")
    ]
    try:
        func = getattr(global_methods_mod, global_method)
    except Exception:
        raise ImportError(
            "The global minimization type, '{}', was not found\nThe following calculation types are supported: {}".format(
                function, ", ".join(calc_list)
            )
        )

    output = func(*args, **kwargs)

    return output


def initialize_constraints(constraints, constraint_type):
    r"""
    A tuple of either constraint classes or dictionaries as required by scipy global optimization methods

    Parameters
    ----------
    constraints : dict
        This dictionary of constraint types and their arguments will be converted into the appropriate form for the chosen optimization method. Although the key can be anything, it must represent a dictionary. The keyword 'function' must be found in the dictionary and represent a valid function name from `constraint_types.py`, as well as the two keys, 'type' and 'args'. The 'args' are inputs into the functions (keys), and 'type' entries depends on 'constraint_type'.
    constraint_type : str
        Either 'dict' or 'class'. Changes the constraint to the specified form. 
 
        - function: Allowed types, see :mod:`~despasito.parameter_fitting.constraint_types` 
        - dict: Allowed types, "eq" or "ineq", eq means must be zero, ineq means it must be non-negative 
        - class: Allowed types, "nonlinear" or "linear", a kwargs keyword may also be added for the constraint class 

    Returns
    -------
    new_constraints : tuple
        A tuple of either constraint classes or dictionaries as required by scipy global optimization methods
        
    """

    calc_list = [
        o[0]
        for o in getmembers(constraints_mod)
        if (isfunction(o[1]) and o[0][0] != "_")
    ]

    new_constraints = []
    for const_type, kwargs in constraints.items():

        if "function" not in kwargs:
            raise ValueError("Constraint function type is not included")

        try:
            func = getattr(constraints_mod, kwargs["function"])
        except Exception:
            raise ImportError(
                "The constraint type, '{}', was not found\nThe following types are supported: {}".format(
                    function, ", ".join(calc_list)
                )
            )

        if "args" not in kwargs:
            raise ValueError(
                "Constraint function, {}, is missing arguements".format(
                    kwargs["function"]
                )
            )

        if constraint_type == "class":
            if "type" not in kwargs or kwargs["type"] in ["linear", "nonlinear"]:
                raise ValueError(
                    "Constraint, {}, does not have type. Type can be 'linear' or 'nonlinear'.".format(
                        kwargs["function"]
                    )
                )
            if kwargs["type"] == "linear":
                if "kwargs" not in kwargs:
                    output = LinearConstraint(func, args[0], args[1])
                else:
                    output = LinearConstraint(func, args[0], args[1], **kwargs)
            elif kwargs["type"] == "nonlinear":
                if "kwargs" not in kwargs:
                    output = NonlinearConstraint(func, args[0], args[1])
                else:
                    output = NonlinearConstraint(func, args[0], args[1], **kwargs)

        elif constraint_type == "dict":
            if "type" not in kwargs or kwargs["type"] in ["eq", "ineq"]:
                raise ValueError(
                    "Constraint, {}, does not have type. Type can be 'eq' or 'ineq'.".format(
                        kwargs["function"]
                    )
                )
            output = {"type": kwargs["type"], "function": func, "args": kwargs["args"]}
        else:
            raise ValueError("Constraint type {}, must be either 'class' or 'dict'.")

        new_constraints.append(output)

    return tuple(new_constraints)


def compute_obj(beadparams, fit_bead, fit_parameter_names, exp_dict, bounds):
    r"""
    Fit defined parameters for equation of state object with given experimental data. 

    Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    parameters_guess : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    # Update bead_library with test parameters

    if len(beadparams) != len(fit_parameter_names):
        raise ValueError(
            "The length of initial guess vector should be the same number of parameters to be fit."
        )

    logger.info(
        (" {}: {}," * len(fit_parameter_names)).format(
            *[val for pair in zip(fit_parameter_names, beadparams) for val in pair]
        )
    )

    # Compute obj_function
    obj_function = []
    for key, data_obj in exp_dict.items():
        try:
            data_obj.update_parameters(fit_bead, fit_parameter_names, beadparams)
            obj_function.append(data_obj.objective())
        except Exception:
            logger.exception(
                "Failed to evaluate objective function for {} of type {}.".format(
                    key, data_obj.name
                )
            )
            obj_function.append(np.inf)

    obj_total = np.nansum(obj_function)
    if obj_total == 0.0 and np.isnan(np.sum(obj_function)):
        obj_total = np.inf

    # Add penalty for being out of bounds for the sake of inner minimization
    for i, param in enumerate(beadparams):
        if param <= bounds[i][0]:
            logger.debug(
                "Adding penalty to {} parameter for being lower than range".format(
                    fit_parameter_names[i]
                )
            )
            obj_total += (1e3 * (param - bounds[i][0])) ** 8
        elif param >= bounds[i][1]:
            logger.debug(
                "Adding penalty to {} parameter for being higher than range".format(
                    fit_parameter_names[i]
                )
            )
            obj_total += (1e3 * (param - bounds[i][1])) ** 8

    # Write out parameters and objective functions for each dataset
    logger.info(
        "\nParameters: {}\nValues: {}\nExp. Data: {}\nObj. Values: {}\nTotal Obj. Value: {}".format(
            fit_parameter_names,
            beadparams,
            list(exp_dict.keys()),
            obj_function,
            obj_total,
        )
    )

    return obj_total


def obj_function_form(
    data_test,
    data0,
    weights=1.0,
    method="average-squared-deviation",
    nan_number=1000,
    nan_ratio=0.1,
):
    """
    Factory method of possible objective functions. 

    Note that if the result is np.nan, that point is removed from the list for the purposes of averaging.

    Parameters
    ----------
    data_test : numpy.ndarray
        Data that is being assessed. Array of data of the same length as `data_test`
    data0 : numpy.ndarray
        Reference data for comparison
    weights : (numpy.ndarray or float), Optional, default=1.0
        Can be a float or array of data of the same length as `data_test`. Allows the user to tune the importance of various data points.
    method : str, Optional, default="mean-squared-relative-error"
        Keyword used to choose the functional form. Can be:

        - average-squared-deviation: sum(((data_test-data0)/data0)**2)/N
        - sum-squared-deviation: sum(((data_test-data0)/data0)**2)
        - sum-squared-deviation-boltz: sum(((data_test-data0)/data0)**2 * exp((data_test_min-data_test)/abs(data_test_min))) [DOI: 10.1063/1.2181979]
        - sum-deviation-boltz: sum(((data_test-data0)/data0) * exp((data_test_min-data_test)/abs(data_test_min)))  [DOI: 10.1063/1.2181979]
        - percent-absolute-average-deviation: sum((data_test-data0)/data0)/N*100

    nan_ratio : float, Optional, default=0.1
        If more than "nan_ratio*100" percent of the calculated data failed to produce NaN, increase the objective function by the number of data_test entries that are NaN multiplied by nan_number.
    nan_number : float, Optional, default=1000
        If a thermodynamic calculation produces NaN, add this quantity to the objective value

    Returns
    -------
    obj_value : float
        Objective value given the calculated and reference information
    """

    if np.size(data0) != np.size(data_test):
        raise ValueError(
            "Input data of length, {}, must be the same length as reference data of length {}".format(
                len(data_test), len(data0)
            )
        )

    if np.size(weights) > 1 and np.size(weights) != np.size(data_test):
        raise ValueError(
            "Weight for data is provided as an array of length, {}, but must be length, {}.".format(
                len(weights), len(data_test)
            )
        )

    data_tmp = np.array(
        [
            (data_test[i] - data0[i]) / data0[i]
            for i in range(len(data_test))
            if not np.isnan((data_test[i] - data0[i]) / data0[i])
        ]
    )

    if method == "average-squared-deviation":
        obj_value = np.mean(data_tmp ** 2 * weights)

    elif method == "sum-squared-deviation":
        obj_value = np.sum(data_tmp ** 2 * weights)

    elif method == "sum-squared-deviation-boltz":
        data_min = np.min(data_tmp)
        obj_value = np.sum(
            data_tmp ** 2 * weights * np.exp((data_min - data_tmp) / np.abs(data_min))
        )

    elif method == "sum-deviation-boltz":
        data_min = np.min(data_tmp)
        obj_value = np.sum(
            data_tmp * weights * np.exp((data_min - data_tmp) / np.abs(data_min))
        )

    elif method == "percent-absolute-average-deviation":
        obj_value = np.mean(np.abs(data_tmp) * weights) * 100

    if len(data_tmp) == 0:
        obj_value = np.nan

    if len(data_test) != len(data_tmp):
        tmp = 1 - len(data_tmp) / len(data_test)
        if tmp > nan_ratio:
            obj_value += (len(data_test) - len(data_tmp)) * nan_number
            logger.debug(
                "Values of NaN were removed from objective value calculation, nan_ratio {} > {}, augment obj. value".format(
                    tmp, nan_ratio
                )
            )
        else:
            logger.debug(
                "Values of NaN were removed from objective value calculation, nan_ratio {} < {}".format(
                    tmp, nan_ratio
                )
            )

    return obj_value
