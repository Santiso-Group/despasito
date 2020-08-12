
import os
import numpy as np
import logging
from inspect import getmembers, isfunction
from scipy.optimize import NonlinearConstraint, LinearConstraint

from . import constraint_types as constraints_mod
from . import global_methods as global_methods_mod

logger = logging.getLogger(__name__)

def initial_guess(opt_params, eos):
    r"""
    Extract initial guess in fit parameters from EOS object. These values were taken from the EOSgroup file.

    Parameters
    ----------
    opt_params : dict
        Parameters used in basin fitting algorithm

        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
        - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).

    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.

    Returns
    -------
    beadparams0 : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
        
    """

    # Update beadlibrary with test paramters

    beadparams0 = np.ones(len(opt_params['fit_params']))
    for i, param in enumerate(opt_params['fit_params']):
        fit_params_list = param.split("_")
        if (fit_params_list[0] == "l" and fit_params_list[1] in ["a","r"]):
            fit_params_list[0] = "{}_{}".format(fit_params_list[0],fit_params_list[1])
            fit_params_list.remove(fit_params_list[1])
        if len(fit_params_list) == 1:
            beadparams0[i] = eos.param_guess(fit_params_list[0], [opt_params['fit_bead']])
        elif len(fit_params_list) == 2:
            beadparams0[i] = eos.param_guess(fit_params_list[0], [opt_params['fit_bead'], fit_params_list[1]])
        else:
            raise ValueError("Parameters for only one bead are allowed to be fit at one time. Please only list one bead type in your fit parameter name.")

    return beadparams0

def check_parameter_bounds(opt_params, eos, bounds):
    r"""
    Extract initial guess in fit parameters from EOS object. These values were taken from the EOSgroup file.
    
    Parameters
    ----------
    opt_params : dict
        Parameters used in basin fitting algorithm
    
        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
        - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    
    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    bounds : numpy.ndarray
        List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    
    Returns
    -------
    bounds : list[tuple]
        Checked with eos object method, this list has a length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    
    """

    new_bounds = [(0,0) for x in range(len(opt_params['fit_params']))]
    # Check boundary parameters to be sure they're in a reasonable range
    for i, param in enumerate(opt_params['fit_params']):
        new_bounds[i] = tuple(eos.check_bounds(opt_params['fit_bead'], param, bounds[i]))

    return new_bounds

def reformat_ouput(cluster):
    r"""
    Takes a list of lists that combine lists and floats and reformats it into a 2D numpy array.
 
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

    # Arrange data
    type_cluster = [type(x[0]) for x in cluster]

    # if input is a list or array
    if len(cluster) == 1:
        matrix = np.transpose(np.array(cluster[0]))
        if cluster[0][0] not in [list,np.ndarray,tuple]:
            len_cluster = [1]
        else:
            len_cluster = [len(cluster[0][0])]

    # If list of lists or arrays
    else:

        # Obtain dimensions of final matrix
        len_cluster = []
        for i,typ in enumerate(type_cluster):
            if typ in [list,np.ndarray,tuple]:
                len_cluster.append(len(cluster[i][0]))
            else:
                len_cluster.append(1)
        matrix_tmp = np.zeros([len(cluster[0]), sum(len_cluster)])

        # Transfer information to final matrix
        ind = 0
        for i, val in enumerate(cluster):
            matrix = np.transpose(np.array(val))
            l = len_cluster[i]
            if l == 1:
                matrix_tmp[:, ind] = np.array(matrix)
                ind += 1
            else:
                for j in range(l):
                    matrix_tmp[:, ind] = matrix[j]
                    ind += 1
        #matrix = np.transpose(np.array(matrix_tmp))
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
        Global optimization method used to fit parameters. Currently scipy.optimize methods 'basinhopping', 'differential_evolution', and 'brute' are supported.
    beadparams0 : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of beadconfig
    fit_params : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_dict : dict, Optional
        Kwargs of global optimization algorithm. See specific option in `global_methods.py` 
    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    constraints : dict, Optional
        This dicitonary of constraint types and their arguements will be converted into the appropriate form for the chosen optimizaiton method.

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    logger.info("Using global optimization method: {}".format(global_method))
    calc_list = [o[0] for o in getmembers(global_methods_mod) if (isfunction(o[1]) and o[0][0] is not "_")]
    try:
        func = getattr(global_methods_mod, global_method)
    except:
        raise ImportError("The global minimization type, '{}', was not found\nThe following calculation types are supported: {}".format(function,", ".join(calc_list)))

    output = func(*args, **kwargs)

    return output

def initialize_constraints(constraints, constraint_type):
    r"""
    A tuple of either constraint classes or dictionaries as required by scipy global optimization methods

    Parameters
    ----------
    constraints : dict
        This dicitonary of constraint types and their arguements will be converted into the appropriate form for the chosen optimizaiton method. The key must be a valid function name from `constraint_types.py`. This key in turn represents a dictionary with two keys, 'type' and 'args'. The 'args' are inputs into the functions (keys), and 'type' entries depends on 'constraint_type'.
    constraint_type : str
        Either 'dict' or 'class'. Changes the constraint to the specified form. 
 
        - dict: Allowed types, "eq" or "ineq", eq means must be zero, ineq means it must be non-negative 
        - class: Allowed types, "nonlinear" or "linear", a kwargs keyword may also be added for the constraint class 

    Returns
    -------
    new_constraints : tuple
        A tuple of either constraint classes or dictionaries as required by scipy global optimization methods
        
    """

    calc_list = [o[0] for o in getmembers(constraints_mod) if (isfunction(o[1]) and o[0][0] is not "_")]

    new_constraints = []
    for const_type, kwargs in constraints.items():
        try:
            func = getattr(constraints_mod, const_type)
        except:
            raise ImportError("The constraint type, '{}', was not found\nThe following types are supported: {}".format(function,", ".join(calc_list)))

        if "args" not in kwargs:
            raise ValueError("Constraint function, {}, is missing arguements".format(const_type))

        if constraint_type == "class":
            if "type" not in kwargs or kwargs["type"] in ['linear', 'nonlinear']:
                raise ValueError("Constraint, {}, does not have type. Type can be 'linear' or 'nonlinear'.".format(const_type))
# NoteHere
            if kwargs["type"] is "linear":
                if "kwargs" not in kwargs:
                    output = LinearConstraint(func, args[0], args[1])
                else:
                    output = LinearConstraint(func, args[0], args[1], **kwargs)
            elif kwargs["type"] is "nonlinear":
                if "kwargs" not in kwargs:
                    output = NonlinearConstraint(func, args[0], args[1])
                else:
                    output = NonlinearConstraint(func, args[0], args[1], **kwargs)

        elif constraint_type == "dict":
            if "type" not in kwargs or kwargs["type"] in ['eq', 'ineq']:
                raise ValueError("Constraint, {}, does not have type. Type can be 'eq' or 'ineq'.".format(const_type))
            output = {"type": kwargs["type"], "function": func, "args": kwargs["args"]}
        else:
            raise ValueError("Constraint type {}, must be either 'class' or 'dict'.")

        new_constraints.append(output)

    return tuple(new_constraints)

def compute_obj(beadparams, fit_bead, fit_params, exp_dict, bounds):
    r"""
    Fit defined parameters for equation of state object with given experimental data. 

    Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    beadparams0 : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of beadconfig
    fit_params : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    bounds : list[tuple]
        List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.

    exp_dict : dict
        Dictionary of experimental data objects.

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    # Update beadlibrary with test paramters

    if len(beadparams) != len(fit_params):
        raise ValueError("The length of initial guess vector should be the same number of parameters to be fit.")    

    logger.info((' {}: {},' * len(fit_params)).format(*[val for pair in zip(fit_params, beadparams) for val in pair]))

    # Compute obj_function
    obj_function = []
    for key,data_obj in exp_dict.items():
        try:
            for i, param in enumerate(fit_params):
                data_obj.eos.update_parameters(fit_bead, param, beadparams[i])
            data_obj.eos.parameter_refresh()
            obj_function.append(data_obj.objective())
        except:
            #raise ValueError("Failed to evaluate objective function for {} of type {}.".format(key,data_obj.name))
            logger.error("Failed to evaluate objective function for {} of type {}.".format(key,data_obj.name))
            obj_function.append(np.inf)

    obj_total = np.nansum(obj_function)
    if obj_total == 0. and np.isnan(np.sum(obj_function)):
        obj_total = np.inf

    # Add penalty for being out of bounds for the sake of inner minimization
    for i, param in enumerate(beadparams):
        if param <= bounds[i][0]:
            obj_total += (1e+3*(param- bounds[i][0]))**8
        elif param >= bounds[i][1]:
            obj_total += (1e+3*(param- bounds[i][1]))**8

    # Write out parameters and objective functions for each dataset
    logger.info("\nParameters: {}\nValues: {}\nExp. Data: {}\nObj. Values: {}\nTotal Obj. Value: {}".format(fit_params,beadparams,list(exp_dict.keys()),obj_function,obj_total))

    return obj_total

def obj_function_form(data_test, data0, weights=1.0, method="average-squared-deviation"):
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

    """

    if np.size(data0) != np.size(data_test):
        raise ValueError("Input data of length, {}, must be the same length as reference data of length {}".format(len(data_test),len(data0)))
    
    if np.size(weights) > 1 and np.size(weights) != np.size(data_test):
        raise ValueError("Weight for data is provided as an array of length, {}, but must be length, {}.".format(len(weights),len(data_test)))

    if method == "average-squared-deviation":
        data_tmp = np.array([(data_test[i]-data0[i])/data0[i] for i in range(len(data_test)) if not np.isnan(data_test[i])])
        obj_value = np.mean(data_tmp**2)
    elif method == "sum-squared-deviation":
        data_tmp = np.array([(data_test[i]-data0[i])/data0[i] for i in range(len(data_test)) if not np.isnan(data_test[i])])
        obj_value = np.sum(data_tmp**2)
    elif method == "sum-squared-deviation-boltz":
        data_tmp = np.array([(data_test[i]-data0[i])/data0[i] for i in range(len(data_test)) if not np.isnan(data_test[i])])
        data_min = np.min(data_tmp)
        obj_value = np.sum(data_tmp**2*np.exp((data_min-data_tmp)/np.abs(data_min)))
    elif method == "sum-deviation-boltz":
        data_tmp = np.array([(data_test[i]-data0[i])/data0[i] for i in range(len(data_test)) if not np.isnan(data_test[i])])
        data_min = np.min(data_tmp)
        obj_value = np.sum(data_tmp*np.exp((data_min-data_tmp)/np.abs(data_min)))
    elif method == "percent-absolute-average-deviation":
        data_tmp = np.array([(data_test[i]-data0[i])/data0[i] for i in range(len(data_test)) if not np.isnan(data_test[i])])
        obj_value = np.mean(np.abs(data_tmp))*100

    if len(data_test) != len(data_tmp):
        logger.debug("Values of NaN were removed from objective value calculation")

    return obj_value
