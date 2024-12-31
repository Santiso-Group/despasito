""" General functions that can be used by multiple modules
"""

import numpy as np
import scipy.optimize as spo
import logging

logger = logging.getLogger(__name__)


def solve_root(func, args=(), method="bisect", x0=None, bounds=None, options={}):
    """
    This function will setup and dispatch thermodynamic jobs.

    Parameters
    ----------
    func : function
        Function used in job. Can be any of the following scipy methods: "brent",
        "least_squares", "TNC", "L-BFGS-B", "SLSQP", 'hybr', 'lm', 'linearmixing',
        'diagbroyden', 'excitingmixing', 'krylov', 'df-sane', 'anderson',
        'hybr_broyden1', 'hybr_broyden2', 'broyden1', 'broyden2', 'bisect'.
    args : tuple, Optional, default=()
        Each entry of this list contains the input arguments for each job
    method : str, Optional, default="bisect"
        Choose the method used to solve the dew point calculation
    x0 : float, Optional, default=None
        Initial guess in parameter to be optimized
    bounds : tuple, Optional, default=None
        Parameter boundaries
    options : dict, Optional, default={}
        These options are used in the scipy method

    Returns
    -------
    output : tuple
        This structure contains the outputs of the jobs given

    """

    if method not in [
        "brentq",
        "least_squares",
        "TNC",
        "L-BFGS-B",
        "SLSQP",
        "hybr",
        "lm",
        "linearmixing",
        "diagbroyden",
        "excitingmixing",
        "krylov",
        "df-sane",
        "anderson",
        "hybr_broyden1",
        "hybr_broyden2",
        "broyden1",
        "broyden2",
        "bisect",
    ]:
        raise ValueError("Optimization method, {}, not supported.".format(method))

    if x0 is None:
        logger.debug("Initial guess in optimization not provided")
    if np.any(bounds is None):
        logger.debug("Optimization bounds not provided")

    if x0 is None and method in [
        "broyden1",
        "broyden2",
        "anderson",
        "hybr",
        "lm",
        "linearmixing",
        "diagbroyden",
        "excitingmixing",
        "krylov",
        "df-sane",
    ]:
        if np.any(bounds is None):
            raise ValueError(
                "Optimization method, {}, requires x0. Because bounds were"
                " not provided, so problem cannot be solved.".format(method)
            )
        else:
            logger.error("Optimization method, {}, requires x0, using bisect instead".format(method))
            method = "bisect"

    if np.size(x0) > 1 and method in ["brentq", "bisect"]:
        logger.error("Optimization method, {}, is for scalar functions, using {}".format(method, "least_squares"))
        method = "least_squares"

    if not isiterable(bounds[0]):
        bounds = [bounds]

    if np.any([np.any(x is None) for x in bounds]) and method in ["brentq", "bisect"]:
        if x0 is None:
            raise ValueError(
                "Optimization method, {}, requires bounds. ".format(method)
                + "Because x0 was not provided, so problem cannot be solved."
            )
        else:
            logger.error("Optimization method, {}, requires bounds, using hybr".format(method))
            method = "hybr"

    if np.any(bounds is not None):
        for i, bnd in enumerate(bounds):
            if len(bnd) != 2:
                raise ValueError("bounds are not of length two")
            else:
                bounds[i] = tuple(bnd)

    # ################### Root Finding without Boundaries ###################
    if method in ["broyden1", "broyden2"]:
        outer_dict = {
            "fatol": 1e-5,
            "maxiter": 25,
            "jac_options": {"reduction_method": "simple"},
        }
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)
    elif method == "anderson":
        outer_dict = {"fatol": 1e-5, "maxiter": 25}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)
    elif method in [
        "hybr",
        "lm",
        "linearmixing",
        "diagbroyden",
        "excitingmixing",
        "krylov",
        "df-sane",
    ]:
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)

    # ################### Minimization Methods with Boundaries ###################
    elif method in ["TNC", "L-BFGS-B"]:
        outer_dict = {
            "gtol": 1e-2 * np.sqrt(np.finfo("float").eps),
            "ftol": np.sqrt(np.finfo("float").eps),
        }
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        if len(bounds) == 2:
            sol = spo.minimize(
                func,
                x0,
                args=args,
                method=method,
                bounds=tuple(bounds),
                options=outer_dict,
            )
        else:
            sol = spo.minimize(func, x0, args=args, method=method, options=outer_dict)
    elif method == "SLSQP":
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        if len(bounds) == 2:
            sol = spo.minimize(
                func,
                x0,
                args=args,
                method=method,
                bounds=tuple(bounds),
                options=outer_dict,
            )
        else:
            sol = spo.minimize(func, x0, args=args, method=method, options=outer_dict)

    # ################### Root Finding with Boundaries ###################
    elif method == "brentq":
        outer_dict = {"rtol": 1e-7}
        for key, value in options.items():
            if key in ["xtol", "rtol", "maxiter", "full_output", "disp"]:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        sol = spo.brentq(func, bounds[0][0], bounds[0][1], args=args, **outer_dict)
    elif method == "least_squares":
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        bnd_tmp = [[], []]
        for bnd in bounds:
            bnd_tmp[0].append(bnd[0])
            bnd_tmp[1].append(bnd[1])
        sol = spo.least_squares(func, x0, bounds=tuple(bnd_tmp), args=args, **outer_dict)
    elif method == "bisect":
        outer_dict = {"maxiter": 100, "rtol": 1e-12}
        for key, value in options.items():
            if key in ["xtol", "rtol", "maxiter", "full_output", "disp"]:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method, outer_dict))
        sol = spo.bisect(func, bounds[0][0], bounds[0][1], args=args, **outer_dict)

    # Given final P estimate
    if method not in ["brentq", "bisect"]:
        solution = sol.x
        logger.info("Optimization terminated successfully: {} {}".format(sol.success, sol.message))
    else:
        logger.info("Optimization terminated successfully: {}".format(sol))
        solution = sol

    return solution


def central_difference(x, func, step_size=1e-5, relative=False, args=()):
    """
    Take the derivative of a dependent variable calculated with a given function using
    the central difference method.

    Parameters
    ----------
    x : numpy.ndarray
        Independent variable to take derivative with respect too, using the central
        difference method. Must be first input of the function.
    func : function
        Function used in job to calculate dependent factor. This function should have
        a single output.
    step_size : float, Optional, default=1E-5
        Either the step size used in the central difference method, or if
        ``relative=True``, this variable is a scaling factor so that the step size for
        each value of x is x * step_size.
    args : tuple, Optional, default=()
        Each entry of this list contains the input arguments for each job
    relative : bool, Optional, default=False
        If False, the step_size is directly used to calculate the derivative. If true,
        step_size becomes a scaling factor, where the step size for each value of x
        becomes step_size*x.

    Returns
    -------
    dydx : numpy.ndarray
        Array of derivative of y with respect to x, given an array of independent
        variables.
    """

    if not isiterable(x):
        x - np.array([x])
    elif not isinstance(x, np.ndarray):
        x = np.array(x)

    if relative:
        step = x * step_size
        if not isiterable(step):
            step = np.array([step])
        step = np.array([2 * np.finfo(float).eps if xx < np.finfo(float).eps else xx for xx in step])
    else:
        step = step_size

    y = func(np.append(x + step, x - step), *args)
    lx = int(len(y) / 2)
    dydx = (y[:lx] - y[lx:]) / (2.0 * step)

    return dydx


def isiterable(array):
    """
    Check if variable is an iterable type with a length (e.g. np.array or list).

    Note that this could be tested with ``isinstance(array, Iterable)``, however
    ``array=np.array(1.0)`` would pass that test and then fail in ``len(array)``.

    Parameters
    ----------
    array
        Variable of some type, that should be iterable

    Returns
    -------
    isiterable : bool
        Will be True if indexing is possible and False if not.
    """

    array_tmp = np.array(array, dtype=object)
    tmp = np.shape(array_tmp)
    if tmp:
        isiterable = True
    else:
        isiterable = False

    return isiterable


def check_length_dict(dictionary, keys, lx=None):
    """
    This function compared the entries, keys, in the provided dictionary to ensure
    they're the same length.

    All entries in the list ``keys``, will be made into numpy arrays (if present).
    If a float or array of length one is provided, it will be expanded to the length
    of other arrays.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing all or some of the keywords, ``keys``, of what should
        be arrays of identical size.
    keys : list
        Possible keywords representing array entries
    lx : int, Optional, default=None
        The size that arrays should conform to

    Returns
    -------
    new_dictionary : dict
        Dictionary of arrays of identical size.

    """

    if lx is None:
        lx_array = []
        for key in keys:
            if key in dictionary:
                tmp = dictionary[key]
                if np.shape(tmp):
                    lx_array.append(len(tmp))
                else:
                    lx_array.append(1)
        if not len(lx_array):
            raise ValueError("None of the provided keys are found in the given dictionary")
        lx = max(lx_array)

    new_dictionary = {}
    for key in keys:
        if key in dictionary:
            tmp = dictionary[key]
            if isiterable(tmp):
                l_tmp = len(tmp)
                if l_tmp == 1:
                    new_dictionary[key] = np.array([tmp[0] for x in range(lx)], float)
                elif l_tmp == lx:
                    new_dictionary[key] = np.array(tmp, float)
                else:
                    raise ValueError("Entry, {}, should be length {}, not {}".format(key, lx, l_tmp))
            else:
                new_dictionary[key] = np.array([tmp for x in range(lx)], float)

    return new_dictionary


def set_defaults(dictionary, keys, values, lx=None):
    """
    This function checks a dictionary for the given keys, and if it's not there, the
    appropriate value is added to the dictionary.

    Parameters
    ----------
    dictionary : dict
        Dictionary of data
    keys : list
        Keys that should be present (of the same length as ``lx``)
    values : list
        Default values for the keys that aren't in dictionary
    lx : int, Optional, default=None
        If not None, and values[i] is a float, the key will be set to an array of
        length, ``lx``, populated by ``values[i]``

    Returns
    -------
    new_dictionary : dict
        Dictionary of arrays of identical size.

    """

    new_dictionary = dictionary.copy()

    key_iterable = isiterable(keys)
    if not isiterable(values):
        if key_iterable:
            values = np.ones(len(keys)) * values
        else:
            values = np.array([values])
            keys = np.array([keys])
    else:
        if key_iterable and len(keys) != len(values):
            raise ValueError("Length of given keys and values must be equivalent.")
        elif not key_iterable:
            if len(values) != 1:
                raise ValueError("Multiple default values for given key, {}, is ambiguous".format(keys))
            else:
                keys = [keys]

    for i, key in enumerate(keys):
        if key not in dictionary:
            tmp = values[i]
            if not isiterable(tmp) and lx is not None:
                new_dictionary[key] = np.ones(lx) * tmp
            else:
                new_dictionary[key] = tmp
            logger.info("Entry, {}, set to default: {}".format(key, tmp))

    return new_dictionary
