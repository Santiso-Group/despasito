
import numpy as np
import scipy.optimize as spo
import logging

logger = logging.getLogger(__name__)

def solve_root( func, args=None, method="bisect", x0=None, bounds=None, options={}):
    """
    This function will setup and dispatch thermodynamic jobs.

    Parameters
    ----------
    func : function
        Function used in job
    args : list, Optional, default: None
        Each entry of this list contains the input arguements for each job
    method : str, Optional, default: "bisect"
        Choose the method used to solve the dew point calculation
    x0 : float, Optional, default: None
        Initial guess in parameter to be optimized
    bounds : tuple, Optional, default: None
         Parameter boundaries
    options : dict, Optional, default: {}
        These options are used in the scipy method

    Returns
    -------
    output : tuple
        This structure contains the outputs of the jobs given

    """

    if method not in ["brent", "least_squares", "TNC", "L-BFGS-B", "SLSQP", 'hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane', 'anderson', 'hybr_broyden1', 'hybr_broyden2', 'broyden1', 'broyden2', 'bisect']:
        logger.error("Optimization method, {}, not supported.".format(method))

    if (x0 is None and method in ['broyden1', 'broyden2','anderson','hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']):
        if np.any(bounds is None):
            raise ValueError("Optimization method, {}, requires x0. Because bounds were not provided, so problem cannot be solved.".format(method))
        else:
            logger.error("Optimization method, {}, requires x0, using bisect".format(method))
            method = "bisect"

    if (np.any(bounds is None) and method in ["brent",'bisect']):
        if x0 is None:
            raise ValueError("Optimization method, {}, requires bounds. Because x0 was not provided, so problem cannot be solved.".format(method))
        else:
            logger.error("Optimization method, {}, requires bounds, using hybr".format(method))
            method = "hybr"

    if np.any(bounds is not None) and len(bounds) != 2:
        raise ValueError("bounds are not of length two")

    #################### Root Finding without Boundaries ###################
    if method in ['broyden1', 'broyden2']:
        outer_dict = {'fatol': 1e-5, 'maxiter': 25, 'jac_options': {'reduction_method': 'simple'}}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)
    elif method == 'anderson':
        outer_dict = {'fatol': 1e-5, 'maxiter': 25}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)
    elif method in ['hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']:
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.root(func, x0, args=args, method=method, options=outer_dict)

    #################### Minimization Methods with Boundaries ###################
    elif method in ["TNC", "L-BFGS-B", "SLSQP"]:
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        if len(bounds) == 2:
            sol = spo.minimize(func, x0, args=args, method=method, bounds=[tuple(bounds)], options=outer_dict)
        else:
            sol = spo.minimize(func, x0, args=args, method=method, options=outer_dict)

    #################### Root Finding with Boundaries ###################
    elif method == "brent":
        outer_dict = {"rtol":1e-7}
        for key, value in options.items():
            if key in ["xtol","rtol","maxiter","full_output","disp"]:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.brentq(func, bounds[0], bounds[1], args=args, **outer_dict)
    elif method == "least_squares":
        outer_dict = {}
        for key, value in options.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.least_squares(func, x0, bounds=(bounds[0],bounds[1]), args=args, **outer_dict)
    elif method == 'bisect':
        outer_dict = {'maxiter': 50}
        for key, value in options.items():
            if key in ['xtol', 'rtol', 'maxiter', 'full_output', 'disp']:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(method,outer_dict))
        sol = spo.bisect(func, bounds[0], bounds[1], args=args, **outer_dict)

    #Given final P estimate
    if method not in ["brent", "bisect"]:
        solution = sol.x
        logger.info("Optimization terminated successfully: {} {}".format(sol.success,sol.message))
    else:
        logger.info("Optimization terminated successfully: {}".format(sol))
        solution = sol

    return solution
