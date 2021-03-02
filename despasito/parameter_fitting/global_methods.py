"""

The function names of this module represents global optimization methods that can be specified.

"""

import os
import numpy as np
import logging
import scipy.optimize as spo
from inspect import getmembers

from despasito.utils.parallelization import MultiprocessingJob
from . import fit_functions as ff
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


def single_objective(
    parameters_guess, bounds, fit_bead, fit_parameter_names, exp_dict, global_opts={}
):
    r"""
    Evaluate parameter set for equation of state with given experimental data

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}
        This dictionary is included for continuity with other global optimization methods, although this method doesn't have options.

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    if len(global_opts) > 0:
        logger.info(
            "The fitting method 'single_objective' does not have further options"
        )

    obj_value = ff.compute_obj(
        parameters_guess, fit_bead, fit_parameter_names, exp_dict, bounds
    )

    result = spo.OptimizeResult(
        x=parameters_guess,
        fun=obj_value,
        success=True,
        nit=0,
        message="Successfully computed objective function for provided parameter set.",
    )

    return result


def differential_evolution(
    parameters_guess,
    bounds,
    fit_bead,
    fit_parameter_names,
    exp_dict,
    global_opts={},
    constraints=None,
):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.differential_evolution with given experimental data. 

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
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

        - init (str) - type of initiation for population, Optional, default="random" 
        - write_intermediate_file (str) - If True, an intermediate file will be written from the method callback, default=False
        - filename (str) - filename for callback output, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - obj_cut (float) - Cut-off objective value to write the parameters, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    obj_kwargs = ["obj_cut", "filename", "write_intermediate_file"]
    if "obj_cut" in global_opts:
        obj_cut = global_opts["obj_cut"]
        del global_opts["obj_cut"]
        global_opts["write_intermediate_file"] = True
    else:
        obj_cut = None

    if "filename" in global_opts:
        filename = global_opts["filename"]
        del global_opts["filename"]
        global_opts["write_intermediate_file"] = True
    else:
        filename = None

    if (
        "write_intermediate_file" in global_opts
        and global_opts["write_intermediate_file"]
    ):
        del global_opts["write_intermediate_file"]
        global_opts["callback"] = _WriteParameterResults(
            fit_parameter_names, obj_cut=obj_cut, filename=filename
        )

    # Options for differential evolution, set defaults in new_global_opts
    new_global_opts = {"init": "random"}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                flag_workers = "workers" in global_opts and global_opts["workers"] > 1
                if value.ncores > 1 and flag_workers:
                    logger.info(
                        "Differential Evolution algorithm is using {} workers.".format(
                            value.ncores
                        )
                    )
                    new_global_opts["workers"] = value._pool.map
                    exp_dict = _del_Data_MultiprocessingObject(exp_dict)
            elif key not in obj_kwargs:
                new_global_opts[key] = value
    global_opts = new_global_opts

    if constraints is not None:
        global_opts["constraints"] = ff.initialize_constraints(constraints, "class")
    logger.info("Differential Evolution Options: {}".format(global_opts))

    result = spo.differential_evolution(
        ff.compute_obj,
        bounds,
        args=(fit_bead, fit_parameter_names, exp_dict, bounds),
        **global_opts
    )

    return result


def shgo(
    parameters_guess,
    bounds,
    fit_bead,
    fit_parameter_names,
    exp_dict,
    global_opts={},
    minimizer_opts={},
    constraints=None,
):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.shgo with given experimental data. 

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - init (str) - type of initiation for population, Optional, default="random" 
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize, Optional, default=nelder-mead
        - options (dict) - This dictionary contains the kwargs available to the chosen method, Optional, default={'maxiter': 50}

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into a tuple of dictionaries that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    # Options for differential evolution, set defaults in new_global_opts
    new_global_opts = {"sampling_method": "sobol"}
    if global_opts:
        for key, value in global_opts.items():
            if key != "MultiprocessingObject" and key not in obj_kwargs:
                new_global_opts[key] = value
    global_opts = new_global_opts

    # Set up options for minimizer in basin hopping
    new_minimizer_opts = {"method": "nelder-mead", "options": {"maxiter": 50}}
    if minimizer_opts:
        for key, value in minimizer_opts.items():
            if key == "method":
                new_minimizer_opts[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_opts[key][key2] = value2
    minimizer_opts = new_minimizer_opts

    if constraints is not None:
        global_opts["constraints"] = ff.initialize_constraints(constraints, "dict")
        if minimizer_opts["method"] not in ["COBYLA", "SLSQP"]:
            minimizer_opts["method"] = "SLSQP"
            for key, value in minimizer_opts["options"].items():
                if key not in [
                    "ftol",
                    "eps",
                    "disp",
                    "maxiter",
                    "finite_diff_rel_step",
                ]:
                    del minimizer_opts["options"][key]

    if minimizer_opts:
        logger.warning(
            "Minimization options were given but aren't used in this method."
        )

    result = spo.shgo(
        ff.compute_obj,
        bounds,
        args=(fit_bead, fit_parameter_names, exp_dict, bounds),
        **global_opts
    )

    return result


def grid_minimization(
    parameters_guess,
    bounds,
    fit_bead,
    fit_parameter_names,
    exp_dict,
    global_opts={},
    minimizer_opts={},
    constraints=None,
):
    r"""
    Fit defined parameters for equation of state object using a custom adaptation of scipy.optimize.brute with given experimental data. 

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - Ns (int) - Number of grid points along the axes, Optional, default=5
        - finish (callable) - An optimization function, default=scipy.optimize.minimize
        - initial_guesses (list) - Replaces grid of values generated with bounds and Ns
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to our `solve_root` function, Optional, default="lm"
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    # Options for brute, set defaults in new_global_opts
    flag_use_mp_object = False
    new_global_opts = {"Ns": 5, "finish": spo.minimize}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                if value.ncores > 1:
                    logger.info(
                        "Grid minimization algorithm is using {} workers.".format(
                            value.ncores
                        )
                    )
                    new_global_opts["MultiprocessingObject"] = value
                    flag_use_mp_object = True
                    exp_dict = _del_Data_MultiprocessingObject(exp_dict)
            else:
                new_global_opts[key] = value
    global_opts = new_global_opts

    if constraints is not None:
        global_opts["constraints"] = ff.initialize_constraints(constraints, "dict")

    logger.info("Grid Minimization Options: {}".format(global_opts))

    # Set up options for minimizer
    new_minimizer_opts = {"method": "lm"}
    # new_minimizer_opts = {"method": 'L-BFGS-B'}
    if minimizer_opts:
        for key, value in minimizer_opts.items():
            if key == "method":
                new_minimizer_opts[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_opts[key][key2] = value2
    minimizer_opts = new_minimizer_opts
    logger.info("    Minimizer Options: {}".format(minimizer_opts))

    args = (fit_bead, fit_parameter_names, exp_dict, bounds)

    # Set up inputs
    if "initial_guesses" in global_opts:
        x0_array = global_opts["initial_guesses"]
    else:
        # Initialization taken from scipy.optimize.brute
        N = len(bounds)
        lrange = list(bounds)
        for k in range(N):
            if type(lrange[k]) is not type(slice(None)):
                if len(lrange[k]) < 3:
                    lrange[k] = tuple(lrange[k]) + (complex(global_opts["Ns"]),)
                lrange[k] = slice(*lrange[k])
        if N == 1:
            lrange = lrange[0]
        x0_array = np.mgrid[lrange]
        inpt_shape = x0_array.shape
        if N > 1:
            x0_array = np.reshape(x0_array, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    inputs = [(x0, args, bounds, constraints, minimizer_opts) for x0 in x0_array]

    # Start computation
    if flag_use_mp_object:
        x0, results, fval = global_opts["MultiprocessingObject"].pool_job(
            _grid_minimization_wrapper, inputs
        )
    else:
        x0, results, fval = MultiprocessingJob.serial_job(
            _grid_minimization_wrapper, inputs
        )

    # Choose final output
    result = [fval[0], results[0]]
    logger.info("For bead: {} and parameters {}".format(fit_bead, fit_parameter_names))
    for i in range(len(x0_array)):
        tmp_result = results[i]
        logger.info("x0: {}, xf: {}, obj: {}".format(x0_array[i], results[i], fval[i]))
        if result[0] > fval[i]:
            result = [fval[i], tmp_result]

    result = spo.OptimizeResult(
        x=result[1],
        fun=result[0],
        success=True,
        nit=len(x0) * global_opts["Ns"],
        message="Termination successful with {} grid points and the minimum value minimized. Note that parameters may be outside of the given bounds because of the minimizing function.".format(
            len(x0) * global_opts["Ns"]
        ),
    )

    return result


def brute(
    parameters_guess, bounds, fit_bead, fit_parameter_names, exp_dict, global_opts={}
):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.brute with given experimental data. 

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - Ns (int) - Number of grid points along the axes, Optional, default=5
        - finish (callable) - An optimization function, default=scipy.optimize.minimize
        - etc. Other keywords for scipy.optimize.brute use the function defaults

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    # Options for brute, set defaults in new_global_opts
    new_global_opts = {"Ns": 5, "finish": spo.minimize}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                flag_workers = "workers" in global_opts and global_opts["workers"] > 1
                if value.ncores > 1 and flag_workers:
                    logger.info(
                        "Brute algorithm is using {} workers.".format(value.ncores)
                    )
                    new_global_opts["workers"] = value._pool.map
                    exp_dict = _del_Data_MultiprocessingObject(exp_dict)
            else:
                new_global_opts[key] = value
    global_opts = new_global_opts
    global_opts["full_output"] = True

    logger.info("Brute Options: {}".format(global_opts))
    x0, fval, grid, Jount = spo.brute(
        ff.compute_obj,
        bounds,
        args=(fit_bead, fit_parameter_names, exp_dict, bounds),
        **global_opts
    )
    result = spo.OptimizeResult(
        x=x0,
        fun=fval,
        success=True,
        nit=len(x0) * global_opts["Ns"],
        message="Termination successful with {} grid points and the minimum value minimized. Note that parameters may be outside of the given bounds because of the minimizing function.".format(
            len(x0) * global_opts["Ns"]
        ),
    )

    return result


def basinhopping(
    parameters_guess,
    bounds,
    fit_bead,
    fit_parameter_names,
    exp_dict,
    global_opts={},
    minimizer_opts={},
):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.basinhopping with given experimental data. 

    Parameters
    ----------
    parameters_guess : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of bead_configuration
    fit_parameter_names : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - niter (int) - The number of basin-hopping iterations, Optional, default=10 
        - T (float) - The "temperature" parameter for the accept or reject criterion. For best results T should be comparable to the separation (in function value) between local minima., Optional, default=0.5
        - niter_success (int) Stop the run if the global minimum candidate remains the same for this number of iterations., Optional, default=3
        - stepsize (float) - Maximum step size for use in the random displacement., Optional, default=0.1
        - take_step (callable) - Set with custom BasinStep class
        - write_intermediate_file (str) - If True, an intermediate file will be written from the method callback, default=False
        - filename (str) - filename for callback output, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - obj_cut (float) - Cut-off objective value to write the parameters, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - etc. Other keywords for scipy.optimize.basinhopping use the function defaults

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """
    obj_kwargs = ["obj_cut", "filename", "write_intermediate_file"]
    if "obj_cut" in global_opts:
        obj_cut = global_opts["obj_cut"]
        del global_opts["obj_cut"]
        global_opts["write_intermediate_file"] = True
    else:
        obj_cut = None

    if "filename" in global_opts:
        filename = global_opts["filename"]
        del global_opts["filename"]
        global_opts["write_intermediate_file"] = True
    else:
        filename = None

    if (
        "write_intermediate_file" in global_opts
        and global_opts["write_intermediate_file"]
    ):
        del global_opts["write_intermediate_file"]
        global_opts["callback"] = _WriteParameterResults(
            fit_parameter_names, obj_cut=obj_cut, filename=filename
        )

    # Options for basin hopping
    new_global_opts = {"niter": 10, "T": 0.5, "niter_success": 3}
    if global_opts:
        for key, value in global_opts.items():
            if key != "MultiprocessingObject":
                new_global_opts[key] = value
    global_opts = new_global_opts

    # Set up options for minimizer in basin hopping
    new_minimizer_opts = {"method": "nelder-mead", "options": {"maxiter": 50}}
    if minimizer_opts:
        for key, value in minimizer_opts.items():
            if key == "method":
                new_minimizer_opts[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_opts[key][key2] = value2
    minimizer_opts = new_minimizer_opts

    try:
        if "stepsize" in global_opts:
            stepsize = global_opts["stepsize"]
        else:
            stepsize = 0.1
        stepmag = np.transpose(np.array(bounds))[1]
        global_opts["take_step"] = _BasinStep(stepmag, stepsize=stepsize)
        custombounds = _BasinBounds(bounds)
    except Exception:
        raise TypeError("Could not initialize BasinStep and/or BasinBounds")

    logger.info("Basin Hopping Options: {}".format(global_opts))
    minimizer_kwargs = {
        "args": (fit_bead, fit_parameter_names, exp_dict, bounds),
        **minimizer_opts,
    }
    if "minimizer_kwargs" in global_opts:
        minimizer_kwargs.update(global_opts["minimizer_kwargs"])
        del global_opts["minimizer_kwargs"]
    result = spo.basinhopping(
        ff.compute_obj,
        parameters_guess,
        **global_opts,
        accept_test=custombounds,
        minimizer_kwargs=minimizer_kwargs
    )

    return result


# ___________ Supporting Classes and Functions ___________________________________________
def _grid_minimization_wrapper(args):
    """ Wrapper for minimization method in grid_minimization
    """

    x0, obj_args, bounds, constraints, opts = args

    # !!!!!! constraints !!!!!!!

    if "method" in opts:
        method = opts["method"]
        del opts["method"]
    else:
        method = "least_squares"

    try:
        result = gtb.solve_root(
            ff.compute_obj,
            args=obj_args,
            method=method,
            x0=x0,
            bounds=bounds,
            options=opts,
        )
    except Exception:
        result = np.nan * np.ones(len(x0))

    if np.sum(np.abs(result - x0)) < 1e-6:
        result = np.nan * np.ones(len(x0))

    logger.info("Starting parameters: {}, converged to: {}".format(x0, result))

    fval = ff.compute_obj(result, *obj_args)

    return x0, result, fval


class _BasinStep(object):
    r""" Custom basin step used by scipy.optimize.basinhopping function.
    """

    def __init__(self, stepmag, stepsize=0.05):
        r"""
            
        Parameters
        ----------
        stepmag : list
            List of step magnitudes
        stepsize : float, Optional, default=0.05
            Step size 

        Attributes
        ----------
        stepmag : list
            List of step magnitudes
        stepsize : float, Optional, default=0.05
            Step size
            
        """

        self._stepsize = stepsize
        self._stepmag = stepmag

    def __call__(self, x):

        r"""
            
        Parameters
        ----------
        x : numpy.ndarray
            Guess in parameters values

        Returns
        -------
        basinstep : numpy.ndarray
            Suggested basin step used in scipy.optimize.basinhopping algorithm
            
        """

        # Save initial guess in array
        xold = np.copy(x)

        # Loop for number of times to start over
        for j in range(1000):
            # reset array x
            x = np.copy(xold)
            breakloop = True
            # Iterate through array of step magnitudes
            for i, mag in enumerate(self._stepmag):
                # Add or subtract a random number within distribution of +- mag*stepsize
                x[i] += np.random.uniform(-mag * self._stepsize, mag * self._stepsize)
                # If a value of x is negative, don't  break the cycle
                if x[i] < 0.0:
                    breakloop = False
            if breakloop:
                break
            logger.info("Basin Step after {} iterations:\n    {}".format(j, x))
        return x


class _BasinBounds(object):
    r""" Object used by scipy.optimize.basinhopping to set bounds of parameters.
    """

    def __init__(self, bounds):
        r"""
            
        Parameters
        ----------
        bounds : numpy.ndarray
            Bounds on parameters
    
        Attributes
        ----------
        xmin : numpy.ndarray
            Array of minimum values for each parameter
        xman : numpy.ndarray
            Array of maximum values for each parameter
            
        """
        bounds = np.transpose(np.array(bounds))
        self.xmin = bounds[0]
        self.xmax = bounds[1]

    def __call__(self, **kwargs):
        r"""
            
        Parameters
        ----------
        kwargs
            Keyword arguments used in BasinBounds object for scipy.optimize.basinhopping
            
            - x_new (numpy.ndarray) - Guess in parameters values
            - f_new (numpy.ndarray) - Objective value for given parameters

        Returns
        -------
        value : bool
            A true or false value that says whether the guess in parameter value is within bounds
            
        """
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        feasible1 = np.abs(kwargs["f_new"]) < np.inf
        feasible2 = not np.isnan(kwargs["f_new"])

        if tmax and tmin and feasible1 and feasible2:
            logger.info(
                "Accept parameters: {}, with obj. function: {}".format(
                    x, kwargs["f_new"]
                )
            )
        else:
            logger.info(
                "Reject parameters: {}, with obj. function: {}".format(
                    x, kwargs["f_new"]
                )
            )

        return tmax and tmin and feasible1 and feasible2


class _WriteParameterResults(object):
    r""" Object used by scipy.optimize.basinhopping to set bounds of parameters.
    """

    def __init__(self, beadnames, obj_cut=None, filename=None):
        r"""
            
        Attributes
        ----------
        beadnames : list[str]
            List of bead names for filename header
        obj_cut : float, Optional, default=np.inf
            Cut-off objective value to write the parameters
        filename : str, Optional, default=parameters.txt
            File to which parameters are written

        Returns
        -------
        Initiate file with parameters
            
        """

        if obj_cut == None:
            self.obj_cut = np.inf
        else:
            self.obj_cut = obj_cut

        if filename == None:
            filename = "parameters.txt"

        if os.path.isfile(filename):
            old_fname = filename
            for i in range(20):
                filename = "{}_{}".format(i, old_fname)
                if not os.path.isfile(filename):
                    logger.info(
                        "File '{}' already exists, using {}.".format(
                            old_fname, filename
                        )
                    )
                    break

        self.beadnames = beadnames
        self.filename = filename
        self.ninit = 0

    def __call__(self, *args, **kwargs):
        r"""
        The provided args and kwargs change depending on the global optimization method. This class is equipped to distinguish the callback function for differential_evolution (length equal to ) and basinhopping.
            
        Parameters
        ----------
        args
            The provided args change depending on the global optimization method.
        
            - x_new (numpy.ndarray) - Current parameter values being evaluated, used in both algorithms
            - f_new (float) - Current object function value for x_new, used in basinhopping
            - accept (bool) - Whether or not that minimum was accepted, used in basinhopping

        kwargs
            The provided kwargs change depending on the global optimization method.

            - convergence (float) - Used in differential evolution, the fractional value of the population convergence. When greater than one the function halts.
            
        Returns
        -------
        value : bool
            A true or false value that says whether the guess in parameter value is within bounds
            
        """

        if "convergence" in kwargs:  # Used in differential_evolution
            if kwargs["convergence"] < self.obj_cut:
                if not os.path.isfile(self.filename):
                    with open(self.filename, "w") as f:
                        f.write(
                            "# n, convergence, {}\n".format(", ".join(self.beadnames))
                        )

                with open(self.filename, "a") as f:
                    tmp = [self.ninit, kwargs["convergence"]] + list(args[0])
                    f.write(("{}, " * len(tmp)).format(*tmp) + "\n")

        elif len(args) == 3:  # used in basinhopping
            if args[2] or args[1] < self.obj_cut:
                if not os.path.isfile(self.filename):
                    with open(self.filename, "w") as f:
                        f.write(
                            "# n, obj. value, accepted, {}\n".format(
                                ", ".join(self.beadnames)
                            )
                        )

                with open(self.filename, "a") as f:
                    tmp = [self.ninit, args[1], args[2]] + list(args[0])
                    f.write(("{}, " * len(tmp)).format(*tmp) + "\n")
        else:
            raise ValueError(
                "Unknown inputs. This function is equipped to handle differential_evolution and basinhopping algorithms."
            )

        self.ninit += 1

        return False


def _del_Data_MultiprocessingObject(dictionary):
    r""" A dictionary of fitting objects will remove MultiprocessingObject attributes so that the multiprocessing pool can be used by the fitting algorithm.

    Parameters
    ----------
    dictionary : dict
        Dictionary of fitting objects

    Returns
    -------
    new_dictionary : dict
        Updated fitting objects
    """

    new_dictionary = dictionary.copy()
    for key in new_dictionary:
        if "MultiprocessingObject" in new_dictionary[key].thermodict:
            del new_dictionary[key].thermodict["MultiprocessingObject"]

    return new_dictionary
