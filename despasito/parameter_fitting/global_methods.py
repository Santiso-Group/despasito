"""

The function names of this module represents global optimization methods that can be
specified as ``global_opts["method"]`` in :func:`~despasito.parameter_fitting.fit`.

"""

import os
import numpy as np
import logging
import scipy.optimize as spo

from despasito.utils.parallelization import MultiprocessingJob
from . import fit_functions as ff
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


def single_objective(parameters_guess, bounds, fit_bead, fit_parameter_names, exp_dict, global_opts={}):
    r"""
    Evaluate parameter set for equation of state with given experimental data

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum
        and maximum bounds of parameter being fit. Defaults from Eos object are broad,
        so we recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See
        EOS documentation for supported parameter names. Cross interaction parameter
        names should be composed of parameter name and the other bead type, separated
        by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}
        This dictionary is included for continuity with other global optimization
        methods, although this method doesn't have options.

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    if len(global_opts) > 0:
        logger.info("The fitting method 'single_objective' does not have further options")

    obj_value = ff.compute_obj(parameters_guess, fit_bead, fit_parameter_names, exp_dict, bounds)

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
    Fit defined parameters for equation of state object using
    scipy.optimize.differential_evolution with given experimental data.

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters. Not used in this method.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum
        and maximum bounds of parameter being fit. Defaults from Eos object are broad,
        so we recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS
        documentation for supported parameter names. Cross interaction parameter names
        should be composed of parameter name and the other bead type, separated by an
        underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional

        - init (str) - Optional, default="random", type of initiation for population
        - write_intermediate_file (str) - Optional, default=False, If True, an
        intermediate file will be written from the method callback
        - filename (str) - Optional, default=None, filename for callback output, if
        provided, ``write_intermediate_file`` will be set to True
        - obj_cut (float) - Optional, default=None, Cut-off objective value to write
        the parameters, if provided, ``write_intermediate_file`` will be set to True
        - etc. Other keywords for scipy.optimize.differential_evolution use the
        function defaults

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into
        a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    global_opts = global_opts.copy()

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

    if "write_intermediate_file" in global_opts and global_opts["write_intermediate_file"]:
        del global_opts["write_intermediate_file"]
        global_opts["callback"] = _WriteParameterResults(fit_parameter_names, obj_cut=obj_cut, filename=filename)

    # Options for differential evolution, set defaults in new_global_opts
    new_global_opts = {"init": "random"}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                flag_workers = "workers" in global_opts and global_opts["workers"] > 1
                if value.ncores > 1 and flag_workers:
                    logger.info("Differential Evolution algorithm is using {} workers.".format(value.ncores))
                    new_global_opts["workers"] = value._pool.map
                    exp_dict = _del_Data_MultiprocessingObject(exp_dict)
            elif key not in obj_kwargs:
                new_global_opts[key] = value
    global_opts = new_global_opts

    if constraints is not None:
        global_opts["constraints"] = ff.initialize_constraints(constraints, "class")
    logger.info("Differential Evolution Options: {}".format(global_opts))

    result = spo.differential_evolution(
        ff.compute_obj, bounds, args=(fit_bead, fit_parameter_names, exp_dict, bounds), **global_opts
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
    Fit defined parameters for equation of state object using scipy.optimize.shgo with
    given experimental data.

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum
        and maximum bounds of parameter being fit. Defaults from Eos object are broad,
        so we recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS
        documentation for supported parameter names. Cross interaction parameter names
        should be composed of parameter name and the other bead type, separated by an
        underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - init (str) - Optional, default="random", type of initiation for population
        - etc. Other keywords for scipy.optimize.shgo use the function defaults

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Optional, default=nelder-mead, Method available to
        scipy.optimize.minimize
        - options (dict) - Optional, default={'maxiter': 50}, This dictionary contains
        the kwargs available to the chosen method

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into
        a tuple of dictionaries that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    global_opts = global_opts.copy()

    # Options for differential evolution, set defaults in new_global_opts
    obj_kwargs = ["obj_cut", "filename", "write_intermediate_file"]
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

    result = spo.shgo(
        ff.compute_obj,
        bounds,
        args=(fit_bead, fit_parameter_names, exp_dict, bounds),
        minimizer_kwargs=minimizer_opts,
        **global_opts,
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
    Fit defined parameters for equation of state object using a custom adaptation of
    scipy.optimize.brute with given experimental data.

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum
        and maximum bounds of parameter being fit. Defaults from Eos object are broad,
        so we recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS
        documentation for supported parameter names. Cross interaction parameter names
        should be composed of parameter name and the other bead type, separated by an
        underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - Ns (int) - Optional, default=5, Number of grid points along the axes
        - finish (callable) - Optional, default=scipy.optimize.minimize, A minimization
        function
        - initial_guesses (list) - Optional, Replaces grid of values generated with
        bounds and Ns
        - split_grid_minimization (int) - Optional, default=0, Choose index of first
        parameter to fit, while the grid is formed from those before. For example, if
        4 parameters are defined and ``split_grid_minimization==2``, then a grid is
        formed for the first two parameters ``parameters_guess[:2]``, and the remaining
        parameters, ``parameters_guess[2:]`` are minimized.

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Optional, default="least_squares", Method available to our
        :func:`~despasito.utils.general_toolbox.solve_root` function
        - options (dict) - Optional, default={}, This dictionary contains the kwargs
        available to the chosen method

    constraints : dict, Optional, default=None
        This dictionary of constraint types and their arguments will be converted into
        a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    # Options for brute, set defaults in new_global_opts
    flag_use_mp_object = False
    new_global_opts = {"Ns": 5, "finish": spo.minimize, "split_grid_minimization": 0}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                if value.ncores > 1:
                    logger.info("Grid minimization algorithm is using {} workers.".format(value.ncores))
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
    new_minimizer_opts = {"method": "least_squares"}
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
        del global_opts["Ns"]
        x0_array = global_opts["initial_guesses"]

        if global_opts["split_grid_minimization"] != 0:
            inputs = []
            bounds = bounds[global_opts["split_grid_minimization"]:]
            for x0 in x0_array:
                tmp1 = x0[global_opts["split_grid_minimization"]:]
                tmp2 = x0[: global_opts["split_grid_minimization"]]
                inputs.append((tmp1, (*args, tmp2), bounds, constraints, minimizer_opts))
        else:
            inputs = [(x0, args, bounds, constraints, minimizer_opts) for x0 in x0_array]

    else:
        # Initialization based on implementation in scipy.optimize.brute
        if global_opts["split_grid_minimization"] == 0:
            N = len(bounds)
            lrange = list(bounds)
            for k in range(N):
                if lrange[k] is not None:
                    if len(lrange[k]) < 3:
                        lrange[k] = tuple(lrange[k]) + (complex(global_opts["Ns"]),)
                    lrange[k] = slice(*lrange[k])
        else:
            if not isinstance(global_opts["split_grid_minimization"], int):
                raise ValueError("Option, split_grid_minimization, must be an integer")
            N = len(bounds[: global_opts["split_grid_minimization"]])
            lrange = list(bounds[: global_opts["split_grid_minimization"]])
            for k in range(N):
                if lrange[k] is not None:
                    if len(lrange[k]) < 3:
                        lrange[k] = tuple(lrange[k]) + (complex(global_opts["Ns"]),)
                    lrange[k] = slice(*lrange[k])
            bounds = bounds[global_opts["split_grid_minimization"]:]

        if N == 1:
            lrange = lrange[0]

        x0_array = np.mgrid[lrange]
        inpt_shape = x0_array.shape
        if N > 1:
            x0_array = np.reshape(x0_array, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

        if global_opts["split_grid_minimization"] != 0:
            min_parameters = list(parameters_guess[global_opts["split_grid_minimization"]:])
            inputs = [(min_parameters, (*args, x0), bounds, constraints, minimizer_opts) for x0 in x0_array]
        else:
            inputs = [(x0, args, bounds, constraints, minimizer_opts) for x0 in x0_array]

    lx = len(x0_array)
    # Start computation
    if flag_use_mp_object:
        x0, results, fval = global_opts["MultiprocessingObject"].pool_job(_grid_minimization_wrapper, inputs)
    else:
        x0, results, fval = MultiprocessingJob.serial_job(_grid_minimization_wrapper, inputs)

    # Choose final output
    if global_opts["split_grid_minimization"] != 0:
        if "initial_guesses" not in global_opts:
            x0_new = np.zeros((lx, len(parameters_guess)))
        results_new = np.zeros((lx, len(parameters_guess)))
        for i in range(len(x0_array)):
            if "initial_guesses" not in global_opts:
                x0_new[i] = np.array(list(x0_array[i]) + list(min_parameters))
                results_new[i] = np.array(list(x0_array[i]) + list(results[i]))
            else:
                results_new[i] = np.array(
                    list(x0_array[i][: global_opts["split_grid_minimization"]]) + list(results[i])
                )
        results = results_new
        if "initial_guesses" not in global_opts:
            x0_array = x0_new

    result = [fval[0], results[0]]
    logger.info("For bead: {} and parameters {}".format(fit_bead, fit_parameter_names))
    for i in range(lx):
        tmp_result = results[i]
        logger.info("x0: {}, xf: {}, obj: {}".format(x0_array[i], tmp_result, fval[i]))
        if result[0] > fval[i]:
            result = [fval[i], tmp_result]

    result = spo.OptimizeResult(
        x=result[1],
        fun=result[0],
        success=True,
        nit=lx,
        message=(
            "Termination successful with {} grid points and".format(lx)
            + " the minimum value minimized. Note that parameters may be outside of"
            " the given bounds because of the minimizing function."
        ),
    )

    return result


def brute(parameters_guess, bounds, fit_bead, fit_parameter_names, exp_dict, global_opts={}):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.brute with
    given experimental data.

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and
        maximum bounds of parameter being fit. Defaults from Eos object are broad, so we
        recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS
        documentation for supported parameter names. Cross interaction parameter names
        should be composed of parameter name and the other bead type, separated by an
        underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - Ns (int) - Optional, default=5, Number of grid points along the axes
        - finish (callable) - Optional, default=scipy.optimize.minimize, An optimization
        function
        - etc. Other keywords for scipy.optimize.brute use the function defaults

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    # Options for brute, set defaults in new_global_opts
    new_global_opts = {"Ns": 5, "finish": spo.minimize}
    if global_opts:
        for key, value in global_opts.items():
            if key == "MultiprocessingObject":
                flag_workers = "workers" in global_opts and global_opts["workers"] > 1
                if value.ncores > 1 and flag_workers:
                    logger.info("Brute algorithm is using {} workers.".format(value.ncores))
                    new_global_opts["workers"] = value._pool.map
                    exp_dict = _del_Data_MultiprocessingObject(exp_dict)
            else:
                new_global_opts[key] = value
    global_opts = new_global_opts
    global_opts["full_output"] = True

    logger.info("Brute Options: {}".format(global_opts))
    x0, fval, grid, Jount = spo.brute(
        ff.compute_obj, bounds, args=(fit_bead, fit_parameter_names, exp_dict, bounds), **global_opts
    )
    result = spo.OptimizeResult(
        x=x0,
        fun=fval,
        success=True,
        nit=len(x0) * global_opts["Ns"],
        message=(
            "Termination successful with {} grid points and the minimum value "
            "minimized. Note that parameters may be outside of the given bounds "
            "because of the minimizing function.".format(len(x0) * global_opts["Ns"])
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
    Fit defined parameters for equation of state object using
    scipy.optimize.basinhopping with given experimental data.

    Parameters
    ----------
    parameters_guess : numpy.ndarray
        An array of initial guesses for parameters.
    bounds : list[tuple]
        List of length equal to fit_parameter_names with lists of pairs for minimum and
        maximum bounds of parameter being fit. Defaults from Eos object are broad, so
        we recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit.
    fit_parameter_names : list[str]
        This list contains the name of the parameter being fit (e.g. epsilon). See EOS
        documentation for supported parameter names. Cross interaction parameter names
        should be composed of parameter name and the other bead type, separated by an
        underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.
    global_opts : dict, Optional, default={}

        - niter (int) - Optional, default=10, The number of basin-hopping iterations
        - T (float) - Optional, default=0.5, The "temperature" parameter for the accept
        or reject criterion. For best results T should be comparable to the separation
        (in function value) between local minima.
        - niter_success (int) Optional, default=3, Stop the run if the global minimum
        candidate remains the same for this number of iterations.
        - stepsize (float) - Optional, default=0.1, Maximum step size for use in the
        random displacement.
        - take_step (callable) - Set with custom BasinStep class
        - write_intermediate_file (str) - Optional, default=False, If True, an
        intermediate file will be written from the method callback
        - filename (str) - Optional, default=None, filename for callback output, if
        provided, `write_intermediate_file` will be set to True
        - obj_cut (float) - Optional, default=None, Cut-off objective value to write
        the parameters, if provided, `write_intermediate_file` will be set to True
        - etc. Other keywords for scipy.optimize.basinhopping use the function
        defaults

    minimizer_opts : dict, Optional, default={}
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen
        method

    Returns
    -------
    Objective : obj
        scipy OptimizedResult object

    """

    global_opts = global_opts.copy()

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

    if "write_intermediate_file" in global_opts and global_opts["write_intermediate_file"]:
        del global_opts["write_intermediate_file"]
        global_opts["callback"] = _WriteParameterResults(fit_parameter_names, obj_cut=obj_cut, filename=filename)

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
        ff.compute_obj, parameters_guess, **global_opts, accept_test=custombounds, minimizer_kwargs=minimizer_kwargs
    )

    return result


# ___________ Supporting Classes and Functions _________________
def _grid_minimization_wrapper(args):
    """Wrapper for minimization method in grid_minimization"""

    x0, obj_args, bounds, constraints, opts = args

    if constraints is not None:
        logger.warning("Constraints defined, but grid_minimization does not support their use.")

    opts = opts.copy()
    if "method" in opts:
        method = opts["method"]
        del opts["method"]

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
        logger.info("Minimization Failed:", exc_info=True)
        result = np.nan * np.ones(len(x0)) if gtb.isiterable(x0) else np.nan

    logger.info("Starting parameters: {}, converged to: {}".format(x0, result))

    fval = ff.compute_obj(result, *obj_args)

    return x0, result, fval


class _BasinStep(object):
    r"""Custom basin step used by scipy.optimize.basinhopping function."""

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
                x[i] += np.random.uniform(-mag * self._stepsize, mag * self._stepsize)
                # If a value of x is negative, don't  break the cycle
                if x[i] < 0.0:
                    breakloop = False
            if breakloop:
                break
            logger.info("Basin Step after {} iterations:\n    {}".format(j, x))
        return x


class _BasinBounds(object):
    r"""Object used by scipy.optimize.basinhopping to set bounds of parameters."""

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
            Keyword arguments used in BasinBounds object for
            scipy.optimize.basinhopping

            - x_new (numpy.ndarray) - Guess in parameters values
            - f_new (numpy.ndarray) - Objective value for given parameters

        Returns
        -------
        value : bool
            A true or false value that says whether the guess in parameter value is
            within bounds

        """
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        feasible1 = np.abs(kwargs["f_new"]) < np.inf
        feasible2 = not np.isnan(kwargs["f_new"])

        if tmax and tmin and feasible1 and feasible2:
            logger.info("Accept parameters: {}, with obj. function: {}".format(x, kwargs["f_new"]))
        else:
            logger.info("Reject parameters: {}, with obj. function: {}".format(x, kwargs["f_new"]))

        return tmax and tmin and feasible1 and feasible2


class _WriteParameterResults(object):
    r"""Object used by scipy.optimize.basinhopping to set bounds of parameters."""

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

        if obj_cut is None:
            self.obj_cut = np.inf
        else:
            self.obj_cut = obj_cut

        if filename is None:
            filename = "parameters.txt"

        if os.path.isfile(filename):
            old_fname = filename
            for i in range(20):
                filename = "{}_{}".format(i, old_fname)
                if not os.path.isfile(filename):
                    logger.info("File '{}' already exists, using {}.".format(old_fname, filename))
                    break

        self.beadnames = beadnames
        self.filename = filename
        self.ninit = 0

    def __call__(self, *args, **kwargs):
        r"""
        The provided args and kwargs change depending on the global optimization
        method. This class is equipped to distinguish the callback function for
        differential_evolution (length equal to ) and basinhopping.

        Parameters
        ----------
        args
            The provided args change depending on the global optimization method.

            - x_new (numpy.ndarray) - Current parameter values being evaluated, used
            in both algorithms
            - f_new (float) - Current object function value for x_new, used in
            basinhopping
            - accept (bool) - Whether or not that minimum was accepted, used in
            basinhopping

        kwargs
            The provided kwargs change depending on the global optimization method.

            - convergence (float) - Used in differential evolution, the fractional
            value of the population convergence. When greater than one the function
            halts.

        Returns
        -------
        value : bool
            A true or false value that says whether the guess in parameter value is
            within bounds

        """

        if "convergence" in kwargs:  # Used in differential_evolution
            if kwargs["convergence"] < self.obj_cut:
                if not os.path.isfile(self.filename):
                    with open(self.filename, "w") as f:
                        f.write("# n, convergence, {}\n".format(", ".join(self.beadnames)))

                with open(self.filename, "a") as f:
                    tmp = [self.ninit, kwargs["convergence"]] + list(args[0])
                    f.write(("{}, " * len(tmp)).format(*tmp) + "\n")

        elif len(args) == 3:  # used in basinhopping
            if args[2] or args[1] < self.obj_cut:
                if not os.path.isfile(self.filename):
                    with open(self.filename, "w") as f:
                        f.write("# n, obj. value, accepted, {}\n".format(", ".join(self.beadnames)))

                with open(self.filename, "a") as f:
                    tmp = [self.ninit, args[1], args[2]] + list(args[0])
                    f.write(("{}, " * len(tmp)).format(*tmp) + "\n")
        else:
            raise ValueError(
                "Unknown inputs. This function is equipped to handle "
                "differential_evolution and basinhopping algorithms."
            )

        self.ninit += 1

        return False


def _del_Data_MultiprocessingObject(dictionary):
    r"""A dictionary of fitting objects will remove MultiprocessingObject attributes
    so that the multiprocessing pool can be used by the fitting algorithm.

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
