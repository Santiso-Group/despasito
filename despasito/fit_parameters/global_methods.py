
import os
import numpy as np
import logging
import scipy.optimize as spo
from inspect import getmembers

from despasito.utils.parallelization import MultiprocessingJob
from . import fit_funcs as ff
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)

def single_objective(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}):
    r"""
    Evaluate parameter set for equation of state with given experimental data

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    bounds : list[tuple]
        List of length equal to fit_params with lists of pairs for minimum and maximum bounds of parameter being fit. Defaults are broad, recommend specification.
    fit_bead : str
        Name of bead whose parameters are being fit, should be in bead list of beadconfig
    fit_params : list[str]
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
    exp_dict : dict
        Dictionary of experimental data objects.

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    if len(global_dict) > 0:
        logger.info("The fitting method 'single_objective' does not have further options")

    obj_value = ff.compute_obj(beadparams0, fit_bead, fit_params, exp_dict, bounds)

    result = spo.OptimizeResult(x=beadparams0,
                                fun=obj_value,
                                success=True,
                                nit=0,
                                message="Successfuly computed objective function for provided parameter set." )

    return result

def differential_evolution(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}, constraints=None):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.differential_evolution with given experimental data. 

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
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

        - init (str) - type of initiation for population, Optional, default="random" 
        - write_intermediate_file (str) - If True, an intermediate file will be written from the method callback, default: False
        - filename (str) - filename for callback output, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - obj_cut (float) - Cut-off objective value to write the parameters, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    constraints : dict, Optional, default=None
        This dicitonary of constraint types and their arguements will be converted into a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    obj_kwargs = ["obj_cut", "filename", "write_intermediate_file"]
    if "obj_cut" in global_dict:
        obj_cut = global_dict["obj_cut"]
        global_dict["write_intermediate_file"] = True
    else:
        obj_cut = None

    if "filename" in global_dict:
        filename = global_dict["filename"]
        global_dict["write_intermediate_file"] = True
    else:
        filename = None

    if "write_intermediate_file" in global_dict and global_dict["write_intermediate_file"]:
        global_dict["callback"] = _WriteParameterResults(fit_params, obj_cut=obj_cut, filename=filename)

    # Options for differential evolution, set defaults in new_global_dict
    new_global_dict = {"init": "random"}
    if global_dict:
        for key, value in global_dict.items():
            if key is "mpObj":
                if value.ncores > 1:
                    logger.info("Differential Evolution algoirithm is using {} workers.".format(value.ncores))
                    new_global_dict["workers"] = value._pool.map
                    exp_dict = _del_Data_mpObj(exp_dict)
            elif key not in obj_kwargs:
                new_global_dict[key] = value
    global_dict = new_global_dict

    if constraints is not None:
        global_dict["constraints"] = ff.initialize_constraints(constraints, "class")
    logger.info("Differential Evolution Options: {}".format(global_dict))

    result = spo.differential_evolution(ff.compute_obj, bounds, args=(fit_bead, fit_params, exp_dict, bounds), **global_dict)

    return result

def shgo(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}, minimizer_dict={}, constraints=None):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.shgo with given experimental data. 

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
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

        - init (str) - type of initiation for population, Optional, default="random" 
        - write_intermediate_file (str) - If True, an intermediate file will be written from the method callback, default: False
        - filename (str) - filename for callback output, if provided, `write_intermediate_file` will be set to True, Optional, default=None
        - obj_cut (float) - Cut-off objective value to write the parameters, if provided, `write_intermediate_file` will be set to True, Optional, default=Non
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize, Optional, default=nelder-mead
        - options (dict) - This dictionary contains the kwargs available to the chosen method, Optional, default={'maxiter': 50}

    constraints : dict, Optional, default=None
        This dicitonary of constraint types and their arguements will be converted into a tuple of dictionaries that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    obj_kwargs = ["obj_cut", "filename", "write_intermediate_file"]
    if "obj_cut" in global_dict:
        obj_cut = global_dict["obj_cut"]
        global_dict["write_intermediate_file"] = True
    else:
        obj_cut = None

    if "filename" in global_dict:
        filename = global_dict["filename"]
        global_dict["write_intermediate_file"] = True
    else:
        filename = None

    if "write_intermediate_file" in global_dict and global_dict["write_intermediate_file"]:
        global_dict["callback"] = _WriteParameterResults(fit_params, obj_cut=obj_cut, filename=filename)

    # Options for differential evolution, set defaults in new_global_dict
    new_global_dict = {"sampling_method": "sobol"}
    if global_dict:
        for key, value in global_dict.items():
            if key is not "mpObj" and key not in obj_kwargs:
                new_global_dict[key] = value
    global_dict = new_global_dict

    # Set up options for minimizer in basin hopping
    new_minimizer_dict = {"method": 'nelder-mead', "options": {'maxiter': 50}}
    if minimizer_dict:
        for key, value in minimizer_dict.items():
            if key == "method":
                new_minimizer_dict[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_dict[key][key2] = value2
    minimizer_dict = new_minimizer_dict

    if constraints is not None:
        global_dict["constraints"] = ff.initialize_constraints(constraints, "dict")
        if minimizer_dict["method"] not in ["COBYLA", "SLSQP"]:
            minimizer_dict["method"] = "SLSQP"
            for key, value in minimizer_dict["options"].items():
                if key not in ["ftol", "eps", "disp", "maxiter", "finite_diff_rel_step"]:
                    del minimizer_dict["options"][key]

    if minimizer_dict:
        logger.warning("Minimization options were given but aren't used in this method.")

    result = spo.shgo(ff.compute_obj, bounds, args=(fit_bead, fit_params, exp_dict, bounds), **global_dict)

    return result

def grid_minimization(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}, minimizer_dict={}, constraints=None):
    r"""
    Fit defined parameters for equation of state object using a custom adaptation of scipy.optimize.brute with given experimental data. 

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
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

        - Ns (int) - Number of grid points along the axes, Optional, default=5
        - finish (callable) - An optimization function, default=scipy.optimize.minimize
        - initial_guesses (list) - Replaces grid of values generated with bounds and Ns
        - etc. Other keywords for scipy.optimize.differential_evolution use the function defaults

    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to our `solve_root` function, Optional, default="lm"
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    constraints : dict, Optional, default=None
        This dicitonary of constraint types and their arguements will be converted into a tuple of constraint classes that is compatible

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

     # Options for brute, set defaults in new_global_dict
    flag_use_mp_object = False
    new_global_dict = {"Ns": 5, "finish":spo.minimize}
    if global_dict:
        for key, value in global_dict.items():
            if key is "mpObj":
                if value.ncores > 1:
                    logger.info("Grid minimization algoirithm is using {} workers.".format(value.ncores))
                    new_global_dict["mpObj"] = value
                    flag_use_mp_object = True
                    exp_dict = _del_Data_mpObj(exp_dict)
            else:
                new_global_dict[key] = value
    global_dict = new_global_dict

    if constraints is not None:
        global_dict["constraints"] = ff.initialize_constraints(constraints, "dict")

    logger.info("Grid Minimization Options: {}".format(global_dict))

    # Set up options for minimizer
    new_minimizer_dict = {"method": 'lm'}
    #new_minimizer_dict = {"method": 'L-BFGS-B'}
    if minimizer_dict:
        for key, value in minimizer_dict.items():
            if key == "method":
                new_minimizer_dict[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_dict[key][key2] = value2
    minimizer_dict = new_minimizer_dict
    logger.info("    Minimizer Options: {}".format( minimizer_dict))

    args = (fit_bead, fit_params, exp_dict, bounds)

    # Set up inputs
    if "initial_guesses" in global_dict:
        x0_array = global_dict["initial_guesses"]
    else:
        # Initialization taken from scipy.optimize.brute
        N = len(bounds)
        lrange = list(bounds)
        for k in range(N):
            if type(lrange[k]) is not type(slice(None)):
                if len(lrange[k]) < 3:
                    lrange[k] = tuple(lrange[k]) + (complex(global_dict["Ns"]),)
                lrange[k] = slice(*lrange[k])
        if (N == 1):
            lrange = lrange[0]
        x0_array = np.mgrid[lrange]
        inpt_shape = x0_array.shape
        if (N > 1):
            x0_array = np.reshape(x0_array, (inpt_shape[0], np.prod(inpt_shape[1:]))).T

    inputs = [(x0, args, bounds, constraints, minimizer_dict) for x0 in x0_array]

    # Start computation
    if flag_use_mp_object:
        x0, results, fval = global_dict["mpObj"].pool_job(_grid_minimization_wrapper, inputs)
    else:
        x0, results, fval = MultiprocessingJob.serial_job(_grid_minimization_wrapper, inputs)

    # Choose final output
    result = [fval[0], results[0]]
    logger.info("For bead: {} and parameters {}".format(fit_bead,fit_params))
    for i in range(len(x0_array)):
        tmp_result = results[i]
        logger.info("x0: {}, xf: {}, obj: {}".format(x0_array[i], results[i], fval[i]))
        if result[0] > fval[i]:
            result = [fval[i], tmp_result]

    result = spo.OptimizeResult(x=result[1],
                                fun=result[0],
                                success=True,
                                nit=len(x0)*global_dict["Ns"],
                                message="Termination successful with {} grid points and the minimum value minimized. Note that parameters may be outside of the given bounds because of the minimizing function.".format(len(x0)*global_dict["Ns"]))

    return result

def brute(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.brute with given experimental data. 

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
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

        - Ns (int) - Number of grid points along the axes, Optional, default=5
        - finish (callable) - An optimization function, default=scipy.optimize.minimize
        - etc. Other keywords for scipy.optimize.brute use the function defaults

    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    # Options for brute, set defaults in new_global_dict
    new_global_dict = {"Ns": 5, "finish":spo.minimize}
    if global_dict:
        for key, value in global_dict.items():
            if key is "mpObj":
                if value.ncores > 1:
                    logger.info("Brute algoirithm is using {} workers.".format(value.ncores))
                    new_global_dict["workers"] = value._pool.map
                    exp_dict = _del_Data_mpObj(exp_dict)
            else:
                new_global_dict[key] = value
    global_dict = new_global_dict
    global_dict["full_output"] = True

    logger.info("Brute Options: {}".format(global_dict))
    x0, fval, grid, Jount = spo.brute(ff.compute_obj, bounds, args=(fit_bead, fit_params, exp_dict, bounds), **global_dict)
    result = spo.OptimizeResult(x=x0,
                                fun=fval,
                                success=True,
                                nit=len(x0)*global_dict["Ns"],
                                message="Termination successful with {} grid points and the minimum value minimized. Note that parameters may be outside of the given bounds because of the minimizing function.".format(len(x0)*global_dict["Ns"]))

    return result

def basinhopping(beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}, minimizer_dict={}):
    r"""
    Fit defined parameters for equation of state object using scipy.optimize.basinhopping with given experimental data. 

    Parameters
    ----------
    beadparams0 : numpy.ndarray 
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

        - niter (int) - The number of basin-hopping iterations, Optional, default=10 
        - T (float) - The "temperature" parameter for the accept or reject criterion. For best results T should be comparable to the separation (in function value) between local minima., Optional, default=0.5
        - niter_success (int) Stop the run if the global minimum candidate remains the same for this number of iterations., Optional, default=3
        - stepsize (float) - Maximum step size for use in the random displacement., Optional, default=0.1
        - take_step (callable) - Set with custom BasinStep class
        - etc. Other keywords for scipy.optimize.basinhopping use the function defaults

    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    Returns
    -------
    Objective : obj
        scipy OptimizedResult
    """

    # Options for basin hopping
    new_global_dict = {"niter": 10, "T": 0.5, "niter_success": 3}
    if global_dict:
        for key, value in global_dict.items():
            if key is not "mpObj":
                new_global_dict[key] = value
    global_dict = new_global_dict

    # Set up options for minimizer in basin hopping
    new_minimizer_dict = {"method": 'nelder-mead', "options": {'maxiter': 50}}
    if minimizer_dict:
        for key, value in minimizer_dict.items():
            if key == "method":
                new_minimizer_dict[key] = value
            elif key == "options":
                for key2, value2 in value.items():
                    new_minimizer_dict[key][key2] = value2
    minimizer_dict = new_minimizer_dict

    # NoteHere: how is this array generated? stepmag = np.array([550.0, 26.0, 4.0e-10, 0.45, 500.0, 150.0e-30, 550.0])
    try:
        if "stepsize" in global_dict:
            stepsize = global_dict["stepsize"]
        else:
            stepsize = 0.1
        stepmag = np.transpose(np.array(bounds))[1]
        global_dict["take_step"] = _BasinStep(stepmag, stepsize=stepsize)
        custombounds = _BasinBounds(bounds)
    except:
            raise TypeError("Could not initialize BasinStep and/or BasinBounds")

    logger.info("Basin Hopping Options: {}".format(global_dict))
    result = spo.basinhopping(ff.compute_obj, beadparams0, **global_dict, accept_test=custombounds, minimizer_kwargs={"args": (fit_bead, fit_params, exp_dict, bounds),**minimizer_dict})

    return result

# ______________________________________________ Supporting Classes and Functions ___________________________________________
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
        result = gtb.solve_root( ff.compute_obj, args=obj_args, method=method, x0=x0, bounds=bounds, options=opts)
    except:
        result = np.nan*np.ones(len(x0))

    if np.sum(np.abs(result-x0)) < 1e-6:
        result = np.nan*np.ones(len(x0))

    logger.info("Starting parameters: {}, converged to: {}".format(x0,result))

    fval = ff.compute_obj(result, *obj_args)

    return x0, result, fval

class _BasinStep(object):
    r"""
    Custom basin step used by scipy.optimize.basinhopping function.
    
    """

    def __init__(self, stepmag, stepsize=0.05):
        r"""
            
        Parameters
        ----------
        stepmag : list
            List of step magnitudes
        stepsize : float
            default 0.05

        Attributes
        ----------
        stepmag : list
            List of step magnitudes
        stepsize : float
            default 0.05
            
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

        # Save intital guess in array
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
            if breakloop: break
            logger.info("Basin Step after {} iterations:\n    {}".format(j,x))
        return x

class _BasinBounds(object):
    r"""
    Object used by scipy.optimize.basinhopping to set bounds of parameters.
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
            Array of minimun values for each parameter
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
        x_new : numpy.ndarray
            Guess in parameters values

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

        if (tmax and tmin and feasible1 and feasible2):
            logger.info("Accept parameters: {}, with obj. function: {}".format(x,kwargs["f_new"]))
        else:
            logger.info("Reject parameters: {}, with obj. function: {}".format(x,kwargs["f_new"]))

        return tmax and tmin and feasible1 and feasible2

class _WriteParameterResults(object):
    r"""
    Object used by scipy.optimize.basinhopping to set bounds of parameters.
    """

    def __init__(self, beadnames, obj_cut=None, filename=None):
        r"""
            
        Attributes
        ----------
        beadnames : list[str]
            List of bead names for filename header
        obj_cut : float, Optional, default: np.inf
            Cut-off objective value to write the parameters
        filename : str, Optional, default: parameters.txt
            File to which parameters are written
            
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
                filename = "{}_{}".format(i,old_fname)
                if not os.path.isfile(filename):
                    logger.info("File '{}' already exists, using {}.".format(old_fname,filename))
                    break
        self.filename = filename
        self.ninit = 0

        with open(filename, 'w') as f:
            f.write("# n, convergence, "+", ".join(beadnames)+"\n")

    def __call__(self, x_new, convergence=0):
        r"""
            
        Parameters
        ----------
        x_new : numpy.ndarray
            Guess in parameters values

        Returns
        -------
        value : bool
            A true or false value that says whether the guess in parameter value is within bounds
            
        """

        if convergence < self.obj_cut:
            with open(self.filename, 'a') as f:
                tmp = [self.ninit, convergence]+[x for x in x_new]
                f.write(("{}, "*len(tmp)).format(*tmp)+"\n")

        self.ninit += 1

        return False

def _del_Data_mpObj(dictionary):
    r""" A dictionary of fitting objects will remove mpObj attributes so that the multiprocessing pool can be used by the fitting algorithm.

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
        if "mpObj" in new_dictionary[key]._thermodict:
            del new_dictionary[key]._thermodict["mpObj"]

    return new_dictionary
