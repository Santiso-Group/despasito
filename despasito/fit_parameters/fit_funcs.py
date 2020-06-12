
import numpy as np
import logging
import scipy.optimize as spo

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

class BasinStep(object):
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

class BasinBounds(object):
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

        logger.info("Reject parameters: {}, with obj. function: {}".format(x,kwargs["f_new"]))

        return tmax and tmin and feasible1 and feasible2


def global_minimization(global_method, beadparams0, bounds, fit_bead, fit_params, exp_dict, global_dict={}, minimizer_dict={}):
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
        Kwargs of golobal optimization algorithm. Here we list the kwargs used in scipy.optimize.basinhopping

        - niter (int) - default: 10, Number of basin hopping iterations
        - T (float) - default: 0.5, Temperature parameter, should be comparable to separation between local minima (i.e. the “height” of the walls separating values).
        - niter_success (int) - default: 3, Stop run if minimum stays the same for this many iterations
        - stepsize (float) - default: 0.1, Maximum step size for use in the random displacement. We use this value to define an object for the `take_step` option that includes a custom routine that produces attribute stepsizes for each parameter.

    minimizer_dict : dict, Optional
        Dictionary used to define minimization type and the associated options.

        - method (str) - Method available to scipy.optimize.minimize
        - options (dict) - This dictionary contains the kwargs available to the chosen method

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    # !!!!!!!!! If another methods is added to the if statement below, please also add it here and update the documentation above. !!!!!!!!
    methods = ["basinhopping", "differential_evolution", "brute"]
    logger.info("Using global optimization method: {}".format(global_method))

    if global_method == "basinhopping":

        # Options for basin hopping
        new_global_dict = {"niter": 10, "T": 0.5, "niter_success": 3}
        if global_dict:
            for key, value in global_dict.items():
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
            global_dict["take_step"] = BasinStep(stepmag, stepsize=stepsize)
            custombounds = BasinBounds(bounds)
        except:
        	raise TypeError("Could not initialize BasinStep and/or BasinBounds")

        result = spo.basinhopping(compute_obj, beadparams0, **global_dict, accept_test=custombounds, disp=True, minimizer_kwargs={"args": (fit_bead, fit_params, exp_dict, bounds),**minimizer_dict})

    elif global_method == "differential_evolution":

        # Options for differential evolution, set defaults in new_global_dict
        new_global_dict = {}
        if global_dict:
            for key, value in global_dict.items():
                new_global_dict[key] = value
        global_dict = new_global_dict

        result = spo.differential_evolution(compute_obj, bounds, args=(fit_bead, fit_params, exp_dict, bounds), **global_dict)

    elif global_method == "brute":

        # Options for brute, set defaults in new_global_dict
        new_global_dict = {}
        if global_dict:
            for key, value in global_dict.items():
                new_global_dict[key] = value
        global_dict = new_global_dict

        result = spo.brute(compute_obj, bounds, args=(fit_bead, fit_params, exp_dict, bounds), **global_dict)

    else:
        raise ValueError("Global optimization method, {}, is not currently supported. Try: {}".format(global_method,", ".join(methods)))

    return result


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
        This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
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

    # Compute obj_function
    obj_function = []
    for key,data_obj in exp_dict.items():
        try:
            for i, param in enumerate(fit_params):
                data_obj.eos.update_parameters(fit_bead, param, beadparams[i])
            data_obj.eos.parameter_refresh()
            obj_function.append(data_obj.objective())
        except:
            raise ValueError("Failed to evaluate objective function for {} of type {}.".format(key,data_obj.name))

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

