
import numpy as np
import logging

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

    #logger = logging.getLogger(__name__)

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

        #logger = logging.getLogger(__name__)
        
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

        #logger = logging.getLogger(__name__)
        
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
            print(x, j)
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
        return tmax and tmin


def compute_SAFT_obj(beadparams, opt_params, eos, exp_dict):
    r"""
    Fit defined parameters for equation of state object with given experimental data. 

    Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    beadparams0 : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    opt_params : dict
        Parameters used in basin fitting algorithm

        - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
        - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).

    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    exp_dict : dict
        Dictionary of experimental data objects.

    Returns
    -------
    Objective : float
        Value of sum of objective values according to appropriate weights. Output file saved in current working directory.
        
    """

    logger = logging.getLogger(__name__)

    # Update beadlibrary with test paramters
    for i, param in enumerate(opt_params['fit_params']):
        fit_params_list = param.split("_")
        if len(fit_params_list) == 1:
            eos.update_parameters(fit_params_list[0], [opt_params['fit_bead']], beadparams[i])
        elif len(fit_params_list) == 2:
            eos.update_parameters(fit_params_list[0], [opt_params['fit_bead'], fit_params_list[1]], beadparams[i])
        else:
            raise ValueError("Parameters for only one bead are allowed to be fit at one time. Please only list one bead type in your fit parameter name.")
    eos.parameter_refresh()

    # Compute obj_function
    obj_function = []
    for key,data_obj in exp_dict.items():
        try:
            obj_function.append(data_obj.objective(eos))
        except:
            raise ValueError("Failed to evaluate objective function for {} of type {}.".format(key,data_obj.name))

    # Write out parameters and objective functions for each dataset
    logger.info("\nParameters: {}\nValues: {}\nExp. Data: {}\nObj. Values: {}\nTotal Obj. Value: {}".format(opt_params['fit_params'],beadparams,list(exp_dict.keys()),obj_function,sum(obj_function)))

    return sum(obj_function)

