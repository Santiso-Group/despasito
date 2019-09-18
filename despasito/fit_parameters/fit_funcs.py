
import os
import numpy as np
import logging

def reformat_ouput(cluster):
    """
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
    type_cluster = [type(x) for x in cluster]

    if len(type_cluster) == 1 and type_cluster not in [list,np.ndarray,tuple]:
        matrix = np.array(cluster).T
        len_cluster = [1. in x in range(len(type_cluster))]
    elif type_cluster not in [list,np.ndarray,tuple]:
        matrix = np.array(cluster).T
        len_cluster = [1. for x in range(len(type_cluster))]
    else:
        len_cluster = []
        for i,typ in enumerate(type_cluster):
            if typ in [list,np.ndarray,tuple]:
                len_cluster.append(len(cluster[i]))
            else:
                len_cluster.append(1)
        matrix_tmp = np.zeros([len(cluster), sum(len_cluster)])

        for i, val in enumerate(cluster):
            ind = 0
            for j,l in enumerate(len_cluster):
                if l == 1:
                    matrix_tmp[i, ind] = val[j]
                else:
                    matrix_tmp[i, ind:ind+l+1] = val[j]
                ind += l
        matrix = np.array(matrix_tmp).T

    return matrix, len_cluster

class BasinStep(object):
    """
    Custom basin step.

    Attributes
    ----------
    stepmag : list
        List of step magnitudes
    stepsize : float
        default 0.05

    Returns
    -------
    basinstep : numpy.ndarray
        Suggested basin steps used in basinhopping algorithm
    """
    def __init__(self, stepmag, stepsize=0.05):
        self._stepsize = stepsize
        self._stepmag = stepmag

    def __call__(self, x):

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
                # If a value of x is negative break cycle
                if x[i] < 0.0:
                    breakloop = False
            if breakloop: break
            print(x, j)
        return x

def compute_SAFT_obj(beadparams, opt_params, eos, exp_dict, output_file="fit_parameters.txt"):
    """
    Fit defined parameters for equation of state object with given experimental data. Each set of experimental data is converted to an object with the build in ability to evaluate its part of objective function. 
    To add another type of supported experimental data, add a class to the fit_classes.py file.

    Parameters
    ----------
    beadparams0 : numpy.ndarray, 
        An array of initial guesses for parameters, these will be optimized throughout the process.
    opt_params : dict
        A dictionary of the parameter fitting information.
        * fit_bead : str, Name of the bead (i.e. group/segment) whose 
parars are being fit
        * fit_params : list[str], A list of parameters to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        * bounds 
    eos : obj
        Equation of state output that writes pressure, max density, and chemical potential
    exp_dict : dict
        Dictionary of experimental data objects.
    output_file : str
        Output file name

    Returns
    -------
        Output file saved in current working directory
    """

    for i, boundval in enumerate(opt_params['bounds']):
        if (beadparams[i] > boundval[1]) or (beadparams[i] < boundval[0]):
            beadparams[i] = np.mean(boundval)

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
            raise ValueError("Failed to evaluate objective function for %s of type %s." % (key,data_obj.name))

    # Write out parameters and objective functions for each dataset
    if os.path.isfile(output_file):
        with open(output_file,"a") as f:
            tmp = [beadparams.tolist() + obj_function + [sum(obj_function)]]
            tmp = [str(x) for x in tmp]
            f.write(", ".join(tmp)+"\n")
    else:
        with open(output_file,"w") as f:
            f.write(", ".join(opt_params['fit_params']+list(exp_dict.keys())+["total obj"])+"\n")
            tmp = [beadparams.tolist() + obj_function + [sum(obj_function)]]
            tmp = [str(x) for x in tmp]
            f.write(", ".join(tmp)+"\n")


    return sum(obj_function)

