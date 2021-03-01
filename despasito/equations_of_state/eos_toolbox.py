import numpy as np
import logging
from inspect import getmembers, isfunction

from . import combining_rule_types

logger = logging.getLogger(__name__)


def remove_insignificant_components(xi_old, massi_old):
    r"""
    This function will remove any components with mole fractions less than or equal to machine precision.

    Parameters
    ----------
    xi_old : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi_old : numpy.ndarray
        Mass for each component [kg/mol]

    Returns
    -------
    xi_new : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi_new : numpy.ndarray
        Mass for each component [kg/mol]

    """
    ind = np.where(np.array(xi_old) < np.finfo(float).eps)[0]
    xi_new = []
    massi_new = []
    for i in range(len(xi_old)):
        if i not in ind:
            xi_new.append(xi_old[i])
            massi_new.append(massi_old[i])
    xi_new = np.array(xi_new)
    massi_new = np.array(massi_new)

    return xi_new, massi_new


def partial_density_central_difference(
    xi, rho, T, func, step_size=1e-2, log_method=False
):
    """
    Take the derivative of a dependent variable calculated with a given function using the central difference method.
    
    Parameters
    ----------
    xi : list[float]
        Mole fraction of each component
    rho : float
        Molar density of system [mol/m^3]
    T : float
        Temperature of the system [K]
    func : function
        Function used in job to calculate dependent factor. This function should have a single output. Inputs arguments should be (rho, T, xi)
    step_size : float, Optional, default=1E-2
        Step size used in central difference method
    log_method : bool, Optional, default=False
        Choose to use a log transform in central difference method. This allows easier calculations for very small numbers.
        
    Returns
    -------
    dydxi : numpy.ndarray
        Array of derivative of y with respect to xi
    """

    dAdrho = np.zeros(len(xi))

    if log_method:  # Central Difference Method with log(y) transform

        dy = step_size
        y = np.log(rho * np.array(xi, float))
        for i in range(np.size(dAdrho)):
            if xi[i] != 0.0:
                Ares = np.zeros(2)
                for j, delta in enumerate((dy, -dy)):
                    y_temp = np.copy(y)
                    y_temp[i] += delta
                    Ares[j] = _partial_density_wrapper(np.exp(y_temp), T, func)
                dAdrho[i] = (Ares[0] - Ares[1]) / (2.0 * dy) / np.exp(y[i])
            else:
                dAdrho[i] = np.finfo(float).eps

    else:  # Traditional Central Difference Method

        dy = step_size
        y = rho * np.array(xi, float)
        for i in range(np.size(dAdrho)):
            if xi[i] != 0.0:
                Ares = np.zeros(2)
                for j, delta in enumerate((dy, -dy)):
                    y_temp = np.copy(y)
                    if y_temp[i] != 0.0:
                        y_temp[i] += delta
                    Ares[j] = _partial_density_wrapper(y_temp, T, func)
                dAdrho[i] = (Ares[0] - Ares[1]) / (2.0 * dy)
            else:
                dAdrho[i] = np.finfo(float).eps

    return dAdrho


def _partial_density_wrapper(rhoi, T, func):
    """
    Compute derivative of Helmholtz energy with respect to density.
    
    Parameters
    ----------
    rhoi : float
        Molar density of each component, add up to the total density [mol/m^3]
    T : float
        Temperature of the system [K]
    func : function
        Function used in job to calculate dependent factor. This function should have a single output.
    
    Returns
    -------
    Ares : float
        Helmholtz energy give number of moles, length of array rho
    """

    # Calculate new xi values
    rho = np.array([np.sum(rhoi)])
    xi = rhoi / rho

    Ares = func(rho, T, xi)

    return Ares


def calc_massi(molecular_composition, bead_library, beads):
    r"""
    This function extracted the mass of each component
    
    Parameters
    ----------
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
        
        - mass: Bead mass [kg/mol]
    
    beads : list[str]
        List of unique bead names used among components
    
    Returns
    -------
    massi : numpy.ndarray
        Bead mass corresponding to array 'beads' [kg/mol]
    """
    massi = np.zeros(len(molecular_composition))
    for i in range(len(molecular_composition)):
        for k, bead in enumerate(beads):
            if "mass" in bead_library[bead]:
                massi[i] += molecular_composition[i, k] * bead_library[bead]["mass"]
            else:
                raise ValueError(
                    "The mass for bead, {}, was not provided.".format(bead)
                )

    return massi


def extract_property(prop, bead_library, beads, default=None):
    r"""
    Extract single property or key from a dictionary within a dictionary (e.g. bead parameters) and into a single array of the same length and order as a list of bead names.

    The expected structure is a dictionary of dictionaries, such as a parameter library.
    
    Parameters
    ----------
    property : str
        Name of property in bead_library
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    beads : list[str]
        List of unique bead names used among components
    default : any, Optional, default=None
        If property is not present, set to this value. Although if the default is None, an error will result.
    
    Returns
    -------
    prop_array : numpy.ndarray
        array of desired property
    """

    prop_array = np.zeros(len(beads))
    for i, bead in enumerate(beads):
        if prop in bead_library[bead]:
            prop_array[i] = bead_library[bead][prop]
        else:
            if default == None:
                raise ValueError(
                    "The property {} for bead, {}, was not provided.".format(prop, bead)
                )
            else:
                prop_array[i] = default

    return prop_array


def check_bead_parameters(bead_library0, parameter_defaults):
    r"""
    Be sure all needed parameters are available for each bead.

    If a parameter is absent and a default value is given, this value will be added to the parameter set. If the default is None, then an error is raised.

    Parameters
    ----------
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters
    parameter_defaults : dict
        A dictionary of default values for the required parameters.

    Returns
    -------
    new_bead_library : dict
        New dictionary with defaults added where relevant

    """

    bead_library = bead_library0.copy()

    for bead, bead_dict in bead_library.items():
        for parameter, default in parameter_defaults.items():
            if parameter not in bead_dict:
                if default != None:
                    bead_library[bead][parameter] = default
                    logger.info(
                        "Parameter, {}, is missing for parametrized group, {}. Set to default, {}".format(
                            parameter, bead, default
                        )
                    )
                else:
                    raise ValueError(
                        "Parameter, {}, should have been defined for parametrized group, {}.".format(
                            parameter, bead
                        )
                    )

    return bead_library


def cross_interaction_from_dict(beads, bead_library, mixing_dict, cross_library={}):
    r"""
    Computes matrices of cross interaction parameters defined as the keys in the mixing dict parameter are extracted from the bead_library and then the cross library.
        
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. Those to be calculated are defined by the keys of mixing_dict
    mixing_dict : dict
        This dictionary contains those bead parameters that should be placed in a matrix and the mixing rules for the cross parameters
    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.
        
    Returns
    -------
    output : dict
        Dictionary of outputs, with the same keys used in mixing dict for the respective interaction matrix
        
    """

    nbeads = len(beads)

    # Set-up output dictionaries
    output = {}
    for key in mixing_dict:
        output[key] = np.zeros((nbeads, nbeads))
        for k in range(nbeads):
            output[key][k, k] = bead_library[beads[k]][key]

    # Put cross_library in output matrices
    for (i, beadname) in enumerate(beads):
        for (j, beadname2) in enumerate(beads):
            if j > i:
                for key in mixing_dict:
                    if (
                        cross_library.get(beadname, {})
                        .get(beadname2, {})
                        .get(key, None)
                        is not None
                    ):
                        output[key][i, j] = cross_library[beadname][beadname2][key]
                    elif (
                        cross_library.get(beadname2, {})
                        .get(beadname, {})
                        .get(key, None)
                        is not None
                    ):
                        output[key][i, j] = cross_library[beadname2][beadname][key]
                    else:
                        try:
                            tmp = combining_rules(
                                bead_library[beadname],
                                bead_library[beadname2],
                                key,
                                **mixing_dict[key]
                            )
                        #                            if mixing_dict[key]["function"]=="multipole":
                        #                                logger.debug("Multipole: {} {}, {}".format(beadname,beadname2,tmp))
                        except Exception:
                            raise ValueError(
                                "Unable to calculate '{}' with '{}' method, for beads: '{}' '{}'".format(
                                    key,
                                    mixing_dict[key]["function"],
                                    beadname,
                                    beadname2,
                                )
                            )
                        for k2, v2 in tmp.items():
                            output[k2][i, j] = v2
                            output[k2][j, i] = v2
                    output[key][j, i] = output[key][i, j]

    return output


def construct_dummy_bead_library(input_dict, keys=None):
    r"""
    Using arrays of values, a dictionary is populated like a bead_library. 

    If keys are included, they are used, otherwise, integers are used.
        
    Parameters
    ----------
    input_dict : dict
        A dictionary where parameter names are the keys to access EOS arrays of self-interaction parameters.
    keys : list[str], Optional, default=None
        List must be the same length as the lists in `input_dict`. These are used as labels.
        
    Returns
    -------
    output_dict : dict
        Dictionary of outputs, with the same keys used in mixing dict for the respective interaction matrix
        
    """

    output = {}
    flag = False
    for parameter in input_dict:
        if keys == None:
            keys = [str(x) for x in range(len(input_dict[parameter]))]
            flag = True
        if len(keys) != len(input_dict[parameter]):
            raise ValueError(
                "Number of keys is not equal to the number of quantities given"
            )

        for i, bead in enumerate(keys):
            if bead not in output:
                output[bead] = {}
            output[bead][parameter] = input_dict[parameter][i]

    if flag:
        return output, keys
    else:
        return output


def combining_rules(beadA, beadB, parameter, function="mean", **kwargs):
    r"""
    Calculates cross interaction parameter according to the calculation method defined.
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    function : str, Optional, default=mean
        Mixing rule function found in `despasito.equations_of_state.combining_rule_types.py`
    kwargs : dict, Optional, default={}
        Keyword arguments used in other averaging function
        
    Returns
    -------
    output_dict : dict
        Dictionary with keyword of parameter and Mixed interaction parameter. If mixing rule type outputs more than one updated variable, it will also be included
    """

    calc_list = [o[0] for o in getmembers(combining_rule_types) if isfunction(o[1])]
    try:
        if function != "None":
            func = getattr(combining_rule_types, function)
    except Exception:
        raise ImportError(
            "The mixing rule type, '{}', was not found\nThe following calculation types are supported: {}".format(
                function, ", ".join(calc_list)
            )
        )

    if function != "None":
        output = func(beadA, beadB, parameter, **kwargs)
        if not isinstance(output, dict):
            tmp = {parameter: output}
            output = tmp
    else:
        output = {}

    return output
