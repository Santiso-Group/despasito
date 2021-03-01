""" Combining rules options called from :func:`~despasito.equations_of_state.eos_toolbox.combining_rules` in the EOS class.
"""

import numpy as np
import logging

# Special functions are imported as needed

logger = logging.getLogger(__name__)


def mean(beadA, beadB, parameter):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    mean: c = (a+b)/2
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
        
    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    return (beadA[parameter] + beadB[parameter]) / 2


def geometric_mean(beadA, beadB, parameter):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    geometric mean: c = np.sqrt(a*b)
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
        
    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    return np.sqrt(beadA[parameter] * beadB[parameter])


def volumetric_geometric_mean(beadA, beadB, parameter, weighting_parameters=[]):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    volumetric geometric mean: c = np.sqrt(a[0]*b[0]) * np.sqrt(a[1]**3 * b[1]**3) / ((a[1] + b[1])/2)**3
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b1.

    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    tmp1 = np.sqrt(beadA[parameter] * beadB[parameter])
    param2 = weighting_parameters[0]
    tmp2 = (
        np.sqrt((beadA[param2] ** 3) * (beadB[param2] ** 3))
        * 8
        / ((beadA[param2] + beadB[param2]) ** 3)
    )
    return tmp1 * tmp2


def weighted_mean(beadA, beadB, parameter, weighting_parameters=[]):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    weighted mean: (a[0]*a[1] + b[0]*b[1]) / (a[1] + b[1])
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b1.

    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    param2 = weighting_parameters[0]
    parameter12 = (
        beadA[parameter] * beadA[param2] + beadB[parameter] * beadB[param2]
    ) / (beadA[param2] + beadB[param2])

    return parameter12


def mie_exponent(beadA, beadB, parameter):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    mie_exponent: 3 + np.sqrt((a-3)*(b-3))
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed

    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    return 3 + np.sqrt((beadA[parameter] - 3.0) * (beadB[parameter] - 3.0))


def square_well_berthelot(beadA, beadB, parameter, weighting_parameters=[]):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    square_well Berthelot geometric mean: c = np.sqrt(a[0]*b[0]) * np.sqrt(a[1]**3 * b[1]**3) / ((a[1] + b[1])/2)**3
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b1.

    Returns
    -------
    parameter12 : float
        Mixed interaction parameter

    """

    param2, param3 = weighting_parameters[0], weighting_parameters[1]

    tmp1 = np.sqrt(beadA[parameter] * beadB[parameter])
    tmp2 = (
        np.sqrt((beadA[param2] ** 3) * (beadB[param2] ** 3))
        * 8
        / ((beadA[param2] + beadB[param2]) ** 3)
    )

    param3_12 = weighted_mean(beadA, beadB, param3, weighting_parameters=[param2])
    tmp3 = np.sqrt((beadA[param3] ** 3 - 1) * (beadB[param3] ** 3 - 1)) / (
        param3_12 ** 3 - 1
    )

    return tmp1 * tmp2 * tmp3


def multipole(
    beadA, beadB, parameter, temperature=None, mode="curve fit", scaled=False
):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    square_well Berthelot geometric mean: c = np.sqrt(a[0]*b[0]) * np.sqrt(a[1]**3 * b[1]**3) / ((a[1] + b[1])/2)**3 
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b1.
    mode : str, Optional, default='curve fit'
        Dictates the mode by which the parameters are fit. By default the Mie parameters are fit to the multipole with the keyword "curve fit". Alternatively, the keyword "analytical" indicates that the energy parameter is explicitly calculated from the definite integral.
    scaled : bool, Optional, default=False
        Dictates whether the shape factor is used to scale the Mie potential 

    Returns
    -------
    output : dict
        Mixed interaction parameter

    """

    try:
        import mapsci as mr
    except Exception:
        raise ImportError(
            "Multipole combining rules require 'mapsci' package, which is currently unavailable. Install it from: https://github.com/jaclark5/mapsci"
        )

    if scaled in [True, "True", "true", "yes", "Yes"]:
        shape_factor_scale = True
    else:
        shape_factor_scale = False

    if not isinstance(temperature, str) and temperature != None:
        tmp = {"beadA": beadA.copy(), "beadB": beadB.copy()}
        for key, value in tmp.items():
            tmp[key]["sigma"] = value["sigma"] * 10  # convert from nm to angstroms

        if mode == "curve fit":
            dict_cross, _ = mr.extended_combining_rules_fitting(
                tmp, temperature, shape_factor_scale=shape_factor_scale
            )
        elif mode == "analytical":
            dict_cross, _ = mr.extended_combining_rules_analytical(
                tmp, temperature, shape_factor_scale=shape_factor_scale
            )
        else:
            raise ValueError(
                "Multipole mixing rule must be either 'curve fit' or 'analytical'."
            )
        output = dict_cross["beadA"]["beadB"]
    else:
        logger.warning("Temperature is None, using geometric mean.")
        output = {parameter: geometric_mean(beadA, beadB, parameter)}

    return output
