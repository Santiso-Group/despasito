
import numpy as np
import logging

from . import tmp_mixingrules as mr

logger = logging.getLogger(__name__)

def mean( beadA, beadB, parameter):
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
    return (beadA[parameter] + beadB[parameter])/2

def geometric_mean( beadA, beadB, parameter):
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

def volumetric_geometric_mean( beadA, beadB, parameter, weighting_parameters=[]):
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
    tmp2 = np.sqrt((beadA[param2] ** 3) * (beadB[param2] ** 3)) * 8 / ((beadA[param2] + beadB[param2]) ** 3)
    return tmp1*tmp2

def weighted_mean( beadA, beadB, parameter, weighting_parameters=[]):
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
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b
1.
    Returns
    -------
    parameter12 : float
        Mixed interaction parameter
    """

    param2 = weighting_parameters[0]
    parameter12 = (beadA[parameter]*beadA[param2] + beadB[parameter]*beadB[param2])/(beadA[param2]+beadB[param2])

    return parameter12

def mie_exponent( beadA, beadB, parameter):
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
    square_well berthelot geometric mean: c = np.sqrt(a[0]*b[0]) * np.sqrt(a[1]**3 * b[1]**3) / ((a[1] + b[1])/2)**3 * np.sqrt(()*())/(a[])
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b
1.
    Returns
    -------
    parameter12 : float
        Mixed interaction parameter
    """
    param2, param3 = weighting_parameters[0], weighting_parameters[1]

    tmp1 = np.sqrt(beadA[parameter] * beadB[parameter])
    tmp2 = np.sqrt((beadA[param2] ** 3) * (beadB[param2] ** 3)) * 8 / ((beadA[param2] + beadB[param2]) ** 3)

    param3_12 = weighted_mean( beadA, beadB, param3, weighting_parameters=[param2])
    tmp3 = np.sqrt((beadA[param3]**3-1)*(beadB[param3]**3-1))/(param3_12**3-1)

    return tmp1*tmp2*tmp3

def multipole(beadA, beadB, parameter, temperature=None, additional_outputs=[]):
    r"""
    Calculates cross interaction parameter according to the calculation method provided.
    square_well berthelot geometric mean: c = np.sqrt(a[0]*b[0]) * np.sqrt(a[1]**3 * b[1]**3) / ((a[1] + b[1])/2)**3 * np.sqrt(()*())/(a[])
    
    Parameters
    ----------
    beadA : dict
        Dictionary of parameters used to describe a bead
    beadB : dict
        Dictionary of parameters used to describe a bead
    parameter : str
        Name of parameter for which a mixed value is needed
    weighting_parameters : list[str], Optional, default=[]
        Given parameter name is a0 and b0, while weighting_parameters should be of length 1 to represent the name for a1 and b
1.
    Returns
    -------
    output : dict
        Mixed interaction parameter
    """

    if temperature is not None:
        tmp = {"beadA": beadA, "beadB": beadB}
        dict_cross, _ = mr.extended_mixing_rules_fitting(tmp, temperature)
        output = dict_cross["beadA"]["beadB"]
    else:
        output = {parameter: geometric_mean( beadA, beadB, parameter)}

    return output
