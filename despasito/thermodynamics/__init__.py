"""
Thermodynamics

This package will take in an equation of state object, and any user defined variables for calculation. The calculation type will then be compared to those available in the thermo.py file and be executed.

"""

# Add imports here
from inspect import getmembers, isfunction
#import logging

from . import calc_types

#logger = logging.getLogger(__name__)

def thermo(eos, calculation_type=None, **thermo_dict):
    """
    Use factory design pattern to search for matching calculation_type with those supported in this module.
    
    To add a new calculation type, add a function to thermo.py in the thermodynamics module..

    Parameters
    ----------
        eos : obj
            Equation of state output that writes pressure, max density, and chemical potential
        calculation_type : str
            Calculation type supported in despasito.thermodynamics.calc_type
        thermo_dict : dict
            Other keywords passed to the function, depends on calculation type
                

    Returns
    -------
        output_dict : dict
            Output of dictionary containing given and calculated values
    """

    if calculation_type == None:
        raise ValueError('No calculation type specified')

    # Extract available calculation types
    calc_list = [o[0] for o in getmembers(calc_types) if isfunction(o[1])]

    # Unpack inputs and check
    try:
        func = getattr(calc_types, calculation_type)

    except:
        raise ImportError("The calculation type, '{}', was not found\nThe following calculation types are supported: {}".format(calculation_type,", ".join(calc_list)))

    output_dict = func(eos, **thermo_dict)
    #try:
    #    output_dict = func(eos, **thermo_dict)
    #except:
    #    raise TypeError("The calculation type, '{}', failed".format(calculation_type))

    return output_dict

