"""
Thermodynamics

This package will take in an equation of state object, and any user defined variables for calculation. The calculation type will then be compared to those available in the thermo.py file and be executed.

"""

from inspect import getmembers, isfunction

# import logging

from . import calculation_types

# logger = logging.getLogger(__name__)


def thermo(Eos, calculation_type=None, **thermo_dict):
    """
    Use factory design pattern to search for matching calculation_type with those supported in this module.
    
    To add a new calculation type, add a new wrapper function to calculation_types.py.

    Parameters
    ----------
    Eos : obj
        Equation of state output that writes pressure, max density, and fugacity coefficient.
    calculation_type : str
        Calculation type supported in :mod:`~despasito.thermodynamics.calculation_types`
    thermo_dict : dict
        Other keywords passed to the function, depends on calculation type

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    if calculation_type == None:
        raise ValueError("No calculation type specified")

    # Extract available calculation types
    calc_list = [o[0] for o in getmembers(calculation_types) if isfunction(o[1])]

    # Unpack inputs and check
    try:
        func = getattr(calculation_types, calculation_type)

    except Exception:
        raise ImportError(
            "The calculation type, '{}', was not found\nThe following calculation types are supported: {}".format(
                calculation_type, ", ".join(calc_list)
            )
        )

    try:
        output_dict = func(Eos, **thermo_dict)
    except Exception:
        raise TypeError("The calculation type, '{}', failed".format(calculation_type))

    return output_dict
