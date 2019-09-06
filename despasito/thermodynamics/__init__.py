"""
Thermodynamics

This package will take in an equation of state object, and any user defined variables for calculation. The calculation type will then be compared to those available in the thermo.py file and be executed.

"""

# Add imports here
from inspect import getmembers, isfunction
from . import calc_types

def thermo(eos, thermo_dict):
    """
    Use factory design pattern to search for matching calctype with those supported in this module.
    To add a new calculation type, add a function to thermo.py in the thermodynamcis module..

    Parameters
    ----------
        eos : obj
            Equation of state output that writes pressure, max density, and chemical potential
        thermo_dict : dict
            Other keywords passed to the function, depends on calculation type
                

    Returns
    -------
        Output file saved in current working directory
    """

    try:
        calctype = thermo_dict['calculation_type']
    except:
        raise Exception('No calculation type specified')

    # Extract available calculation types
    calc_list = [o[0] for o in getmembers(calc_types) if isfunction(o[1])]

    # Unpack inputs and check
    sys_dict, kwargs = {}, {}
    for key, value in thermo_dict.items():
        if key not in ['rhodict','output_file','calculation_type']:
            sys_dict[key] = value
        elif key != 'calculation_type':
            kwargs[key] = value

    try:
        func = getattr(calc_types, calctype)
    except:
        raise ImportError("The calculation type, '"+calctype+"', was not found\nThe following calculation types are supported: "+", ".join(calc_list))

    try:
        func(eos, sys_dict, **kwargs)
    except:
        raise TypeError("The calculation type, '"+calctype+"', failed")

