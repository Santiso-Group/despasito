"""
Thermodynamics

This package will take in an equation of state object, and any user defined variables for calculation. The calculation type will then be compared to those available in the thermo.py file and be executed.

"""

# Add imports here
from inspect import getmembers, isfunction
from . import calc_types

def thermo(calctype, eos, thermo_dict):
    """
    Use factory design pattern to search for matching calctype with those supported in this module.
    To add a new calculation type, add a function to thermo.py in the thermodynamcis module..

    Parameters
    ----------
        calctype [str]: Input should be a function in the thermo.py file.
        eos      [obj]: Equation of state output that writes pressure, max density, and chemical potential
        kwargs   []: other keywords passed to the function
                

    Returns
    -------
        Output file saved in current working directory
    """

    # Extract available calculation types
    calc_list = [o[0] for o in getmembers(calc_types) if isfunction(o[1])]

    # Unpack inputs and check
    sys_dict, kwargs = {}, {}
    for key, value in thermo_dict.items():
        if key not in ['rhodict','output_file']:
            sys_dict[key] = value
        else:
            kwargs[key] = value

    # Try to run calculation
    if calctype == "xiTphase":
        calctype = "phase_xiT"

    try:
        func = getattr(calc_types, calctype)
        func(eos, sys_dict, **kwargs)
    except:
        raise ImportError("The calculation type, '"+calctype+"', was not found\nThe following calculation types are supported: "+", ".join(calc_list))

