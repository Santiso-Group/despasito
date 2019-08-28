"""

This vision for this library is that this file will define an EOS class, and import sub-libraries of each EOS family

"""

# Add imports here
from importlib import import_module


def eos(eos_type, **kwargs):
    """
    This function acts as an interface between the user and our library of equations of state (EOS). Input the name of a desired EOS and a factory design pattern with a dictionary will search available classes to allow easy implementation of new EOS.

    Parameters
    ----------
        eos_types : str
            Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie). Note that the name of the class is EOSfamily_EOSname.
        kwargs : dict, Optional
            A dictionary of inputs for the desired EOS. See specific EOS documentation for required inputs.
                
    Returns
    -------
        instance : obj
            An instance of the defined EOS class to be used in thermodynamic computations.
    """

    try:
        eos_fam, eos = eos_type.split('.')
    except:
        raise Exception("Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).")

    try:
        eos_module = import_module('.' + eos, package="despasito.equations_of_state." + eos_fam)
        class_name = "_".join([eos_fam, eos])
        eos_class = getattr(eos_module, class_name)
        instance = eos_class(kwargs)
    except (AttributeError):
        raise ImportError(
            "Based on your input, '%s', we expect the class, %s, in a module, %s, found in the package, %s, which indicates the EOS family."
            % (eos_type, class_name, eos, eos_fam))

    return instance

