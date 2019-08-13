"""
despasito
DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output

This vision for this library is that this file will define an EOS class, and import sublibraries of each EOS family.

"""

# Add imports here
from importlib import import_module


def eos(eos_type, **kwargs):
    """
    Use factory design pattern with a dictionary to allow easy implementation of new equations of state (EOS).

    Parameters
    ----------
        eos_types$ [str]: Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie). Note that the name of the class is the same as the module name,
                

    Returns
    -------
        cls_instance [obj] : An instance of the defined EOS class to be used in later computations.
    """

    try:
        eos_fam, eos = eos_type.split('.')
    except:
        raise Exception("Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).")

    try:
        eos_module = import_module('.' + eos, package="despasito.lib_eos." + eos_fam)
        class_name = "_".join([eos_fam, eos])
        eos_class = getattr(eos_module, class_name)
        instance = eos_class(kwargs)
    except (AttributeError):
        raise ImportError(
            "Based on your input, '%s', we expect the class, %s, in a module, %s, found in the package, %s, which indicates the EOS family."
            % (eos_type, class_name, eos, eos_fam))

    return instance
