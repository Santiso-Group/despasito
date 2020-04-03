"""

Create an EOS class from options taken from factory design pattern.

"""

# Add imports here
from importlib import import_module
import logging

class jit_stat:
    disable_jit = True

class cython_stat:
    disable_cython = True

logger = logging.getLogger(__name__)

def eos(**kwargs):
    """
    Interface between the user and our library of equations of state (EOS).

    Input the name of a desired EOS and a factory design pattern with a dictionary will search available classes to allow easy implementation of new EOS.

    Parameters
    ----------
        kwargs : dict, Optional
            A dictionary of inputs for the desired EOS. See specific EOS documentation for required inputs.

            - eos : str - Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie). Note that the name of the class is EOSfamily_EOSname.
                
    Returns
    -------
        instance : obj
            An instance of the defined EOS class to be used in thermodynamic computations.
    """


    if "eos" not in kwargs:
        eos_type = "saft.gamma_mie"
        logger.info("Trying default EOS, {}".format(eos_type))
    else:
        eos_type = kwargs["eos"]  
        del kwargs["eos"]
        logger.info("Trying user defined EOS, {}".format(eos_type))

    if 'jit' not in kwargs:
        jit_stat.disable_jit = True
    else:
        jit_stat.disable_jit = not kwargs['jit']

    if 'cython' not in kwargs:
        cython_stat.disable_cython = True
    else:
        cython_stat.disable_cython = not kwargs['cython']

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
            "Based on your input, '{}', we expect the class, {}, in a module, {}, found in the package, {}, which indicates the EOS family.".format(eos_type, class_name, eos, eos_fam))
    logger.info("Created {} eos object".format(eos_type))

    return instance

