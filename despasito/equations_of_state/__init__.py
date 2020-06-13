"""

Create an EOS class from options taken from factory design pattern.

"""

# Add imports here
from importlib import import_module
import logging

logger = logging.getLogger(__name__)

def eos(**input_dict):
    """
    Interface between the user and our library of equations of state (EOS).

    Input the name of a desired EOS and a factory design pattern with a dictionary will search available classes to allow easy implementation of new EOS.

    Parameters
    ----------
        input_dict : dict, Optional
            A dictionary of inputs for the desired EOS. See specific EOS documentation for required inputs.

            - eos : str - Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie). Note that the name of the class is EOSfamily_EOSname.
                
    Returns
    -------
        instance : obj
            An instance of the defined EOS class to be used in thermodynamic computations.
    """

    factory_families = ["saft"] # eos families in this list have a general object with a factory to import relevent modules

    if "eos" not in input_dict:
        eos_type = "saft.gamma_mie"
        logger.info("Trying default EOS, {}".format(eos_type))
    else:
        eos_type = input_dict["eos"]  
        del input_dict["eos"]
        logger.info("Trying user defined EOS, {}".format(eos_type))

    try:
        eos_fam, eos = eos_type.split('.')
    except:
        raise Exception("Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).")

    class_name = "_".join([eos_fam, eos])
    try:
        if eos_fam in factory_families:
            eos_module = import_module('.' + eos_fam, package="despasito.equations_of_state." + eos_fam)
            eos_class = getattr(eos_module, eos_fam)
            input_dict['saft_name'] = eos
        
        else:
            eos_module = import_module('.' + eos, package="despasito.equations_of_state." + eos_fam)
            eos_class = getattr(eos_module, class_name)
    except (AttributeError):
        raise ImportError("Based on your input, '{}', we expect the class, {}, in a module, {}, found in the package, {}, which indicates the EOS family.".format(eos_type, class_name, eos, eos_fam))
    instance = eos_class(input_dict)

    logger.info("Created {} eos object".format(eos_type))

    return instance

