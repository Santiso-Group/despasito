"""

Create an EOS class from options taken from factory design pattern.

"""

# Add imports here
from importlib import import_module
import logging

class method_stat:

    def __init__(self, numba=False, cython=False, python=False):

        self.numba = numba
        self.cython = cython
        self.python = python

        if not any([numba, cython, python]):
            self.fortran = True
        else:
            self.fortran = False

    def __str__(self):

        string = "Compilation: numba {}, cython {}, python {}, fortran {}".format(self.numba, self.cython, self.python, self.fortran)

        return string

logger = logging.getLogger(__name__)


def initiate_eos(
    eos="saft.gamma_mie", numba=False, cython=False, python=False, **input_dict
):
    """
    Interface between the user and our library of equations of state (EOS).

    Input the name of a desired EOS and a factory design pattern with a dictionary will search available classes to allow easy implementation of new EOS.

    Parameters
    ----------
    eos : str, Optional, default="saft.gamma_mie"
        Name of EOS, see EOS Types in the documentation for additional options. Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).
    input_dict : dict, Optional
        A dictionary of inputs for the desired EOS. See specific EOS documentation for required inputs.
    numba : bool, Optional, default=False
        If True, numba Just-In-Time compilation is used.
    cython : bool, Optional, default=False
        If True, cython pre-compiled modules are used.
    python : bool, Optional, default=False
        If True, pure python is used for everything, note that if association sites are present in the SAFT EOS, this is detrimentally slow
                
    Returns
    -------
    instance : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    """

    input_dict["method_stat"] = method_stat(numba=numba, cython=cython, python=python)

    factory_families = [
        "saft"
    ]  # Eos families in this list have a general object with a factory to import relevant modules

    logger.info("Using EOS: {}".format(eos))

    try:
        eos_fam, eos_type = eos.split(".")
    except Exception:
        raise ValueError(
            "Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie)."
        )

    class_name = "EosType"
    try:
        if eos_fam in factory_families:
            eos_module = import_module(
                "." + eos_fam, package="despasito.equations_of_state." + eos_fam
            )
            input_dict["saft_name"] = eos_type

        else:
            eos_module = import_module(
                "." + eos_type, package="despasito.equations_of_state." + eos_fam
            )
        eos_class = getattr(eos_module, class_name)
    except AttributeError:
        raise ImportError(
            "Based on your input, '{}', we expect the class, {}, in a module, {}, found in the package, {}, which indicates the EOS family.".format(
                eos, class_name, eos_type, eos_fam
            )
        )
    instance = eos_class(**input_dict)

    logger.info("Created {} Eos object".format(eos))

    return instance
