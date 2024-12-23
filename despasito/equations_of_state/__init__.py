"""

Create an EOS class from options taken from factory design pattern.

"""

# Add imports here
from importlib import import_module
import logging


class method_stat:

    def __init__(self, numba=True, cython=False, python=False):

        self.cython = cython
        self.python = python
        if self.cython or self.python:
            self.numba = False
        else:
            self.numba = numba

        if not any([numba, cython, python]):
            raise ValueError("Calculation type has not been specified.")

    def __str__(self):

        string = "Compilation: numba {}, cython {}, python {}".format(self.numba, self.cython, self.python)

        return string


logger = logging.getLogger(__name__)


def initiate_eos(eos="saft.gamma_mie", numba=True, cython=False, python=False, **kwargs):
    """
    Interface between the user and our library of equations of state (EOS).

    Input the name of a desired EOS and available classes are automatically searched
    to allow easy implementation of new EOS.

    Parameters
    ----------
    eos : str, Optional, default="saft.gamma_mie"
        Name of EOS, see :ref:`EOS-types` in the documentation for additional options.
        Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).
    numba : bool, Optional, default=False
        If True and available for chosen EOS, numba Just-In-Time compilation is used.
    cython : bool, Optional, default=False
        If True and available for chosen EOS, cython pre-compiled modules are used.
    python : bool, Optional, default=False
        If True and available for chosen EOS, pure python is used for everything, note
        that if association sites are present in the SAFT EOS, this is detrimentally
        slow
    kwargs
        Other keyword argument inputs for the desired EOS. See specific EOS
        documentation for required inputs.

    Returns
    -------
    instance : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    """

    kwargs["method_stat"] = method_stat(numba=numba, cython=cython, python=python)

    factory_families = ["saft"]  # Eos families in this list have a general object with a factory to import
    # relevant modules

    logger.info("Using EOS: {}".format(eos))

    try:
        eos_fam, eos_type = eos.split(".")
    except Exception:
        raise ValueError("Input should be in the form EOSfamily.EOSname (e.g. saft.gamme_mie).")

    class_name = "EosType"
    try:
        if eos_fam in factory_families:
            eos_module = import_module("." + eos_fam, package="despasito.equations_of_state." + eos_fam)
            kwargs["saft_name"] = eos_type

        else:
            eos_module = import_module("." + eos_type, package="despasito.equations_of_state." + eos_fam)
        eos_class = getattr(eos_module, class_name)
    except AttributeError:
        raise ImportError(
            "Based on your input, '{}', we expect the class, {}, in a module, {},"
            " found in the package, {}, which indicates the EOS family.".format(eos, class_name, eos_type, eos_fam)
        )
    instance = eos_class(**kwargs)

    logger.info("Created {} Eos object".format(eos))

    return instance
