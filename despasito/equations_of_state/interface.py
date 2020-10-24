"""
    Interface needed to create further equation of state (EOS) objects. 

    All folders in this directory refer back to this interface. Using this template all future EOS will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

import numpy as np

from abc import ABC, abstractmethod

# __________________ EOS Interface _________________
class EosTemplate(ABC):

    """
    Interface used in all EOS object options.

    By using this template, all EOS objects are then easily exchanged.
    """

    def __init__(self, beads, beadlibrary, numba=False, cython=False, python=False, **kwargs):

        self.number_of_components = np.nan
        for bead in beads:
            if bead not in beadlibrary:
                raise ValueError("The group, '{}', was not found in parameter library".format(bead))

    @abstractmethod
    def pressure(self, rho, T, xi):
        """
        Output pressure value predicted by EOS.
        """
        pass

    @abstractmethod
    def fugacity_coefficient(self, P, rho, xi, T):
        """
        Output chemical potential predicted by EOS.
        """
        pass

    @abstractmethod
    def density_max(self, xi, T):
        """
        Output maximum packing density predicted by EOS.
        """
        pass

    @abstractmethod
    def param_guess(self, param_name, bead_names):
        """
        Output a guess for the given parameter type.
        """
        pass
    
    @abstractmethod
    def check_bounds(self, fit_bead, param_name, bounds):
        """
        Check given boundaries and possibly correct for the given parameter type.
        """
        pass

    @abstractmethod
    def update_parameters(self, fit_bead, param_name, param_value):
        """
        Update a given parameter in EOS.
        """
        pass

