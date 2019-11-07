"""
    This file contains the interface needed to create further equation of state (EOS) objects. All folders in this directory refer back to this interface. Using this template all future EOS will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

from abc import ABC, abstractmethod


# __________________ EOS Interface _________________
class EOStemplate(ABC):

    """
    Equation of state (EOS ) interface needed to create additional EOS objects. All classes in this directory refer back to this interface. Using this template all EOS objects are then easily exchanged.
    """

    @abstractmethod
    def P(self):
        """
        Output pressure value predicted by EOS.
        """
        pass

    @abstractmethod
    def fugacity_coefficient(self):
        """
        Output chemical potential predicted by EOS.
        """
        pass

    @abstractmethod
    def density_max(self):
        """
        Output maximum packing density predicted by EOS.
        """
        pass

    @abstractmethod
    def param_guess(self):
        """
        Output a guess for the given parameter type.
        """
        pass

    @abstractmethod
    def update_parameters(self):
        """
        Update a given parameter in EOS.
        """
        pass

