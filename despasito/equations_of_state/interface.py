"""
    This file contains the interface needed to create further equation of state (eos) objects. All folders in this directory refer back to this interface. Using this template all future eos will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

from abc import ABC, abstractmethod


# __________________ EOS Interface _________________
class EOStemplate(ABC):

    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Attributes
    ----------
    quote : str
        Compiled string including quote and optional attribution
    """

    @abstractmethod
    def P(self):
        pass

    @abstractmethod
    def chemicalpotential(self):
        pass

    @abstractmethod
    def density_max(self):
        pass
