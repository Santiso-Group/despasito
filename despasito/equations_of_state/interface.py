"""
    despasito
    DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output
    
    This file contains the interface needed to create further equation of state (eos) objects. All folders in this directory refer back to this interface. Using this template all future eos will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

from abc import ABC, abstractmethod


# __________________ EOS Interface _________________
class EOStemplate(ABC):
    @abstractmethod
    def P(self):
        pass

    @abstractmethod
    def chemicalpotential(self):
        pass

    @abstractmethod
    def density_max(self):
        pass
