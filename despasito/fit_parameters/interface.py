"""
    This file contains the interface needed to create further objects to represent experimental data. 

    Using this template all future data types will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

from abc import ABC, abstractmethod


# __________________ EOS Interface _________________
class ExpDataTemplate(ABC):

    """
    Interface needed to create further objects to represent experimental data.

     Using this template all future data types will be easily exchanged.
    """

    @abstractmethod
    def objective(self, eos):
        """
        Float representing objective function of from comparing predictions to experimental data.
        """
        pass

