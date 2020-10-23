"""
    This file contains the interface needed to create further objects to represent experimental data. 

    Using this template all future data types will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# __________________ EOS Interface _________________
class ExpDataTemplate(ABC):

    """
    Interface needed to create further objects to represent experimental data.

     Using this template all future data types will be easily exchanged.
    """
    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        
        if "eos_obj" in data_dict:
            self.eos = data_dict["eos_obj"]
        else:
            raise ValueError("An eos object should have been included")
        
        if "weights" in data_dict:
            self.weights = data_dict["weights"]
        else:
            self.weights = {}
        
        self.obj_opts = {}
        if "objective_method" in data_dict:
            self.obj_opts["method"] = data_dict["objective_method"]
        fitting_opts = ["nan_number", "nan_ratio"]
        for key in fitting_opts:
            if key in data_dict:
                self.obj_opts[key] = data_dict[key]
        logger.info("Objective function options: {}".format(self.obj_opts))
        
        self.npoints = np.nan
        
        # Add to thermo_dict
        self.thermodict = {"calculation_type": None}
        thermo_dict_keys = ["mpObj", "density_dict", "calculation_type"]
        for key in thermo_dict_keys:
            if key in data_dict:
                self.thermodict[key] = data_dict[key]
    
    @abstractmethod
    def objective(self, eos):
        """
        Float representing objective function of from comparing predictions to experimental data.
        """
        pass
    
    def __str__(self):
    
        string = "Data Set Object\nname: {}\ncalculation_type: {}\nNdatapts: {}".format(self.name, self.thermodict["calculation_type"], self.npoints)
        return string
