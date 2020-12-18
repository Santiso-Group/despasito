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
            del data_dict["eos_obj"]
        else:
            raise ValueError("An eos object should have been included")
        
        if "weights" in data_dict:
            self.weights = data_dict["weights"]
            del data_dict["weights"]
        else:
            self.weights = {}
        
        self.obj_opts = {}
        if "objective_method" in data_dict:
            self.obj_opts["method"] = data_dict["objective_method"]
            del data_dict["objective_method"]
        fitting_opts = ["nan_number", "nan_ratio"]
        for key in fitting_opts:
            if key in data_dict:
                self.obj_opts[key] = data_dict[key]
                del data_dict[key]
        logger.info("Objective function options: {}".format(self.obj_opts))
        
        self.npoints = np.nan
        
        # Add to thermo_dict
        self.thermodict = {"calculation_type": None}
        thermo_dict_keys = ["mpObj", "density_dict", "calculation_type"]
        for key in thermo_dict_keys:
            if key in data_dict:
                self.thermodict[key] = data_dict[key]
                del data_dict[key]

    def update_parameters(self, fit_bead, param_names, param_values):
        r"""
        Update a single parameter value during parameter fitting process.

        To refresh those parameters that are dependent on to beadlibrary or crosslibrary, use method "parameter refresh".
        
        Parameters
        ----------
        fit_bead : str
            Name of bead being fit
        param_names : list
            Parameters to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        param_values : list
            Value of parameter
        """

        for i, param in enumerate(param_names):
            bead_names = [fit_bead]
            fit_params_list = param.split("_")
            param = fit_params_list[0]
            if len(fit_params_list) > 1:
                bead_names.append(fit_params_list[1])

            if len(fit_params_list) == 1:
                self.eos.update_parameter(fit_params_list[0], [fit_bead], param_values[i])
            elif len(fit_params_list) == 2:
                self.eos.update_parameter(fit_params_list[0], [fit_bead, fit_params_list[1]], param_values[i])
            else:
                raise ValueError("Parameters for only one bead are allowed to be fit. Multiple underscores in a parameter name suggest more than one bead type in your fit parameter name, {}".format(param))

        if hasattr(self.eos, "parameter_refresh"):
            self.eos.parameter_refresh()
    
    @abstractmethod
    def objective(self, eos):
        """
        Float representing objective function of from comparing predictions to experimental data.
        """
        pass
    
    def __str__(self):
    
        string = "Data Set Object\nname: {}\ncalculation_type: {}\nNdatapts: {}".format(self.name, self.thermodict["calculation_type"], self.npoints)
        return string
