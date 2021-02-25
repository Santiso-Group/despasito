""" Interface needed to create further objects to represent experimental data.
"""

# All folders in this directory refer back to this interface
import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExpDataTemplate(ABC):
    r"""
    Interface needed to create further objects to represent experimental data.

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of TLVE temperature dependent liquid vapor equilibria

        * calculation_type (str) - Optional, default=*to be set*
        * eos_obj (obj) - Equation of state object
        * weights (dict) - A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * density_opts (dict) - Optional, default={}, Dictionary of options used in calculating pressure vs. mole fraction curves.
        * Allowed property keys and associated values
        * kwargs for :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`

    Attributes
    ----------
    name : str
        Data type, in this case TLVE
    Eos : obj
        Equation of state object
    weights : dict, Optional, default: {"some_property": 1.0 ...}
        Dictionary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    obj_opts : dict
        Keywords to compute the objective function with :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`.
    npoints : int
        Number of sets of system conditions this object computes
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
    
        - calculation_type (str) default=*to be set*
        - density_opts (dict) default={}
  
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = "To be set"

        if "eos_obj" in data_dict:
            self.Eos = data_dict["eos_obj"]
            del data_dict["eos_obj"]
        else:
            raise ValueError("An Eos object should have been included")

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
        thermo_dict_keys = ["MultiprocessingObject", "density_opts", "calculation_type"]
        for key in thermo_dict_keys:
            if key in data_dict:
                self.thermodict[key] = data_dict[key]
                del data_dict[key]

    def update_parameters(self, fit_bead, param_names, param_values):
        r"""
        Update a single parameter value during parameter fitting process.

        To refresh those parameters that are dependent on to bead_library or cross_library, use method "parameter refresh".
        
        Parameters
        ----------
        fit_bead : str
            Name of bead being fit
        param_names : list
            Parameters to be fit. See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        param_values : list
            Value of parameter
            
        """

        for i, param in enumerate(param_names):
            bead_names = [fit_bead]
            fit_parameter_names_list = param.split("_")
            param = fit_parameter_names_list[0]
            if len(fit_parameter_names_list) > 1:
                bead_names.append(fit_parameter_names_list[1])

            if len(fit_parameter_names_list) == 1:
                self.Eos.update_parameter(
                    fit_parameter_names_list[0], [fit_bead], param_values[i]
                )
            elif len(fit_parameter_names_list) == 2:
                self.Eos.update_parameter(
                    fit_parameter_names_list[0],
                    [fit_bead, fit_parameter_names_list[1]],
                    param_values[i],
                )
            else:
                raise ValueError(
                    "Parameters for only one bead are allowed to be fit. Multiple underscores in a parameter name suggest more than one bead type in your fit parameter name, {}".format(
                        param
                    )
                )

        if hasattr(self.Eos, "parameter_refresh"):
            self.Eos.parameter_refresh()

    @abstractmethod
    def objective(self, Eos):
        """ Float representing objective function of from comparing predictions to experimental data.
        """
        pass

    def __str__(self):

        string = "Data Set Object\nName: {}\nCalculation_type: {}\nNumber of Points: {}".format(
            self.name, self.thermodict["calculation_type"], self.npoints
        )
        return string
