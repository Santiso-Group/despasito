"""
    Interface needed to create further equation of state (EOS) objects. 

    All folders in this directory refer back to this interface. Using this template all future EOS will be easily exchanged.
    
"""

# All folders in this directory refer back to this interface

import numpy as np
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# __________________ EOS Interface _________________
class EosTemplate(ABC):

    """
    Interface used in all EOS object options.

    By using this template, all EOS objects are then easily exchanged.
    """

    def __init__(self, beads, beadlibrary, **kwargs):


        self.parameter_types = None
        self.parameter_bound_extreme = None

        self.number_of_components = None
        for bead in beads:
            if bead not in beadlibrary:
                raise ValueError("The group, '{}', was not found in parameter library".format(bead))

        self.beads = None
        self.beadlibrary = None
        self.crosslibrary = None

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

    def param_guess(self, param_name, bead_names):
        """
        Output a guess for the given parameter type.
        """
        """
        Generate initial guesses for the parameters to be fit.

        Parameters
        ----------
        parameter : str
            Parameter to be fit. See EOS documentation for supported parameter names.
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.

        Returns
        -------
        param_initial_guess : numpy.ndarray, 
            An initial guess for parameter, it will be optimized throughout the process.
        """

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.beads)))

        if param_name not in self.parameter_types:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.parameter_types)))

        param_value = None
        # Self interaction parameter
        if len(bead_names) == 1:
            if bead_names[0] in self.beadlibrary:
                param_value = self.beadlibrary[bead_names[0]][param_name]
        # Cross interaction parameter
        elif len(bead_names) == 2:
            if bead_names[1] in self.crosslibrary and bead_names[0] in self.crosslibrary[bead_names[1]]:
                param_value = self.crosslibrary[bead_names[1]][bead_names[0]][param_name]
            elif bead_names[0] in self.crosslibrary and bead_names[1] in self.crosslibrary[bead_names[0]]:
                param_value = self.crosslibrary[bead_names[0]][bead_names[1]][param_name]

        if param_value == None:
            bounds = self.check_bounds(bead_names[0], param_name, np.empty(2))
            param_value = (bounds[1]-bounds[0])/2 + bounds[0]

        return param_value
    
    def check_bounds(self, parameter, param_name, bounds):
        """
        Generate initial guesses for the parameters to be fit.
        
        Parameters
        ----------
        parameter : str
            Parameter to be fit. See EOS documentation for supported parameter names.
        param_name : str
            Full parameter string to be fit. See EOS documentation for supported parameter names.
        bounds : list
            Upper and lower bound for given parameter type
        
        Returns
        -------
        bounds : list
            A screened and possibly corrected low and a high value for the parameter, param_name
        """

        fit_params_list = param_name.split("_")
        parameter = fit_params_list[0]

        bounds_new = np.zeros(2)
        # Non bonded parameters
        if (parameter in self.parameter_bound_extreme):

            if bounds[0] < self.parameter_bound_extreme[parameter][0]:
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],self.parameter_bound_extreme[parameter][0]))
                bounds_new[0] = self.parameter_bound_extreme[parameter][0]
            elif bounds[0] > self.parameter_bound_extreme[parameter][1]:
                logger.debug("Given {} lower boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],self.parameter_bound_extreme[parameter][0]))
                bounds_new[0] = self.parameter_bound_extreme[parameter][0]
            else:
                bounds_new[0] = bounds[0]

            if (bounds[1] > self.parameter_bound_extreme[parameter][1]):
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],self.parameter_bound_extreme[parameter][1]))
                bounds_new[1] = self.parameter_bound_extreme[parameter][1]
            elif (bounds[1] < self.parameter_bound_extreme[parameter][0]):
                logger.debug("Given {} upper boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],self.parameter_bound_extreme[parameter][1]))
                bounds_new[1] = self.parameter_bound_extreme[parameter][1]
            else:
                bounds_new[1] = bounds[1]

        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.parameter_types)))

        return bounds_new

    def update_parameter(self, param_name, bead_names, param_value):
        r"""
        Update a single parameter value during parameter fitting process.

        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        param_value : float
            Value of parameter
        """

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.beads)))

        if param_name in self.parameter_types:
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.beadlibrary:
                    self.beadlibrary[bead_names[0]][param_name] = param_value
                else:
                    self.beadlibrary[bead_names[0]] = {param_name: param_value}
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self.crosslibrary and bead_names[0] in self.crosslibrary[bead_names[1]]:
                    self.crosslibrary[bead_names[1]][bead_names[0]][param_name] = param_value
                elif bead_names[0] in self.crosslibrary:
                    if bead_names[1] in self.crosslibrary[bead_names[0]]:
                        self.crosslibrary[bead_names[0]][bead_names[1]][param_name] = param_value
                    else:
                        self.crosslibrary[bead_names[0]][bead_names[1]] = {param_name: param_value}
                else:
                    self.crosslibrary[bead_names[0]] = {bead_names[1]: {param_name: param_value}}

        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.parameter_types)))

