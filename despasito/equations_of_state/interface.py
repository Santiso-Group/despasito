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

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters
    cross_library : dict, Optional
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. The remaining are estimated with mixing rules.
    kwargs
        Additional keywords from EOS object type

    Attributes
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters
    cross_library : dict
        Library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.
    parameter_types : list[str]
        This list of parameter names must be defined in an EOS object for parameter fitting
    parameter_bound_extreme : dict
        With each parameter names as an entry representing a list with the minimum and maximum feasible parameter value. 
    number_of_components : int
        Number of components in mixture represented by given EOS object.
        
    """

    def __init__(self, beads, bead_library, **kwargs):
        """ Initiation of EOS object with attributes needed by other modules.
        """

        self.parameter_types = None
        self.parameter_bound_extreme = None

        self.number_of_components = None
        for bead in beads:
            if bead not in bead_library:
                raise ValueError(
                    "The group, '{}', was not found in parameter library".format(bead)
                )

        self.beads = None
        self.bead_library = None
        self.cross_library = None

    @abstractmethod
    def pressure(self, rho, T, xi):
        """
        Compute pressure given system information
       
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : list[float]
            Mole fraction of each component
       
        Returns
        -------
        P : numpy.ndarray
            Array of pressure values [Pa] associated with each density and so equal in length
        """
        pass

    @abstractmethod
    def fugacity_coefficient(self, P, rho, xi, T):
        r"""
        Compute fugacity coefficient
      
        Parameters
        ----------
        P : float
            Pressure of the system [Pa]
        rho : float
            Molar density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : list[float]
            Mole fraction of each component
    
        Returns
        -------
        fugacity_coefficient : numpy.ndarray
            :math:`\mu_i`, Array of fugacity coefficient values for each component
        """
        pass

    @abstractmethod
    def density_max(self, xi, T, maxpack=0.9):
        """
        Estimate the maximum density based on the hard sphere packing fraction.
        
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]
        maxpack : float, Optional, default=0.9
            Maximum packing fraction
        
        Returns
        -------
        max_density : float
            Maximum molar density [mol/m^3]
        """
        pass

    def guess_parameters(self, param_name, bead_names):
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

        keys = ["beads", "parameter_types", "bead_library", "cross_library"]
        for key in keys:
            if getattr(self, key) == None:
                raise ValueError(
                    "EOS object attribute, {}, cannot be None. Ensure EOS object initiates this attribute".format(
                        key
                    )
                )

        if len(bead_names) > 2:
            raise ValueError(
                "The bead names {} were given, but only a maximum of 2 are permitted.".format(
                    ", ".join(bead_names)
                )
            )
        if not set(bead_names).issubset(self.beads):
            raise ValueError(
                "The bead names {} were given, but they are not in the allowed list: {}".format(
                    ", ".join(bead_names), ", ".join(self.beads)
                )
            )

        param_value = None
        # Self interaction parameter
        if len(bead_names) == 1:
            if (
                bead_names[0] in self.bead_library
                and param_name in self.bead_library[bead_names[0]]
            ):
                param_value = self.bead_library[bead_names[0]][param_name]
        # Cross interaction parameter
        elif len(bead_names) == 2:
            if (
                bead_names[1] in self.cross_library
                and bead_names[0] in self.cross_library[bead_names[1]]
            ):
                if param_name in self.cross_library[bead_names[1]][bead_names[0]]:
                    param_value = self.cross_library[bead_names[1]][bead_names[0]][
                        param_name
                    ]
            elif (
                bead_names[0] in self.cross_library
                and bead_names[1] in self.cross_library[bead_names[0]]
            ):
                if param_name in self.cross_library[bead_names[0]][bead_names[1]]:
                    param_value = self.cross_library[bead_names[0]][bead_names[1]][
                        param_name
                    ]

        if param_value == None:
            bounds = self.check_bounds(bead_names[0], param_name, np.empty(2))
            param_value = (bounds[1] - bounds[0]) / 2 + bounds[0]

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

        keys = ["parameter_types", "parameter_bound_extreme"]
        for key in keys:
            if getattr(self, key) == None:
                raise ValueError(
                    "EOS object attribute, {}, cannot be None. Ensure EOS object initiates this attribute".format(
                        key
                    )
                )

        fit_parameter_names_list = param_name.split("_")
        parameter = fit_parameter_names_list[0]

        bounds_new = np.zeros(2)
        # Non bonded parameters
        if parameter in self.parameter_bound_extreme:

            if bounds[0] < self.parameter_bound_extreme[parameter][0]:
                logger.debug(
                    "Given {} lower boundary, {}, is less than what is recommended by Eos object. Using value of {}.".format(
                        param_name,
                        bounds[0],
                        self.parameter_bound_extreme[parameter][0],
                    )
                )
                bounds_new[0] = self.parameter_bound_extreme[parameter][0]
            elif bounds[0] > self.parameter_bound_extreme[parameter][1]:
                logger.debug(
                    "Given {} lower boundary, {}, is greater than what is recommended by Eos object. Using value of {}.".format(
                        param_name,
                        bounds[0],
                        self.parameter_bound_extreme[parameter][0],
                    )
                )
                bounds_new[0] = self.parameter_bound_extreme[parameter][0]
            else:
                bounds_new[0] = bounds[0]

            if bounds[1] > self.parameter_bound_extreme[parameter][1]:
                logger.debug(
                    "Given {} upper boundary, {}, is greater than what is recommended by Eos object. Using value of {}.".format(
                        param_name,
                        bounds[1],
                        self.parameter_bound_extreme[parameter][1],
                    )
                )
                bounds_new[1] = self.parameter_bound_extreme[parameter][1]
            elif bounds[1] < self.parameter_bound_extreme[parameter][0]:
                logger.debug(
                    "Given {} upper boundary, {}, is less than what is recommended by Eos object. Using value of {}.".format(
                        param_name,
                        bounds[1],
                        self.parameter_bound_extreme[parameter][1],
                    )
                )
                bounds_new[1] = self.parameter_bound_extreme[parameter][1]
            else:
                bounds_new[1] = bounds[1]

        else:
            raise ValueError(
                "The parameter name {} is not found in the allowed parameter types: {}".format(
                    param_name, ", ".join(self.parameter_types)
                )
            )

        return bounds_new

    def update_parameter(self, param_name, bead_names, param_value):
        r"""
        Update a single parameter value during parameter fitting process.

        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        param_value : float
            Value of parameter

        """

        keys = ["beads", "parameter_types", "bead_library", "cross_library"]
        for key in keys:
            if getattr(self, key) == None:
                raise ValueError(
                    "EOS object attribute, {}, cannot be None. Ensure EOS object initiates this attribute".format(
                        key
                    )
                )

        if len(bead_names) > 2:
            raise ValueError(
                "The bead names {} were given, but only a maximum of 2 are permitted.".format(
                    ", ".join(bead_names)
                )
            )
        if not set(bead_names).issubset(self.beads):
            raise ValueError(
                "The bead names {} were given, but they are not in the allowed list: {}".format(
                    ", ".join(bead_names), ", ".join(self.beads)
                )
            )

        if not any([x in param_name for x in self.parameter_types]):
            raise ValueError(
                "The parameter name {} is not found in the allowed parameter types: {}".format(
                    param_name, ", ".join(self.parameter_types)
                )
            )

        # Self interaction parameter
        if len(bead_names) == 1:
            if bead_names[0] in self.bead_library:
                self.bead_library[bead_names[0]][param_name] = param_value
            else:
                self.bead_library[bead_names[0]] = {param_name: param_value}
        # Cross interaction parameter
        elif len(bead_names) == 2:
            if (
                bead_names[1] in self.cross_library
                and bead_names[0] in self.cross_library[bead_names[1]]
            ):
                self.cross_library[bead_names[1]][bead_names[0]][
                    param_name
                ] = param_value
            elif bead_names[0] in self.cross_library:
                if bead_names[1] in self.cross_library[bead_names[0]]:
                    self.cross_library[bead_names[0]][bead_names[1]][
                        param_name
                    ] = param_value
                else:
                    self.cross_library[bead_names[0]][bead_names[1]] = {
                        param_name: param_value
                    }
            else:
                self.cross_library[bead_names[0]] = {
                    bead_names[1]: {param_name: param_value}
                }
