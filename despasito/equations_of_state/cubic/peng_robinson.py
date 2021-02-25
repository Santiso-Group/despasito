# -- coding: utf8 --

"""
    EOS object for Peng-Robinson
    
"""

import numpy as np
import logging

from despasito.equations_of_state import constants
from despasito.equations_of_state.interface import EosTemplate
import despasito.utils.general_toolbox as gtb

# logger = logging.getLogger(__name__)


class EosType(EosTemplate):
    r"""
    EOS object for the Peng-Robinson EOS. 

    All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    beads : list[str]
        List of unique component names
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - Tc: :math:`T_{C}`, Critical temperature [K]
        - Pc: :math:`P_{C}`, Critical pressure [Pa]
        - omega: :math:`\omega`, Acentric factor 

    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - kij: :math:`k_{ij}`, binary interaction parameter
        
    Attributes
    ----------
    beads : list[str]
        List of component names
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. See description under *Parameters*
    cross_library : dict
        Library of bead cross interaction parameters. See description under *Parameters*
    parameter_types : list[str]
        List of parameter names used Peng-Robinson EOS: ["ai", "bi", "kij", "Tc", "Pc", "omega"]. Used in parameter fitting. 
    parameter_bound_extreme : dict
        With each parameter names as an entry representing a list with the minimum and maximum feasible parameter value. For the Peng-Robinson EOS:

        - ai: [0., 50.]
        - bi: [0., 1e-3]
        - kij: [-1.,1.]
        - omega: [0,1]
        - Tc: [0, 1000] [K]
        - Pc: [1, 1e+8] [Pa]

    number_of_components : int
        Number of components in mixture represented by given EOS object.
    T : float, default=numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.parameter_types = ["ai", "bi", "kij", "Tc", "Pc", "omega"]
        self.parameter_bound_extreme = {
            "ai": [0.0, 50.0],
            "bi": [0.0, 1e-3],
            "kij": [-1.0, 1.0],
            "omega": [0, 1],
            "Tc": [0, 1000],
            "Pc": [1, 1e8],
        }

        # Self interaction parameters
        self.beads = kwargs["beads"]
        self.bead_library = kwargs["bead_library"]
        self.number_of_components = len(self.beads)

        self._test_kappa = [False for _ in self.beads]
        self._test_critical = [False for _ in self.beads]
        self._test_parameters = [False for _ in self.beads]
        for i, bead in enumerate(self.beads):
            if (
                "omega" in self.bead_library[bead]
                and "kappa" not in self.bead_library[bead]
            ):
                self._test_kappa[i] = True

            self._test_critical[i] = (
                "Tc" in self.bead_library[bead] and "Pc" in self.bead_library[bead]
            )
            self._test_parameters[i] = (
                "ai" in self.bead_library[bead] and "bi" in self.bead_library[bead]
            )

            if not self._test_critical[i] and not self._test_parameters[i]:
                raise ValueError(
                    "Either 'Tc' or 'Pc' was not provided for component: {}".format(
                        bead
                    )
                )

        # Cross interaction parameters
        if "cross_library" in kwargs:
            self.cross_library = kwargs["cross_library"]
        else:
            self.cross_library = {}

        self.eos_dict = {
            "ai": np.zeros(self.number_of_components),
            "bi": np.zeros(self.number_of_components),
            "alpha": np.zeros(self.number_of_components),
            "aij": np.nan,
            "bij": np.nan,
            "kij": np.zeros((self.number_of_components, self.number_of_components)),
        }

        # Initialize temperature attribute
        self.T = None
        self.parameter_refresh()

    def _calc_temp_dependent_parameters(self, T):
        """
        Compute ai and alpha given temperature
       
        Parameters
        ----------
        T : float
            Temperature of the system [K]

        Attributes
        ----------
        ai : numpy.ndarray
            Peng-Robinson parameter a [m^6/mol^2]
        alpha : numpy.ndarray
            Peng-Robinson parameter b [m^3/mol]
        """
        if T != self.T:
            self.T = T

        for i, bead in enumerate(self.beads):
            if "kappa" in self.bead_library[bead]:
                self.eos_dict["alpha"][i] = (
                    1
                    + self.bead_library[bead]["kappa"]
                    * (1 - np.sqrt(T / self.bead_library[bead]["Tc"]))
                ) ** 2
            else:
                self.eos_dict["alpha"][i] = 1.0

    def _calc_mixed_parameters(self, xi, T):

        """
        Compute mixing aij and bij given composition
       
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]

        Attributes
        ----------
        aij : float
            Peng-Robinson parameter a [m^6/mol^2]
        bij : float
            Peng-Robinson parameter b [m^3/mol]
        """

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        aij = 0
        index = range(len(xi))
        for i in index:
            for j in index:
                aij += (
                    xi[i]
                    * xi[j]
                    * np.sqrt(
                        self.eos_dict["ai"][i]
                        * self.eos_dict["alpha"][i]
                        * self.eos_dict["ai"][j]
                        * self.eos_dict["alpha"][j]
                    )
                    * (1.0 - self.eos_dict["kij"][i][j])
                )

        self.eos_dict["aij"] = aij
        self.eos_dict["bij"] = np.sum(xi * self.eos_dict["bi"])

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

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi, T)

        if not gtb.isiterable(rho):
            rho = np.array([rho])
        elif not isinstance(rho, np.ndarray):
            rho = np.array(rho)

        P = constants.R * self.T * rho / (
            1 - self.eos_dict["bij"] * rho
        ) - rho ** 2 * self.eos_dict["aij"] / (
            (1 + self.eos_dict["bij"] * rho)
            + rho * self.eos_dict["bij"] * (1 - self.eos_dict["bij"] * rho)
        )

        return P

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

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi, T)

        tmp_RT = constants.R * T

        Z = P / (tmp_RT * rho)
        Ai = self.eos_dict["ai"] * self.eos_dict["alpha"] * P / tmp_RT ** 2
        Bi = self.eos_dict["bi"] * P / tmp_RT
        B = self.eos_dict["bij"] * P / tmp_RT
        A = self.eos_dict["aij"] * P / tmp_RT ** 2

        sqrt2 = np.sqrt(2.0)
        tmp1 = (
            A
            / (2.0 * sqrt2 * B)
            * np.log((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B))
        )
        tmp3 = Bi * (Z - 1) / B - np.log(Z - B)
        tmp2 = np.zeros(len(xi))

        index = range(len(xi))
        for i in index:
            Aij = np.zeros(len(xi))
            for j in index:
                Aij[j] = np.sqrt(Ai[i] * Ai[j]) * (1.0 - self.eos_dict["kij"][i][j])
            tmp2[i] = Bi[i] / B - 2 * np.sum(xi * Aij) / A
        phi = np.exp(tmp1 * tmp2 + tmp3)

        return phi

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

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi, T)

        max_density = maxpack / self.eos_dict["bij"]

        return max_density

    def update_parameter(self, param_name, bead_names, param_value):
        r"""
        Update a single parameter value during parameter fitting process.

        To refresh those parameters that are dependent on to bead_library or cross_library, use method "parameter refresh".
        
        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. kij_CO2).
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        param_value : float
            Value of parameter
        """

        if (
            param_name in ["ai", "bi"]
            and self._test_critical[self.beads.index(bead_names[0])]
        ):
            raise ValueError(
                "Bead, {}, initialized with critical properties, not ai and bi".format(
                    bead_names[0]
                )
            )
        super().update_parameter(param_name, bead_names, param_value)

    def parameter_refresh(self):
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on bead_library and cross_library attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        for i, bead in enumerate(self.beads):
            if (
                "omega" in self.bead_library[bead]
                and "kappa" not in self.bead_library[bead]
            ):
                self.bead_library[bead]["kappa"] = (
                    0.37464
                    + 1.54226 * self.bead_library[bead]["omega"]
                    - 0.26992 * self.bead_library[bead]["omega"] ** 2
                )

            if self._test_critical[i] and not self._test_parameters[i]:
                self.bead_library[bead]["ai"] = (
                    0.45723553
                    * (constants.R * self.bead_library[bead]["Tc"]) ** 2
                    / self.bead_library[bead]["Pc"]
                )
                self.bead_library[bead]["bi"] = 0.07779607 * (
                    constants.R
                    * self.bead_library[bead]["Tc"]
                    / self.bead_library[bead]["Pc"]
                )

            parameters = ["ai", "bi"]
            for key in parameters:
                self.eos_dict[key][i] = self.bead_library[bead][key]

            parameters = ["kij"]
            for key in parameters:
                for j, bead2 in enumerate(self.beads):
                    if (
                        bead in self.cross_library
                        and bead2 in self.cross_library[bead]
                        and key in self.cross_library[bead][bead2]
                    ):
                        tmp = self.cross_library[bead][bead2][key]
                        self.eos_dict[key][i][j] = tmp
                        self.eos_dict[key][j][i] = tmp
                    elif (
                        bead2 in self.cross_library
                        and bead in self.cross_library[bead2]
                        and key in self.cross_library[bead2][bead]
                    ):
                        tmp = self.cross_library[bead2][bead][key]
                        self.eos_dict[key][j][i] = tmp
                        self.eos_dict[key][i][j] = tmp

        if self.T != None:
            self._calc_temp_dependent_parameters(self.T)

    def __str__(self):

        string = "Beads:" + str(self.beads) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string
