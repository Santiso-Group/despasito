# -- coding: utf8 --

r"""
    
    Parent SAFT EOS class

    An additional class with variant specifics imported from ``saft_type.py`` completes the EOS.
    
"""
import sys
import numpy as np
import logging

from despasito.equations_of_state import constants
import despasito.utils.general_toolbox as gtb
import despasito.equations_of_state.eos_toolbox as tb
from despasito.equations_of_state.interface import EosTemplate

from despasito.equations_of_state.saft import Aideal
from despasito.equations_of_state.saft import Aassoc

logger = logging.getLogger(__name__)


def saft_type(name):
    r"""
    Initialize EOS object for SAFT variant.
    
    All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    name : str
        Name of supported saft variant, the following are currently supported:

        - gamma_mie: :class:`~despasito.equations_of_state.saft.gamma_mie.SaftType`
        - gamma_sw: :class:`~despasito.equations_of_state.saft.gamma_sw.SaftType`

    Returns
    -------
    saft_source : obj
        SAFT variant class to be initiated in :class:`~despasito.equations_of_state.saft.saft.EosType`

    """
    if name == "gamma_mie":
        from despasito.equations_of_state.saft.gamma_mie import SaftType as saft_source
    elif name == "gamma_sw":
        from despasito.equations_of_state.saft.gamma_sw import SaftType as saft_source
    else:
        raise ValueError(
            "SAFT type, {}, is not supported. Be sure the class is added to the factory function 'saft_type'".format(
                name
            )
        )

    return saft_source


class EosType(EosTemplate):

    r"""
    Initialize EOS object for SAFT variant.
    
    All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - mass: Bead mass [kg/mol]
        - epsilonHB-\*-\*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K-\*-\*: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - rc-\*-\*: Optional, Cutoff distance for association sites. Asterisk represents two strings from sitenames.
        - rd-\*-\*: Optional, Site position. Asterisk represents two strings from sitenames.
        - Nk-\*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.
        - etc. depending on SAFT variant

    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - epsilonHB-\*-\*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K-\*-\*: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - rc-\*-\*: Optional, Cutoff distance for association sites. Asterisk represents two strings from sitenames.
        - rd-\*-\*: Optional, Site position. Asterisk represents two strings from sitenames.
        - etc. depending on SAFT variant

    combining_rules : dict, Optional, default=None
        Provided to overwrite functional form of mixing rules defined for parameters in specific SAFT variant. See appropriate class.
    saft_name : str, Optional, default="gamma_mie"
        Define the SAFT variant, options listed in :func:`~despasito.equations_of_state.saft.saft.saft_type`.
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component, as it corresponds to the `beads` array.
    Aideal_method : str, Optional
        Functional form of ideal gas contribution for Helmholtz energy. Default is defined in SAFT variant.
    reduction_ratio : float, Optional
        Reduced distance of the sites from the center of the sphere of interaction. This value is used when site position, rd_klab is None. See "func"`~despasito.equations_of_state.saft.Aassoc.calc_bonding_volume` for more details
    method_stat : obj
        EOS object containing the the method status of the available options. 
    kwargs
        Other keywords that are specific to the chosen SAFT variant

    Attributes
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. See **Parameters** section.
    cross_library : dict
        Library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. See **Parameters** section.
    number_of_components : int
        Number of components in mixture represented by given EOS object.
    parameter_types : list[str]
        This list of parameter names for the association site calculation, "epsilonHB","K", "rc", and/or "rd" as well as parameters for the specific SAFT variant. 
    parameter_bound_extreme : dict
        With each parameter names as an entry representing a list with the minimum and maximum feasible parameter value.

        - epsilonHB: [100.,5000.]
        - K: [1e-5,10000.]
        - rc: [0.1,10.0]
        - rd: [0.1,10.0]
        - etc., other parameters from SAFT variant

    saft_name : str, Optional, default="gamma_mie"
        Define the SAFT variant, options listed in :func:`~despasito.equations_of_state.saft.saft.saft_type`.
    saft_source : obj
        Object representing SAFT variant. This attribute can be used to access intermediate calculations.
    eos_dict : dict
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class. Others may be added once the SAFT variant object is initiated.

        - molecular_composition (numpy.ndarray) - :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component
        - residual_helmholtz_contributions (list[str]) - List of methods from the specified saft_source representing contributions to the Helmholtz energy that are functions of density, temperature, and composition 
        - Aideal_method (str) - "Abroglie" the default functional form of the ideal gas contribution of the Helmholtz energy
        - massi (list) - List of molecular weight for each group in `beads` array
        - flag_assoc (bool) - flag indicating whether there is an association site contribution to the Helmholtz energy
        - sitenames (list[str]) - List of sitenames for association site interactions. This array is extracted from `bead_library` entries
        - nk (numpy.ndarray) - A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
        - epsilonHB (numpy.ndarray) - Optional: Interaction energy between each bead and association site.
        - Kklab (numpy.ndarray) - Optional: Bonding volume between each association site
        - rc_klab, Optional: Cutoff distance for association sites
        - rd_klab, Optional: Association site position
        - reduction_ratio (float) - Reduced distance of the sites from the center of the sphere of interaction. This value is used when site position, rd_klab is None
    
    """

    def __init__(
        self, saft_name="gamma_mie", Aideal_method=None, combining_rules=None, **kwargs
    ):

        super().__init__(**kwargs)

        self.saft_name = saft_name
        saft_source = saft_type(saft_name)
        self.saft_source = saft_source(**kwargs)

        if "method_stat" in kwargs:
            self.method_stat = self.saft_source.method_stat

        if not hasattr(self, "eos_dict"):
            self.eos_dict = {}

        # Extract needed variables from saft type file (e.g. gamma_mie)
        self.parameter_types = ["epsilonHB", "K", "rc", "rd"]
        self.parameter_bound_extreme = {
            "epsilonHB": [100.0, 5000.0],
            "K": [1e-5, 10000.0],
            "rc": [0.1, 10.0],
            "rd": [0.1, 10.0],
        }

        saft_attributes = [
            "Aideal_method",
            "parameter_types",
            "parameter_bound_extreme",
            "residual_helmholtz_contributions",
        ]
        for key in saft_attributes:
            try:
                tmp = getattr(self.saft_source, key)
                if key == "parameter_bound_extreme":
                    self.parameter_bound_extreme.update(tmp)
                elif key == "parameter_types":
                    self.parameter_types = list(set(self.parameter_types) | set(tmp))
                else:
                    self.eos_dict[key] = tmp
            except Exception:
                raise ValueError(
                    "SAFT type, {}, is missing the variable {}.".format(saft_name, key)
                )

        for res in self.eos_dict["residual_helmholtz_contributions"]:
            setattr(self, res, getattr(self.saft_source, res))

        if Aideal_method != None:
            logger.info(
                "Switching Aideal method from {} to {}.".format(
                    self.eos_dict["Aideal_method"], Aideal_method
                )
            )
            self.eos_dict["Aideal_method"] = Aideal_method

        # Extract needed values from kwargs
        needed_attributes = ["bead_library", "molecular_composition", "beads"]
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError(
                    "The one of the following inputs is missing: {}".format(
                        ", ".join(tmp)
                    )
                )

            if key == "molecular_composition":
                self.eos_dict[key] = kwargs[key]
            else:
                setattr(self, key, kwargs[key])
        self.number_of_components = len(self.eos_dict["molecular_composition"])

        if "cross_library" not in kwargs:
            self.cross_library = {}
        else:
            self.cross_library = kwargs["cross_library"]
            self.cross_library = self.cross_library

        if "massi" not in self.eos_dict:
            self.eos_dict["massi"] = tb.calc_massi(
                self.eos_dict["molecular_composition"], self.bead_library, self.beads
            )

        if "reduction_ratio" in kwargs:
            self.eos_dict["reduction_ratio"] = kwargs["reduction_ratio"]

        # Initiate association site terms
        self.eos_dict["sitenames"], self.eos_dict["nk"], self.eos_dict[
            "flag_assoc"
        ] = Aassoc.initiate_assoc_matrices(
            self.beads, self.bead_library, self.eos_dict["molecular_composition"]
        )
        assoc_output = Aassoc.calc_assoc_matrices(
            self.beads,
            self.bead_library,
            self.eos_dict["molecular_composition"],
            sitenames=self.eos_dict["sitenames"],
            cross_library=self.cross_library,
            nk=self.eos_dict["nk"],
        )
        self.eos_dict.update(assoc_output)
        if np.size(np.where(self.eos_dict["epsilonHB"] != 0.0)) == 0:
            self.eos_dict["flag_assoc"] = False

        if combining_rules != None:
            logger.info("Accepted new mixing rule definitions")
            self.saft_source.combining_rules.update(combining_rules)
            self.parameter_refresh()

    def residual_helmholtz_energy(self, rho, T, xi):
        r"""
        Return a vector of residual Helmholtz energy.
        
        :math:`\frac{A^{res}}{N k_{B} T}`
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0

            
        Returns
        -------
        Ares : numpy.ndarray
            Residual Helmholtz energy for each density value given.
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        rho = self._check_density(rho)

        if any(np.array(xi) < 0.0):
            raise ValueError("Mole fractions cannot be less than zero.")

        Ares = np.zeros(len(rho))
        for res in self.eos_dict["residual_helmholtz_contributions"]:
            Ares += getattr(self.saft_source, res)(rho, T, xi)

        if self.eos_dict["flag_assoc"]:
            Ares += self.Aassoc(rho, T, xi)

        return Ares

    def helmholtz_energy(self, rho, T, xi):
        r"""
        Return a vector of Helmholtz energy.
        
        :math:`\frac{A}{N k_{B} T}`
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
            
        Returns
        -------
        A : numpy.ndarray
            Total Helmholtz energy for each density value given.
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        rho = self._check_density(rho)

        A = self.residual_helmholtz_energy(rho, T, xi) + self.Aideal(
            rho, T, xi, method=self.eos_dict["Aideal_method"]
        )

        return A

    def Aideal(self, rho, T, xi, method="Abroglie"):
        r"""
        Return a vector of ideal contribution of Helmholtz energy.
    
        :math:`\frac{A^{ideal}}{N k_{B} T}`
    
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        massi : numpy.ndarray
            Vector of component masses that correspond to the mole fractions in xi [kg/mol]
        method : str, Optional, default=Abroglie
            The function name of the method to calculate the ideal contribution of the Helmholtz energy. To add a new one, add a function to: despasito.equations_of_state.saft.Aideal.py
    
        Returns
        -------
        Aideal : numpy.ndarray
            Helmholtz energy of ideal gas for each density given.
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        rho = self._check_density(rho)

        return Aideal.Aideal_contribution(
            rho, T, xi, self.eos_dict["massi"], method=method
        )

    def Aassoc(self, rho, T, xi):
        r"""
        Return a vector of association site contribution of Helmholtz energy.
    
        :math:`\frac{A^{association}}{N k_{B} T}`
    
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
    
        Returns
        -------
        Aassoc : numpy.ndarray
            Helmholtz energy of ideal gas for each density given.
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        rho = self._check_density(rho)

        # compute F_klab
        Fklab = np.exp(self.eos_dict["epsilonHB"] / T) - 1.0
        if "rc_klab" in self.eos_dict:
            opts = {}
            keys = ["rd_klab", "reduction_ratio"]
            for key in keys:
                if key in self.eos_dict:
                    opts[key] = self.eos_dict[key]
            Kklab = self.saft_source.calc_Kijklab(T, self.eos_dict["rc_klab"], **opts)
            Ktype = "ijklab"
        else:
            Kklab = self.eos_dict["Kklab"]
            Ktype = "klab"

        gr_assoc = self.saft_source.calc_gr_assoc(rho, T, xi, Ktype=Ktype)

        # Compute Xika: with python with numba  {BottleNeck}
        indices = Aassoc.assoc_site_indices(
            self.eos_dict["nk"], self.eos_dict["molecular_composition"], xi=xi
        )
        Xika = Aassoc._calc_Xika_wrap(
            indices,
            rho,
            xi,
            self.eos_dict["molecular_composition"],
            self.eos_dict["nk"],
            Fklab,
            Kklab,
            gr_assoc,
            method_stat=self.method_stat
        )

        # Compute A_assoc
        Assoc_contribution = np.zeros(np.size(rho))
        for ind, (i, k, a) in enumerate(indices):
            if self.eos_dict["nk"][k, a] != 0.0:
                tmp = np.log(Xika[:, ind]) + ((1.0 - Xika[:, ind]) / 2.0)
                Assoc_contribution += (
                    xi[i]
                    * self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["nk"][k, a]
                    * tmp
                )

        return Assoc_contribution

    def pressure(self, rho, T, xi, step_size=1e-6):
        """
        Compute pressure given system information.
       
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
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        # derivative of Aideal_broglie here wrt to rho is 1/rho
        rho = self._check_density(rho)
        P_tmp = gtb.central_difference(
            rho, self.helmholtz_energy, args=(T, xi), step_size=step_size
        )
        pressure = P_tmp * T * constants.R * rho ** 2

        return pressure

    def fugacity_coefficient(self, P, rho, xi, T, dy=1e-5, log_method=True):

        """
        Compute fugacity coefficient.
      
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
        log_method : bool, Optional, default=False
            Choose to use a log transform in central difference method. This allows easier calculations for very small numbers.
    
        Returns
        -------
        fugacity_coefficient : numpy.ndarray
            Array of fugacity coefficient values for each component
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        rho = self._check_density(rho)
        logZ = np.log(P / (rho * T * constants.R))
        Ares = self.residual_helmholtz_energy(rho, T, xi)
        dAresdrho = tb.partial_density_central_difference(
            xi, rho, T, self.residual_helmholtz_energy, step_size=dy, log_method=True
        )

        phi = np.exp(Ares + rho * dAresdrho - logZ)

        return phi

    def density_max(self, xi, T, maxpack=0.65):

        """
        Estimate the maximum density based on the hard sphere packing fraction.
        
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]
        maxpack : float, Optional, default=0.65
            Maximum packing fraction
        
        Returns
        -------
        max_density : float
            Maximum molar density [mol/m^3]
        """
        if len(xi) != self.number_of_components:
            raise ValueError(
                "Number of components in mole fraction list, {}, doesn't match self.number_of_components, {}".format(
                    len(xi), self.number_of_components
                )
            )

        max_density = self.saft_source.density_max(xi, T, maxpack=maxpack)

        return max_density

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

        # Remove association site names
        param_name = param_name.split("-")[0]
        bounds_new = super().check_bounds(parameter, param_name, bounds)

        return bounds_new

    def update_parameter(self, param_name, bead_names, param_value):
        r"""
        Update a single parameter value during parameter fitting process.

        To refresh those parameters that are dependent on to bead_library or cross_library, use method "parameter refresh".
        
        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        param_value : float
            Value of parameter
        """

        parameter_list = param_name.split("-")
        if len(parameter_list) > 1 and len(parameter_list[1:]) != 2:
            raise ValueError(
                "Sitenames should be two different sites in the list: {}. You gave: {}".format(
                    self.eos_dict["sitenames"], ", ".join(parameter_list[1:])
                )
            )

        super().update_parameter(param_name, bead_names, param_value)

    def parameter_refresh(self):
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on bead_library and cross_library attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        self.saft_source.parameter_refresh(self.bead_library, self.cross_library)

        # Update Association site matrices
        if self.eos_dict["flag_assoc"]:
            assoc_output = Aassoc.calc_assoc_matrices(
                self.beads,
                self.bead_library,
                self.eos_dict["molecular_composition"],
                sitenames=self.eos_dict["sitenames"],
                cross_library=self.cross_library,
                nk=self.eos_dict["nk"],
            )
            self.eos_dict.update(assoc_output)

    def _check_density(self, rho):
        r"""
        This function checks the attributes of the density array
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        """

        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)
        if len(np.shape(rho)) == 2:
            rho = rho[0]

        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
            raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.0):
            raise ValueError("Density values cannot be negative.")

        return rho

    def __str__(self):

        string = "EOS: SAFT-{}, Beads: {},\nMasses: {} kg/mol\nSitenames: {}".format(
            self.saft_name,
            self.beads,
            self.eos_dict["massi"],
            self.eos_dict["sitenames"],
        )
        return string
