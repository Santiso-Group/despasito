# -- coding: utf8 --
r"""
    EOS object for SAFT-:math:`\gamma`-SW
    
    Equations referenced in this code are from Lymperiadis, A. et. al, J. Chem. Phys. 127, 234903 (2007)
    
"""

import numpy as np
import logging
import os
import sys

import despasito.equations_of_state.eos_toolbox as tb
from despasito.equations_of_state import constants
import despasito.equations_of_state.saft.saft_toolbox as stb
from despasito.equations_of_state.saft import Aassoc

logger = logging.getLogger(__name__)

from despasito.equations_of_state import method_stat

if not method_stat.cython and not method_stat.numba:
    pass
elif method_stat.cython:
    logger.warning("saft.gamma_sw does not use cython.")
elif method_stat.numba:
    logger.warning("saft.gamma_sw does not use numba.")

ckl_coef = np.array(
    [
        [2.25855, -1.50349, 0.249434],
        [-0.669270, 1.40049, -0.827739],
        [10.1576, -15.0427, 5.30827],
    ]
)


class SaftType:

    r"""
    Object of SAFT variant

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters
    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.

    Attributes
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. See **Parameters** section.
    cross_library : dict
        Library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. See **Parameters** section.
    Aideal_method : str
        The default functional form of the ideal gas contribution of the Helmholtz energy
    residual_helmholtz_contributions : list[str]
        List of methods from the specified saft_source representing contributions to the Helmholtz energy that are functions of density, temperature, and composition.
    parameter_types : list[str]
        This list of parameter names for the specific SAFT variant. 
    parameter_bound_extreme : dict
        With each parameter name as an entry representing a list with the minimum and maximum feasible parameter value.
    combining_rules : dict
        Contains functional form and additional information for calculating cross interaction parameters that are not found in `cross_library`. Function must be one of those contained in :mod:`~despasito.equations_of_state.combining_rule_types`.
    eos_dict : dict
        Dictionary of parameters and specific settings 

        - molecular_composition (numpy.ndarray) - :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.

    ncomp : int
        Number of components in the system
    nbeads : int
        Number of beads in system that are shared among components

    """

    def __init__(self, **kwargs):

        ####### The Following lines are **MANDATORY** but can be edited as needed

        # Standard attributes of the EosTemplate are 'density_max' and 'parameter_refresh', while all saft versions require 'calc_gr_assoc' for calculating association sites

        # This keyword is used by the saft.py class and subsequently the `Aideal_contribution` function. If a user desired a different default method of calculating the ideal contribution, add a new function to Aideal.py
        self.Aideal_method = "Abroglie"

        # This list of strings are the bead types that the parameter update method can expect.
        # The bead library should use these same parameter names.
        # Parameter names should not contain '-' or '_'
        self.parameter_types = ["example"]

        # This dictionary will contain the feasible bounds of the parameters listed in 'parameter_types'. Bounds are necessary for each type.
        self.parameter_bound_extreme = {"example": [0, 1]}

        # When calculating the cross-interaction term between two beads, the parameter name and the combining rule type should be listed here. The combining rule keyword can be any that are supported in saft_toolbox.combining_rules. This attribute must exist.
        self.combining_rules = {"example": {"function": "mean"}}

        # This list contains the Helmholtz energy contributions contained in this class below. The class in saft.py will add these as it's own attributes to calculate the total Helmholtz energy.
        self.residual_helmholtz_contributions = ["Amonomer", "Achain"]
        # Note that these strings must represent methods for the Helmholtz contribution below and are added to Aideal and Aassoc in the main SAFT class. If a function is to be optimized with Cython or Numba, an extensions library can be added to the 'compiled_modules' directory. A python version with the same function names should also be present as well. The import if-structure at the beginning of this module is then used to import the desired form, see gamma_mie.py for an example.
        # When deciding whether to include an additional function as a class method, or if it should simply be imported from a library. Think about whether accessing that function would be nice when handling an EOS object in a python script. For instance, the 'reduced_density', 'effective_packing_fraction', 'Ahard_sphere', 'Afirst_order', and other methods in gamma_mie.py would be nice to have as attributes.

        # Now we start processing the given variables. The following three attributes are always needs for the saft.py class. If other inputs are needed for the specific SAFT type at hand, feel free to add them to this list.
        if not hasattr(self, "eos_dict"):
            self.eos_dict = {}

        needed_attributes = ["molecular_composition", "beads", "bead_library"]
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError(
                    "The one of the following inputs is missing: {}".format(
                        ", ".join(tmp)
                    )
                )
            elif key == "molecular_composition":
                if "molecular_composition" not in self.eos_dict:
                    self.eos_dict[key] = kwargs[key]
            elif not hasattr(self, key):
                setattr(self, key, kwargs[key])

        # Check bead_library to be sure all parameters are present. If one is missing that has a standard default then that is added
        self._parameter_defaults = {
            "epsilon": None,
            "lambdar": None,
            "lambdaa": None,
            "sigma": None,
            "Sk": 1.0,
            "Vks": 1.0,
        }
        self.bead_library = tb.check_bead_parameters(
            self.bead_library, self._parameter_defaults
        )

        if "cross_library" not in kwargs:
            self.cross_library = {}
        else:
            self.cross_library = kwargs["cross_library"]

        ###### The following lines are *OPTIONAL* and are completely for internal use for this specific SAFT type and aren't mandatory

        # Initialize composition attribute. This is for composition dependent properties. By recording this, we can avoid recalculating those parameters unnecessarily. In saft-gamma_mie we also have temperature dependent parameters and so self.T is included.
        if not hasattr(self, "xi"):
            self.xi = np.nan

        # These are initialized for loops used in some of the methods below. Depending on how you choose to break things up, these might not be needed.
        if not hasattr(self, "nbeads") or not hasattr(self, "ncomp"):
            self.ncomp, self.nbeads = np.shape(self.eos_dict["molecular_composition"])

        # Initiate cross interaction terms, as mentioned above, some combining rules use a particular combination of parameters and these are handled here.
        output = tb.cross_interaction_from_dict(
            self.beads,
            self.bead_library,
            self.combining_rules,
            cross_library=self.cross_library,
        )
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.eos_dict["epsilon_kl"] = output["epsilon"]
        self.eos_dict["lambda_kl"] = output["lambda"]

        # This optional keyword can be passed in an input file as "eos_num_rings". If your chosen version of SAFT needs special keyword passed that don't fall under a category above. Make a note in the doc string to pass it as "eos_somekeyword"
        if "num_rings" in kwargs:
            self.eos_dict["num_rings"] = kwargs["num_rings"]
            logger.info(
                "Accepted component ring structure: {}".format(kwargs["num_rings"])
            )
        else:
            self.eos_dict["num_rings"] = np.zeros(
                len(self.eos_dict["molecular_composition"])
            )

    def Amonomer(self, rho, T, xi):
        r"""

        ** Example of Helmholtz contribution to put in residual_helmholtz_contributions **

        Outputs the monomer contribution of the Helmholtz energy :math:`A^{mono.}/Nk_{b}T`.
    
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
        Amonomer : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        if np.all(rho > self.density_max(xi, T)):
            raise ValueError(
                "Density values should not all be greater than {}, or calc_Amono will fail in log calculation.".format(
                    self.density_max(xi, T)
                )
            )

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zetax = self.reduced_density(rho, xi)[:, 3]

        Amonomer = (
            self.Ahard_sphere(rho, T, xi)
            + self.Afirst_order(rho, T, xi, zetax=zetax)
            + self.Asecond_order(rho, T, xi, zetax=zetax)
        )

        return Amonomer

    def density_max(self, xi, T, maxpack=0.65):

        """
        ** Mandatory **

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

        self._check_composition_dependent_parameters(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        max_density = (
            maxpack
            * 6.0
            / (
                self.eos_dict["Cmol2seg"]
                * np.pi
                * np.sum(self.eos_dict["xskl"] * (self.eos_dict["sigma_kl"] ** 3))
            )
            / constants.molecule_per_nm3
        )

        return max_density

    def calc_gr_assoc(self, rho, T, xi, Ktype="ijklab"):
        r"""
        ** Mandatory **

        Reference fluid pair correlation function used in calculating association sites
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        Ktype : str, Optional, default='ijklab'
            Indicates which radial distribution function to return. The only option is 'ijklab': The bonding volume was calculated from self.calc_Kijklab, return gHS_dij)
    
        Returns
        -------
        gr : numpy.ndarray
            A temperature-density polynomial correlation of the association integral for a Lennard-Jones monomer. This matrix is (len(rho) x Ncomp x Ncomp)
        """

        rho = self._check_density(rho)
        gSW = self.calc_gSW(rho, T, xi)

        return gSW

    def calc_Kijklab(self, T, rc_klab, rd_klab=None, reduction_ratio=0.25):
        r"""
        ** Mandatory **

        Calculation of association site bonding volume, dependent on molecule in addition to group

        Lymperiadis Fluid Phase Equilibria 274 (2008) 85–104
        
        Parameters
        ----------
        T : float
            Temperature of the system [K], Note used in this version of saft, but included to allow saft.py to be general
        rc_klab : numpy.ndarray
            This matrix of cutoff distances for association sites for each site type in each group type
        rd_klab : numpy.ndarray, Optional, default=None
            Position of association site in each group (nbead, nbead, nsite, nsite)
        reduction_ratio : float, Optional, default=0.25
            Reduced distance of the sites from the center of the sphere of interaction. This value is used when site position, rd_klab is None
    
        Returns
        -------
        Kijklab : numpy.ndarray
            Matrix of binding volumes
        """

        dij_bar = np.zeros((self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                dij_bar[i, j] = np.mean(
                    [self.eos_dict["sigma_ij"][i], self.eos_dict["sigma_ij"][j]]
                )

        Kijklab = Aassoc.calc_bonding_volume(
            rc_klab, dij_bar, rd_klab=rd_klab, reduction_ratio=reduction_ratio
        )

        return Kijklab

    def parameter_refresh(self, bead_library, cross_library):
        r""" 
        ** Mandatory **

        To refresh dependent parameters
        
        Those parameters that are dependent on bead_library and cross_library attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.

        Attributes
        ----------
        alpha : np.array
            van der Waals attractive parameter for square-well segments, equal to :math:`\alpha_{k,l}/k_B`.
        eos_dict : dict
            The following entries are updated:

            - epsilon_kl (numpy.ndarray) - Matrix of well depths for groups (k,l)
            - sigma_kl (numpy.ndarray) - Matrix of bead diameters (k,l)
            - lambda_kl (numpy.ndarray) - Matrix of range of potential well depth (k,l)
            - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
            - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.

        """

        self.bead_library.update(bead_library)
        self.cross_library.update(cross_library)

        # Update Non bonded matrices
        output = tb.cross_interaction_from_dict(
            self.beads,
            self.bead_library,
            self.combining_rules,
            cross_library=self.cross_library,
        )
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.eos_dict["epsilon_kl"] = output["epsilon"]
        self.eos_dict["lambda_kl"] = output["lambda"]
        self.calc_component_averaged_properties()

        if not np.any(np.isnan(self.xi)):
            self.eos_dict["Cmol2seg"], self.eos_dict[
                "xskl"
            ] = stb.calc_composition_dependent_variables(
                xi,
                self.eos_dict["molecular_composition"],
                self.bead_library,
                self.beads,
            )
        self.alphakl = (
            2.0
            * np.pi
            / 3.0
            * self.eos_dict["epsilon_kl"]
            * self.eos_dict["sigma_kl"] ** 3
            * (self.eos_dict["lambda_kl"] ** 3 - 1.0)
        )

    def _check_density(self, rho):
        r"""
        ** Mandatory **
        
        This function checks that the density array is in the correct format for further calculations.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]

        Returns
        -------
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
            raise ValueError("No value of density was given")
        elif any(rho < 0.0):
            raise ValueError("Density values cannot be negative.")

        return rho
