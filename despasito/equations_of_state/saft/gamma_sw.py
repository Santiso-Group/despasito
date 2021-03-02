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

ckl_coef = np.array(
    [
        [2.25855, -1.50349, 0.249434],
        [-0.669270, 1.40049, -0.827739],
        [10.1576, -15.0427, 5.30827],
    ]
)


class SaftType:

    r"""
    Object of SAFT-𝛾-SW (for square well potential)
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    
        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter, contact distance [nm]
        - lambda: :math:`\lambda_{k,k}`, Range of the attractive interaction of well depth, epsilon
        - Sk: Optional, default=1, Shape factor, reflects the proportion which a given segment contributes to the total free energy
        - Vks: Optional, default=1, Number of segments in this molecular group

    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.
        
        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter, well depth, scaled by Boltzmann Constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter, contact distance [nm]
        - lambda: :math:`\lambda_{k,l}`, Range of the attractive interaction of well depth, epsilon

    num_rings : list
        Number of rings in each molecule. This will impact the chain contribution to the Helmholtz energy.
        
    Attributes
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. See **Parameters** section.
    cross_library : dict
        Library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. See **Parameters** section.
    Aideal_method : str
        "Abroglie" the default functional form of the ideal gas contribution of the Helmholtz energy
    residual_helmholtz_contributions : list[str]
        List of methods from the specified saft_source representing contributions to the Helmholtz energy that are functions of density, temperature, and composition. For this variant, [`Amonomer`, `Achain`]
    parameter_types : list[str]
        This list of parameter names, "epsilon", "lambda", "sigma", and/or "Sk" as well as parameters for the specific SAFT variant. 
    parameter_bound_extreme : dict
        With each parameter name as an entry representing a list with the minimum and maximum feasible parameter value.

        - epsilon: [10.,1000.]
        - lambda: [1.0,10.]
        - sigma: [0.1,10.0]
        - Sk: [0.1,1.0]

    combining_rules : dict
        Contains functional form and additional information for calculating cross interaction parameters that are not found in `cross_library`. Function must be one of those contained in :mod:`~despasito.equations_of_state.combining_rule_types`. The default values are:

        - sigma: {"function": "mean"}
        - lambda: {"function": "weighted_mean","weighting_parameters": ["sigma"]}
        - epsilon: {"function": "square_well_berthelot","weighting_parameters": ["sigma", "lambda"]}

    eos_dict : dict
        Dictionary of parameters and specific settings 

        - molecular_composition (numpy.ndarray) - :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
        - num_rings (list) - Number of rings in each molecule. This will impact the chain contribution to the Helmholtz energy.
        - Sk (numpy.ndarray) - Shape factor, reflects the proportion which a given segment contributes to the total free energy. Length of `beads` array.
        - Vks (numpy.ndarray) - Number of segments in this molecular group. Length of `beads` array.
        - epsilon_kl (numpy.ndarray) - Matrix of well depths for groups (k,l)
        - sigma_kl (numpy.ndarray) - Matrix of bead diameters (k,l)
        - lambda_kl (numpy.ndarray) - Matrix of range of potential well depth (k,l)
        - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.
        - epsilon_ij (numpy.ndarray) - Matrix of average molecular well depths (k,l)
        - sigma_ij (numpy.ndarray) - Matrix of average molecular diameter (k,l)
        - lambda_ij (numpy.ndarray) - Matrix of average molecular range of potential well depth (k,l)
        - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
        - alphakl (np.array) - van der Waals attractive parameter for square-well segments, equal to :math:`\alpha_{k,l}/k_B`.
        
    ncomp : int
        Number of components in the system
    nbeads : int
        Number of beads in system that are shared among components
    xi : numpy.ndarray
        Mole fraction of each molecule in mixture. Default initialization is np.nan
    """

    def __init__(self, **kwargs):

        if "method_stat" in kwargs:
            self.method_stat = kwargs["method_stat"]
            del kwargs["method_stat"]
            logger.info("This EOS doesn't use compiled modules for Amonomer and Achain")
        else:
            self.method_stat = None

        self.Aideal_method = "Abroglie"
        self.residual_helmholtz_contributions = ["Amonomer", "Achain"]
        self.parameter_types = ["epsilon", "lambda", "sigma", "Sk"]
        self._parameter_defaults = {
            "epsilon": None,
            "lambda": None,
            "sigma": None,
            "Sk": 1.0,
            "Vks": 1.0,
        }
        self.parameter_bound_extreme = {
            "epsilon": [10.0, 1000.0],
            "lambda": [1.0, 10.0],
            "sigma": [0.1, 10.0],
            "Sk": [0.1, 1.0],
        }
        self.combining_rules = {
            "sigma": {"function": "mean"},
            "lambda": {"function": "weighted_mean", "weighting_parameters": ["sigma"]},
            "epsilon": {
                "function": "square_well_berthelot",
                "weighting_parameters": ["sigma", "lambda"],
            },
        }  # Note in this EOS object, the mixing rules for the group parameters are also used for their corresponding molecular averaged parameters.

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

        self.bead_library = tb.check_bead_parameters(
            self.bead_library, self._parameter_defaults
        )

        if "cross_library" not in kwargs:
            self.cross_library = {}
        else:
            self.cross_library = kwargs["cross_library"]

        if "Vks" not in self.eos_dict:
            self.eos_dict["Vks"] = tb.extract_property(
                "Vks", self.bead_library, self.beads, default=1.0
            )
        if "Sk" not in self.eos_dict:
            self.eos_dict["Sk"] = tb.extract_property(
                "Sk", self.bead_library, self.beads, default=1.0
            )

        # Initialize component attribute
        if not hasattr(self, "xi"):
            self.xi = np.nan
        if not hasattr(self, "nbeads") or not hasattr(self, "ncomp"):
            self.ncomp, self.nbeads = np.shape(self.eos_dict["molecular_composition"])

        # Initiate cross interaction terms
        output = tb.cross_interaction_from_dict(
            self.beads,
            self.bead_library,
            self.combining_rules,
            cross_library=self.cross_library,
        )
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.eos_dict["epsilon_kl"] = output["epsilon"]
        self.eos_dict["lambda_kl"] = output["lambda"]

        if "num_rings" in kwargs:
            self.eos_dict["num_rings"] = kwargs["num_rings"]
            logger.info(
                "Accepted component ring structure: {}".format(kwargs["num_rings"])
            )
        else:
            self.eos_dict["num_rings"] = np.zeros(
                len(self.eos_dict["molecular_composition"])
            )

        # Initiate average interaction terms
        self.calc_component_averaged_properties()
        self.alphakl = (
            2.0
            * np.pi
            / 3.0
            * self.eos_dict["epsilon_kl"]
            * self.eos_dict["sigma_kl"] ** 3
            * (self.eos_dict["lambda_kl"] ** 3 - 1.0)
        )

    def calc_component_averaged_properties(self):
        r"""
        Calculate component averaged properties specific to SAFT-𝛾-SW 
        
        Attributes
        ----------
        eos_dict : dict
            Dictionary of outputs, the following possibilities are calculated if all relevant beads have those properties.
    
            - epsilon_ij (numpy.ndarray) - Matrix of average molecular well depths (k,l)
            - sigma_ij (numpy.ndarray) - Matrix of average molecular diameter (k,l)
            - lambda_ij (numpy.ndarray) - Matrix of average molecular range of potential well depth (k,l)
    
        """

        ncomp, nbeads = np.shape(self.eos_dict["molecular_composition"])
        zki = np.zeros((ncomp, nbeads), float)
        zkinorm = np.zeros(ncomp, float)

        epsilonii = np.zeros(ncomp, float)
        sigmaii = np.zeros(ncomp, float)
        lambdaii = np.zeros(ncomp, float)

        # compute zki
        for i in range(ncomp):
            for k in range(nbeads):
                zki[i, k] = (
                    self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["Vks"][k]
                    * self.eos_dict["Sk"][k]
                )
                zkinorm[i] += zki[i, k]

        for i in range(ncomp):
            for k in range(nbeads):
                zki[i, k] = zki[i, k] / zkinorm[i]

        for i in range(ncomp):
            for k in range(nbeads):
                sigmaii[i] += zki[i, k] * self.eos_dict["sigma_kl"][k, k] ** 3
                for l in range(nbeads):

                    epsilonii[i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["epsilon_kl"][k, l]
                    )
                    lambdaii[i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["lambda_kl"][k, l]
                    )
            sigmaii[i] = sigmaii[i] ** (1.0 / 3.0)

        input_dict = {"sigma": sigmaii, "lambda": lambdaii, "epsilon": epsilonii}
        dummy_dict, dummy_labels = tb.construct_dummy_bead_library(input_dict)
        output_dict = tb.cross_interaction_from_dict(
            dummy_labels, dummy_dict, self.combining_rules
        )
        self.eos_dict["sigma_ij"] = output_dict["sigma"]
        self.eos_dict["lambda_ij"] = output_dict["lambda"]
        self.eos_dict["epsilon_ij"] = output_dict["epsilon"]

    def reduced_density(self, rho, xi):
        r"""
        Reduced density matrix where the segment number density is reduced by powers of the size parameter, sigma.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Returns
        -------
        zeta : numpy.ndarray
            Reduced density matrix of length 4 of varying degrees of dependence on sigma. Units: [molecules/nm^3, molecules/nm^2, molecules/nm, molecules]
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        rho2 = rho * constants.molecule_per_nm3 * self.eos_dict["Cmol2seg"]

        reduced_density = np.zeros((np.size(rho), 4))
        for m in range(4):
            reduced_density[:, m] = rho2 * (
                np.sum(
                    np.sqrt(np.diag(self.eos_dict["xskl"]))
                    * (np.diag(self.eos_dict["sigma_kl"]) ** m)
                )
                * (np.pi / 6.0)
            )

        return reduced_density

    def effective_packing_fraction(self, rho, xi, zetax=None, mode="normal"):
        r"""
        Effective packing fraction for SAFT-gamma with a square-wave potential
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default="normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        
        Returns
        -------
        zeta_eff : numpy.ndarray
            Effective packing fraction (len(rho), Nbeads, Nbeads)
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if mode == "normal":
            lambdakl = self.eos_dict["lambda_kl"]
        elif mode == "effective":
            lambdakl = self.eos_dict["lambda_ij"]
        lx = len(lambdakl)  # lx is nbeads for normal and ncomp for effective

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]

        zetax_pow = np.zeros((np.size(rho), 3))
        zetax_pow[:, 0] = zetax
        for i in range(1, 3):
            zetax_pow[:, i] = zetax_pow[:, i - 1] * zetax_pow[:, 0]

        zetakl = np.zeros((np.size(rho), lx, lx))
        for k in range(lx):
            for l in range(lx):
                if lambdakl[k, l] != 0.0:
                    cikl = np.dot(
                        ckl_coef,
                        np.array(
                            (1.0, lambdakl[k, l], lambdakl[k, l] ** 2),
                            dtype=ckl_coef.dtype,
                        ),
                    )
                    zetakl[:, k, l] = np.dot(zetax_pow, cikl)

        return zetakl

    def _dzetaeff_dzetax(self, rho, xi, zetax=None, mode="normal"):
        r"""
        Derivative of effective packing fraction with respect to the reduced density. Eq. 33
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default="normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        
        Returns
        -------
        dzetakl : numpy.ndarray
            Derivative of effective packing fraction (len(rho), Nbeads, Nbeads) with respect to the reduced density
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if mode == "normal":
            lambdakl = self.eos_dict["lambda_kl"]
        elif mode == "effective":
            lambdakl = self.eos_dict["lambda_ij"]
        lx = len(lambdakl)  # lx is nbeads for normal and ncomp for effective

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]

        zetax_pow = np.transpose(
            np.array([np.ones(len(rho)), 2 * zetax, 3 * zetax ** 2])
        )

        # check if you have more than 1 bead types
        dzetakl = np.zeros((np.size(rho), lx, lx))
        for k in range(lx):
            for l in range(lx):
                if lambdakl[k, l] != 0.0:
                    cikl = np.dot(
                        ckl_coef,
                        np.array(
                            (1.0, lambdakl[k, l], lambdakl[k, l] ** 2),
                            dtype=ckl_coef.dtype,
                        ),
                    )
                    dzetakl[:, k, l] = np.dot(zetax_pow, cikl)

        return dzetakl

    def Ahard_sphere(self, rho, T, xi):
        r"""
        Outputs hard sphere approximation of Helmholtz free energy, :math:`A^{HS}/Nk_{b}T`.
        
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
        Ahard_sphere : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zeta = self.reduced_density(rho, xi)

        tmp = 6.0 / (np.pi * rho * constants.molecule_per_nm3)
        tmp1 = np.log1p(-zeta[:, 3]) * (
            zeta[:, 2] ** 3 / (zeta[:, 3] ** 2) - zeta[:, 0]
        )
        tmp2 = 3.0 * zeta[:, 2] / (1 - zeta[:, 3]) * zeta[:, 1]
        tmp3 = zeta[:, 2] ** 3 / (zeta[:, 3] * ((1.0 - zeta[:, 3]) ** 2))
        AHS = tmp * (tmp1 + tmp2 + tmp3)

        return AHS

    def Afirst_order(self, rho, T, xi, zetax=None):
        r"""
        Outputs :math:`A^{1st order}/Nk_{b}T`. This is the first order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        
        Returns
        -------
        Afirst_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]

        g0HS = self.calc_g0HS(rho, xi, zetax=zetax)
        a1kl_tmp = np.tensordot(
            rho * constants.molecule_per_nm3, self.eos_dict["xskl"] * self.alphakl, 0
        )
        A1 = -(self.eos_dict["Cmol2seg"] ** 2 / T) * np.sum(
            a1kl_tmp * g0HS, axis=(1, 2)
        )  # Units of K

        return A1

    def Asecond_order(self, rho, T, xi, zetax=None, KHS=None):
        r"""
        Outputs :math:`A^{2nd order}/Nk_{b}T`. This is the second order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        KHS : numpy.ndarray, Optional, default=None
            (length of densities) isothermal compressibility of system with packing fraction zetax
        
        Returns
        -------
        Asecond_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]
        # Note that zetax = zeta3

        if KHS is None:
            KHS = stb.calc_KHS(zetax)

        dzetakl = self._dzetaeff_dzetax(rho, xi, zetax=zetax)
        zeta_eff = self.effective_packing_fraction(rho, xi, zetax=zetax)
        g0HS = self.calc_g0HS(rho, xi, zetax=zetax)

        rho2 = self.eos_dict["Cmol2seg"] * rho * constants.molecule_per_nm3

        tmp1 = KHS * rho2 / 2.0
        tmp2 = self.eos_dict["epsilon_kl"] * self.alphakl * self.eos_dict["xskl"]
        a2kl_tmp = np.tensordot(tmp1, tmp2, 0)

        a2 = a2kl_tmp * (
            g0HS
            + zetax[:, np.newaxis, np.newaxis]
            * dzetakl
            * (2.5 - zeta_eff)
            / (1 - zeta_eff) ** 4
        )

        # Lymperiadis 2007 has a disconnect where Eq. 24 != Eq. 30, as Eq. 24 is missing a minus sign. (Same in Lymperiadis 2008 for Eq. 32 and Eq. 38)
        A2 = -(self.eos_dict["Cmol2seg"] / (T ** 2)) * np.sum(a2, axis=(1, 2))

        return A2

    def Amonomer(self, rho, T, xi):
        r"""
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

    def calc_g0HS(self, rho, xi, zetax=None, mode="normal"):
        r"""
        The contact value of the pair correlation function of a hypothetical pure fluid of diameter sigmax evaluated at an effective packing fraction, zeta_eff.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default="normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective", where normal used bead interaction matrices, and effective uses component averaged parameters.
        
        Returns
        -------
        g0HS : numpy.ndarray
            The contact value of the pair correlation function of a hypothetical pure fluid
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]

        zeta_eff = self.effective_packing_fraction(rho, xi, mode=mode, zetax=zetax)

        g0HS = (1.0 - zeta_eff / 2.0) / (1.0 - zeta_eff) ** 3

        return g0HS

    def calc_gHS(self, rho, xi):
        r"""
        Hypothetical pair correlation function of a hypothetical pure fluid.

        This fluid is of diameter sigmax evaluated at contact and effective packing fraction zeta_eff.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Returns
        -------
        gHS : numpy.ndarray
            Hypothetical pair correlation function of a hypothetical pure fluid of diameter sigmax evaluated at contact and effective packing fraction zeta_eff.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zetam = self.reduced_density(rho, xi)

        tmp1 = 1.0 / (1.0 - zetam[:, 3])
        tmp2 = zetam[:, 2] / (1.0 - zetam[:, 3]) ** 2
        tmp3 = zetam[:, 2] ** 2 / (1.0 - zetam[:, 3]) ** 3

        gHS = np.zeros((np.size(rho), self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                tmp = (
                    self.eos_dict["sigma_ij"][i, i]
                    * self.eos_dict["sigma_ij"][j, j]
                    / (
                        self.eos_dict["sigma_ij"][i, i]
                        + self.eos_dict["sigma_ij"][j, j]
                    )
                )
                gHS[:, i, j] = tmp1 + 3 * tmp * tmp2 + 2 * tmp ** 2 * tmp3

        return gHS

    def calc_gSW(self, rho, T, xi, zetax=None):
        r"""
        Calculate the square-well pair correlation function at the effective contact distance and the actual packing fraction of the mixture.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        
        Returns
        -------
        gSW : numpy.ndarray
            Square-well pair correlation function at the effective contact distance and the actual packing fraction of the mixture.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)
        kT = T * constants.kb

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:, 3]

        g0HS = self.calc_g0HS(rho, xi, zetax=zetax, mode="effective")
        gHS = self.calc_gHS(rho, xi)
        zeta_eff = self.effective_packing_fraction(
            rho, xi, mode="effective", zetax=zetax
        )
        dg0HSdzetaeff = (2.5 - zeta_eff) / (1.0 - zeta_eff) ** 4

        ncomp = len(xi)
        dckl_coef = np.array(
            [[-1.50349, 0.249434], [1.40049, -0.827739], [-15.0427, 5.30827]]
        )
        zetax_pow = np.transpose(np.array([zetax, zetax ** 2, zetax ** 3]))
        dzetaijdlambda = np.zeros((np.size(rho), ncomp, ncomp))
        for i in range(ncomp):
            for j in range(ncomp):
                cikl = np.dot(
                    dckl_coef, np.array([1.0, (2 * self.eos_dict["lambda_ij"][i, j])])
                )
                dzetaijdlambda[:, i, j] = np.dot(zetax_pow, cikl)

        dzetaijdzetax = self._dzetaeff_dzetax(rho, xi, zetax=zetax, mode="effective")
        dzetaeff = (
            self.eos_dict["lambda_ij"][np.newaxis, :, :] / 3.0 * dzetaijdlambda
            - zetax[:, np.newaxis, np.newaxis] * dzetaijdzetax
        )

        gSW = gHS + self.eos_dict["epsilon_ij"][np.newaxis, :, :] / T * (
            g0HS
            + (self.eos_dict["lambda_ij"][np.newaxis, :, :] ** 3 - 1.0)
            * dg0HSdzetaeff
            * dzetaeff
        )

        return gSW

    def Achain(self, rho, T, xi):
        r"""
        Outputs chain contribution to the Helmholtz energy :math:`A^{chain}/Nk_{b}T`.
    
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
        Achain : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)

        gii = self.calc_gSW(rho, T, xi)

        Achain = 0.0
        for i in range(self.ncomp):
            beadsum = -1.0 + self.eos_dict["num_rings"][i]
            for k in range(self.nbeads):
                beadsum += (
                    self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["Vks"][k]
                    * self.eos_dict["Sk"][k]
                )
            Achain -= xi[i] * beadsum * np.log(gii[:, i, i])

        if np.any(np.isnan(Achain)):
            logger.error("Some Helmholtz values are NaN, check energy parameters.")

        return Achain

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

        self.eos_dict["Sk"] = tb.extract_property(
            "Sk", self.bead_library, self.beads, default=1.0
        )

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
                self.xi,
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

    def _check_composition_dependent_parameters(self, xi):
        r"""
        This function updates composition dependent variables
        
        Parameters
        ----------
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Attributes
        ----------
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        eos_dict : dict
            The following entries are updated:

            - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
            - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.

        """
        xi = np.array(xi)
        if not np.all(self.xi == xi):
            self.eos_dict["Cmol2seg"], self.eos_dict[
                "xskl"
            ] = stb.calc_composition_dependent_variables(
                xi,
                self.eos_dict["molecular_composition"],
                self.bead_library,
                self.beads,
            )
            self.xi = xi

    def __str__(self):

        string = "Beads: {}".format(self.beads)
        return string
