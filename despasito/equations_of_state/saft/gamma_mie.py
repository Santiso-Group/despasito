# -- coding: utf8 --
r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import os
import sys

import despasito.equations_of_state.eos_toolbox as tb
from despasito.equations_of_state import constants
import despasito.equations_of_state.saft.saft_toolbox as stb
from despasito.equations_of_state.saft import Aassoc
from .compiled_modules.ext_gamma_mie_python import prefactor, calc_Iij

logger = logging.getLogger(__name__)

def _import_supporting_functions(method_stat=None):
    """ Import appropriate functions for compilation mode
    """

    if method_stat == None or method_stat.fortran or method_stat.python:
        import despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_python as cm

    elif method_stat.numba:
        import despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_numba as cm

    elif method_stat.cython:
        import despasito.equations_of_state.saft.compiled_modules.ext_gamma_mie_cython as cm

    else:
        raise ValueError("Unknown instructions for importing supportive functions of SAFT")

    return cm

class SaftType:

    r"""
    Object of SAFT-𝛾-Mie
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    
        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [nm]
        - mass: Bead mass [kg/mol]
        - lambdar: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - lambdaa: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Sk: Optional, default=1, Shape factor, reflects the proportion which a given segment contributes to the total free energy
        - Vks: Optional, default=1, Number of segments in this molecular group

    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [nm]
        - mass: Bead mass [kg/mol]
        - lambdar: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - lambdaa: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k

    num_rings : list
        Number of rings in each molecule. This will impact the chain contribution to the Helmholtz energy.
        
    Attributes
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters. See **Parameters** section.
    cross_library : dict, Optional, default={}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used. See **Parameters** section.
    Aideal_method : str
        "Abroglie" the default functional form of the ideal gas contribution of the Helmholtz energy
    residual_helmholtz_contributions : list[str]
        List of methods from the specified saft_source representing contributions to the Helmholtz energy that are functions of density, temperature, and composition. For this variant, [`Amonomer`, `Achain`]
    parameter_types : list[str]
        This list of parameter names, "epsilon", "lambdar", "lambdaa", "sigma", and/or "Sk" as well as parameters for the specific SAFT variant. 
    parameter_bound_extreme : dict
        With each parameter name as an entry representing a list with the minimum and maximum feasible parameter value.

        - epsilon: [100.,1000.]
        - lambdar: [6.0,100.]
        - lambdaa: [3.0,100.]
        - sigma: [0.1,10.0]
        - Sk: [0.1,1.0]
  
    combining_rules : dict
        Contains functional form and additional information for calculating cross interaction parameters that are not found in `cross_library`. Function must be one of those contained in :mod:`~despasito.equations_of_state.combining_rule_types`. The default values are:

        - sigma: {"function": "mean"}
        - lambdar: {"function": "mie_exponent"}
        - lambdar: {"function": "mie_exponent"}
        - epsilon: {"function": "volumetric_geometric_mean", "weighting_parameters": ["sigma"]}

    eos_dict : dict
        Dictionary of parameters and specific settings 

        - molecular_composition (numpy.ndarray) - :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
        - num_rings (list) - Number of rings in each molecule. This will impact the chain contribution to the Helmholtz energy.
        - Sk (numpy.ndarray) - Shape factor, reflects the proportion which a given segment contributes to the total free energy. Length of `beads` array.
        - Vks (numpy.ndarray) - Number of segments in this molecular group. Length of `beads` array.
        - Ckl (numpy.ndarray) - Matrix of Mie potential prefactors between beads  (l,k)
        - epsilonkl (numpy.ndarray) - Matrix of Mie potential well depths for groups (k,l)
        - sigmakl (numpy.ndarray) - Matrix of bead diameters (k,l)
        - lambdarkl (numpy.ndarray) - Matrix of repulsive Mie exponent for groups (k,l)
        - lambdaakl (numpy.ndarray) - Matrix of attractive Mie exponent for groups (k,l)
        - dkl (numpy.ndarray) - Matrix of hard sphere equivalent for each bead and interaction between them (l,k)
        - x0kl (numpy.ndarray) - Matrix of sigmakl/dkl, sigmakl is the Mie radius for groups (k,l)
        - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.
        - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
        - alphakl (np.array) - (Ngroup,Ngroup) "A dimensionless form of the integrated vdW energy of the Mie potential" eq. 33
        - epsilonii_avg (numpy.ndarray) - Matrix of molecule averaged well depths (i.j)
        - sigmaii_avg (numpy.ndarray) - Matrix of molecule averaged Mie diameter  (i.j)
        - lambdaaii_avg (numpy.ndarray) - Matrix of molecule averaged Mie potential attractive exponents  (i.j)
        - lambdarii_avg (numpy.ndarray) - Matrix of molecule averaged Mie potential attractive exponents (i.j)
        - dii_eff (numpy.ndarray) - Matrix of mole averaged hard sphere equivalent for each bead and interaction between them (i.j)
        - x0ii (numpy.ndarray) - Matrix of sigmaii_avg/dii_eff, sigmaii_avg is the average molecular Mie radius and dii_eff the average molecular hard sphere diameter


    ncomp : int
        Number of components in the system
    nbeads : int
        Number of beads in system that are shared among components
    xi : numpy.ndarray
        Mole fraction of each molecule in mixture. Default initialization is np.nan
    T : float
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, **kwargs):

        if "method_stat" in kwargs:
            self.method_stat = kwargs["method_stat"]
            del kwargs["method_stat"]
        else:
            self.method_stat = None

        self._cm = _import_supporting_functions(self.method_stat)

        self.Aideal_method = "Abroglie"
        self.parameter_types = ["epsilon", "sigma", "lambdar", "lambdaa", "Sk"]
        self._parameter_defaults = {
            "epsilon": None,
            "lambdar": None,
            "lambdaa": None,
            "sigma": None,
            "Sk": 1.0,
            "Vks": 1.0,
        }
        self.parameter_bound_extreme = {
            "epsilon": [100.0, 1000.0],
            "sigma": [0.1, 1.0],
            "lambdar": [6.0, 100.0],
            "lambdaa": [3.0, 100.0],
            "Sk": [0.1, 1.0],
        }
        self.residual_helmholtz_contributions = ["Amonomer", "Achain"]
        self.combining_rules = {
            "sigma": {"function": "mean"},
            "lambdar": {"function": "mie_exponent"},
            "lambdaa": {"function": "mie_exponent"},
            "epsilon": {
                "function": "volumetric_geometric_mean",
                "weighting_parameters": ["sigma"],
            },
        }

        self._mixing_temp_dependence = None

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

        # Initialize temperature attribute
        if not hasattr(self, "T"):
            self.T = np.nan
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
        self.eos_dict["sigmakl"] = output["sigma"]
        self.eos_dict["epsilonkl"] = output["epsilon"]
        self.eos_dict["lambdaakl"] = output["lambdaa"]
        self.eos_dict["lambdarkl"] = output["lambdar"]

        # compute alphakl eq. 33
        self.eos_dict["Ckl"] = prefactor(
            self.eos_dict["lambdarkl"], self.eos_dict["lambdaakl"]
        )
        self.eos_dict["alphakl"] = self.eos_dict["Ckl"] * (
            (1.0 / (self.eos_dict["lambdaakl"] - 3.0))
            - (1.0 / (self.eos_dict["lambdarkl"] - 3.0))
        )

        # Initiate average interaction terms
        self.calc_component_averaged_properties()

        if "num_rings" in kwargs:
            self.eos_dict["num_rings"] = kwargs["num_rings"]
            logger.info(
                "Accepted component ring structure: {}".format(kwargs["num_rings"])
            )
        else:
            self.eos_dict["num_rings"] = np.zeros(
                len(self.eos_dict["molecular_composition"])
            )

    def calc_component_averaged_properties(self):
        r"""
        Calculate component averaged properties specific to SAFT-𝛾-Mie        

        Attributes
        ----------
        output : dict
            Dictionary of outputs, the following possibilities are calculated if all relevant beads have those properties.
    
            - epsilonii_avg (numpy.ndarray) - Matrix of molecule averaged well depths
            - sigmaii_avg (numpy.ndarray) - Matrix of molecule averaged Mie diameter
            - lambdaaii_avg (numpy.ndarray) - Matrix of molecule averaged Mie potential attractive exponents
            - lambdarii_avg (numpy.ndarray) - Matrix of molecule averaged Mie potential attractive exponents
    
        """

        zki = np.zeros((self.ncomp, self.nbeads), float)
        zkinorm = np.zeros(self.ncomp, float)

        output = {}
        output["epsilonii_avg"] = np.zeros(self.ncomp, float)
        output["sigmaii_avg"] = np.zeros(self.ncomp, float)
        output["lambdarii_avg"] = np.zeros(self.ncomp, float)
        output["lambdaaii_avg"] = np.zeros(self.ncomp, float)

        # compute zki
        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = (
                    self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["Vks"][k]
                    * self.eos_dict["Sk"][k]
                )
                zkinorm[i] += zki[i, k]

        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = zki[i, k] / zkinorm[i]

        for i in range(self.ncomp):
            for k in range(self.nbeads):
                for l in range(self.nbeads):
                    output["sigmaii_avg"][i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["sigmakl"][k, l] ** 3
                    )
                    output["epsilonii_avg"][i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["epsilonkl"][k, l]
                    )
                    output["lambdarii_avg"][i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["lambdarkl"][k, l]
                    )
                    output["lambdaaii_avg"][i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["lambdaakl"][k, l]
                    )
            output["sigmaii_avg"][i] = output["sigmaii_avg"][i] ** (1 / 3.0)

        self.eos_dict.update(output)

    def Ahard_sphere(self, rho, T, xi):
        r"""
        Outputs monomer contribution to the Helmholtz energy, :math:`A^{HS}/Nk_{B}T`.
        
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
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        eta = np.zeros((np.size(rho), 4))
        for m in range(4):
            eta[:, m] = (
                rho
                * constants.molecule_per_nm3
                * self.eos_dict["Cmol2seg"]
                * (
                    np.sum(
                        np.sqrt(np.diag(self.eos_dict["xskl"]))
                        * (np.diag(self.eos_dict["dkl"]) ** m)
                    )
                    * (np.pi / 6.0)
                )
            )

        tmp = 6.0 / (np.pi * rho * constants.molecule_per_nm3)
        if self.ncomp == 1:
            tmp1 = 0
        else:
            tmp1 = np.log1p(-eta[:, 3]) * (
                eta[:, 2] ** 3 / (eta[:, 3] ** 2) - eta[:, 0]
            )
        tmp2 = 3.0 * eta[:, 2] / (1 - eta[:, 3]) * eta[:, 1]
        tmp3 = eta[:, 2] ** 3 / (eta[:, 3] * ((1.0 - eta[:, 3]) ** 2))

        AHS = tmp * (tmp1 + tmp2 + tmp3)

        return AHS

    def Afirst_order(self, rho, T, xi, zetax=None):
        r"""
        Outputs :math:`A^{1st order}/Nk_{B}T`. This is the first order term in the high-temperature perturbation expansion
        
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
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = stb.calc_zetax(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["dkl"],
            )

        # compute components of eq. 19
        a1kl = self._cm.calc_a1ii(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dkl"],
            self.eos_dict["lambdaakl"],
            self.eos_dict["lambdarkl"],
            self.eos_dict["x0kl"],
            self.eos_dict["epsilonkl"],
            zetax,
        )

        # eq. 18
        a1 = np.einsum("ijk,jk->i", a1kl, self.eos_dict["xskl"])
        A1 = (self.eos_dict["Cmol2seg"] / T) * a1  # Units of K

        return A1

    def Asecond_order(self, rho, T, xi, zetaxstar=None, zetax=None, KHS=None):
        r"""
        Outputs :math:`A^{2nd order}/Nk_{B}T`. This is the second order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetaxstar : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on sigma for groups (k,l)
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
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = stb.calc_zetax(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["dkl"],
            )

        if zetaxstar is None:
            zetaxstar = stb.calc_zetaxstar(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["sigmakl"],
            )

        if KHS is None:
            KHS = stb.calc_KHS(zetax)

        ## compute a2kl, eq. 30 #####

        # compute f1, f2, and f3 for eq. 32
        fmlist123 = self.calc_fm(self.eos_dict["alphakl"], np.array([1, 2, 3]))

        chikl = (
            np.einsum("i,jk", zetaxstar, fmlist123[0])
            + np.einsum("i,jk", zetaxstar ** 5, fmlist123[1])
            + np.einsum("i,jk", zetaxstar ** 8, fmlist123[2])
        )

        a1s_2la = self._cm.calc_a1s(
            rho,
            self.eos_dict["Cmol2seg"],
            2.0 * self.eos_dict["lambdaakl"],
            zetax,
            self.eos_dict["epsilonkl"],
            self.eos_dict["dkl"],
        )
        a1s_2lr = self._cm.calc_a1s(
            rho,
            self.eos_dict["Cmol2seg"],
            2.0 * self.eos_dict["lambdarkl"],
            zetax,
            self.eos_dict["epsilonkl"],
            self.eos_dict["dkl"],
        )
        a1s_lalr = self._cm.calc_a1s(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["lambdaakl"] + self.eos_dict["lambdarkl"],
            zetax,
            self.eos_dict["epsilonkl"],
            self.eos_dict["dkl"],
        )
        B_2la = self._cm.calc_Bkl(
            rho,
            2.0 * self.eos_dict["lambdaakl"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dkl"],
            self.eos_dict["epsilonkl"],
            self.eos_dict["x0kl"],
            zetax,
        )
        B_2lr = self._cm.calc_Bkl(
            rho,
            2.0 * self.eos_dict["lambdarkl"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dkl"],
            self.eos_dict["epsilonkl"],
            self.eos_dict["x0kl"],
            zetax,
        )
        B_lalr = self._cm.calc_Bkl(
            rho,
            self.eos_dict["lambdaakl"] + self.eos_dict["lambdarkl"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dkl"],
            self.eos_dict["epsilonkl"],
            self.eos_dict["x0kl"],
            zetax,
        )

        a2kl = (
            (self.eos_dict["x0kl"] ** (2.0 * self.eos_dict["lambdaakl"]))
            * (a1s_2la + B_2la)
            / constants.molecule_per_nm3
            - (
                (
                    2.0
                    * self.eos_dict["x0kl"]
                    ** (self.eos_dict["lambdaakl"] + self.eos_dict["lambdarkl"])
                )
                * (a1s_lalr + B_lalr)
                / constants.molecule_per_nm3
            )
            + (
                (self.eos_dict["x0kl"] ** (2.0 * self.eos_dict["lambdarkl"]))
                * (a1s_2lr + B_2lr)
                / constants.molecule_per_nm3
            )
        )
        a2kl *= (
            (1.0 + chikl) * self.eos_dict["epsilonkl"] * (self.eos_dict["Ckl"] ** 2)
        )  # *(KHS/2.0)

        a2kl = np.einsum("i,ijk->ijk", KHS / 2.0, a2kl)

        # eq. 29
        a2 = np.einsum("ijk,jk->i", a2kl, self.eos_dict["xskl"])
        A2 = (self.eos_dict["Cmol2seg"] / (T ** 2)) * a2

        return A2

    def Athird_order(self, rho, T, xi, zetaxstar=None):
        r"""
        Outputs :math:`A^{3rd order}/Nk_{B}T`. This is the third order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetaxstar : numpy.ndarray, Optional, default=None
            Matrix of hypothetical packing fraction based on sigma for groups (k,l)
        
        Returns
        -------
        Athird_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetaxstar is None:
            zetaxstar = stb.calc_zetaxstar(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["sigmakl"],
            )

        # compute a3kl
        fmlist456 = self.calc_fm(self.eos_dict["alphakl"], np.array([4, 5, 6]))

        a3kl = np.einsum(
            "i,jk", zetaxstar, -(self.eos_dict["epsilonkl"] ** 3) * fmlist456[0]
        ) * np.exp(
            np.einsum("i,jk", zetaxstar, fmlist456[1])
            + np.einsum("i,jk", zetaxstar ** 2, fmlist456[2])
        )  # a3kl=-(epsilonkl**3)*fmlist456[0]*zetaxstar*np.exp((fmlist456[1]*zetaxstar)+(fmlist456[2]*(zetaxstar**2)))

        # eq. 37
        a3 = np.einsum("ijk,jk->i", a3kl, self.eos_dict["xskl"])
        A3 = (self.eos_dict["Cmol2seg"] / (T ** 3)) * a3

        return A3

    def Amonomer(self, rho, T, xi):
        r"""
        Outputs :math:`A^{mono.}/Nk_{B}T`. This is composed
        
        Outputs :math:`A^{HS}/Nk_{B}T, A_1/Nk_{B}T, A_2/Nk_{B}T`, and :math:`A_3/Nk_{B}T` (number of densities) :math:`A^{mono.}/Nk_{B}T` components as well as some related quantities. Eta is really zeta
    
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

        if np.all(rho >= self.density_max(xi, T, maxpack=1.0)):
            raise ValueError(
                "Density values should not all be greater than {}, or calc_Amono will fail in log calculation.".format(
                    self.density_max(xi, T)
                )
            )

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        zetax = stb.calc_zetax(
            rho, self.eos_dict["Cmol2seg"], self.eos_dict["xskl"], self.eos_dict["dkl"]
        )
        zetaxstar = stb.calc_zetaxstar(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["xskl"],
            self.eos_dict["sigmakl"],
        )
        Amonomer = (
            self.Ahard_sphere(rho, T, xi)
            + self.Afirst_order(rho, T, xi, zetax=zetax)
            + self.Asecond_order(rho, T, xi, zetax=zetax, zetaxstar=zetaxstar)
            + self.Athird_order(rho, T, xi, zetaxstar=zetaxstar)
        )

        return Amonomer

    def gdHS(self, rho, T, xi, zetax=None):
        r"""
        The zeroth order expansion term in calculating the radial distribution function of a Mie fluid. 

        This is also known as the hard sphere radial distribution function.
        
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
        gdHS : numpy.ndarray
            Hard sphere radial distribution function
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = stb.calc_zetax(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["dkl"],
            )

        km = np.zeros((np.size(rho), 4))
        gdHS = np.zeros((np.size(rho), np.size(xi)))

        km[:, 0] = -np.log(1.0 - zetax) + (
            42.0 * zetax - 39.0 * zetax ** 2 + 9.0 * zetax ** 3 - 2.0 * zetax ** 4
        ) / (6.0 * (1.0 - zetax) ** 3)
        km[:, 1] = (zetax ** 4 + 6.0 * zetax ** 2 - 12.0 * zetax) / (
            2.0 * (1.0 - zetax) ** 3
        )
        km[:, 2] = -3.0 * zetax ** 2 / (8.0 * (1.0 - zetax) ** 2)
        km[:, 3] = (-zetax ** 4 + 3.0 * zetax ** 2 + 3.0 * zetax) / (
            6.0 * (1.0 - zetax) ** 3
        )

        for i in range(self.ncomp):
            gdHS[:, i] = np.exp(
                km[:, 0]
                + km[:, 1] * self.eos_dict["x0ii"][i]
                + km[:, 2] * self.eos_dict["x0ii"][i] ** 2
                + km[:, 3] * self.eos_dict["x0ii"][i] ** 3
            )

        return gdHS

    def g1(self, rho, T, xi, zetax=None):
        r"""
        Calculate the first order expansion term in calculating the radial distribution function of a Mie fluid
        
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
        g1 : numpy.ndarray
            First order expansion term in calculating the radial distribution function of a Mie fluid
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = stb.calc_zetax(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["dkl"],
            )

        da1iidrhos = self._cm.calc_da1iidrhos(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["lambdaaii_avg"],
            self.eos_dict["lambdarii_avg"],
            self.eos_dict["x0ii"],
            self.eos_dict["epsilonii_avg"],
            zetax,
        )

        a1sii_lambdaaii_avg = self._cm.calc_a1s_eff(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["lambdaaii_avg"],
            zetax,
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
        )
        a1sii_lambdarii_avg = self._cm.calc_a1s_eff(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["lambdarii_avg"],
            zetax,
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
        )

        Bii_lambdaaii_avg = self._cm.calc_Bkl_eff(
            rho,
            self.eos_dict["lambdaaii_avg"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["x0ii"],
            zetax,
        )
        Bii_lambdarii_avg = self._cm.calc_Bkl_eff(
            rho,
            self.eos_dict["lambdarii_avg"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["x0ii"],
            zetax,
        )

        Cii = prefactor(self.eos_dict["lambdarii_avg"], self.eos_dict["lambdaaii_avg"])

        tmp1 = 1.0 / (
            2.0
            * np.pi
            * self.eos_dict["epsilonii_avg"]
            * self.eos_dict["dii_eff"] ** 3
            * constants.molecule_per_nm3 ** 2
        )
        tmp11 = 3.0 * da1iidrhos
        tmp21 = (
            Cii
            * self.eos_dict["lambdaaii_avg"]
            * (self.eos_dict["x0ii"] ** self.eos_dict["lambdaaii_avg"])
        )
        tmp22 = np.einsum(
            "ij,i->ij",
            (a1sii_lambdaaii_avg + Bii_lambdaaii_avg),
            1.0 / (rho * self.eos_dict["Cmol2seg"]),
        )
        tmp31 = (
            Cii
            * self.eos_dict["lambdarii_avg"]
            * (self.eos_dict["x0ii"] ** self.eos_dict["lambdarii_avg"])
        )
        tmp32 = np.einsum(
            "ij,i->ij",
            (a1sii_lambdarii_avg + Bii_lambdarii_avg),
            1.0 / (rho * self.eos_dict["Cmol2seg"]),
        )
        g1 = tmp1 * (tmp11 - tmp21 * tmp22 + tmp31 * tmp32)

        return g1

    def g2(self, rho, T, xi, zetax=None):
        r"""
        Calculate the second order expansion term in calculating the radial distribution function of a Mie fluid
        
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
        g2 : numpy.ndarray
            Second order expansion term in calculating the radial distribution function of a Mie fluid
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = stb.calc_zetax(
                rho,
                self.eos_dict["Cmol2seg"],
                self.eos_dict["xskl"],
                self.eos_dict["dkl"],
            )
        zetaxstar = stb.calc_zetaxstar(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["xskl"],
            self.eos_dict["sigmakl"],
        )
        KHS = stb.calc_KHS(zetax)

        Cii = prefactor(self.eos_dict["lambdarii_avg"], self.eos_dict["lambdaaii_avg"])

        phi7 = np.array([10.0, 10.0, 0.57, -6.7, -8.0])
        alphaii = Cii * (
            (1.0 / (self.eos_dict["lambdaaii_avg"] - 3.0))
            - (1.0 / (self.eos_dict["lambdarii_avg"] - 3.0))
        )
        theta = np.exp(self.eos_dict["epsilonii_avg"] / T) - 1.0

        gammacii = np.zeros((np.size(rho), np.size(xi)))
        for i in range(self.ncomp):
            gammacii[:, i] = (
                phi7[0]
                * (-np.tanh(phi7[1] * (phi7[2] - alphaii[i])) + 1.0)
                * zetaxstar
                * theta[i]
                * np.exp(phi7[3] * zetaxstar + phi7[4] * (zetaxstar ** 2))
            )

        da2iidrhos = self._cm.calc_da2ii_1pchi_drhos(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["x0ii"],
            self.eos_dict["lambdarii_avg"],
            self.eos_dict["lambdaaii_avg"],
            zetax,
        )

        a1sii_2lambdaaii_avg = self._cm.calc_a1s_eff(
            rho,
            self.eos_dict["Cmol2seg"],
            2.0 * self.eos_dict["lambdaaii_avg"],
            zetax,
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
        )
        a1sii_2lambdarii_avg = self._cm.calc_a1s_eff(
            rho,
            self.eos_dict["Cmol2seg"],
            2.0 * self.eos_dict["lambdarii_avg"],
            zetax,
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
        )
        a1sii_lambdarii_avglambdaaii_avg = self._cm.calc_a1s_eff(
            rho,
            self.eos_dict["Cmol2seg"],
            self.eos_dict["lambdaaii_avg"] + self.eos_dict["lambdarii_avg"],
            zetax,
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["dii_eff"],
        )

        Bii_2lambdaaii_avg = self._cm.calc_Bkl_eff(
            rho,
            2.0 * self.eos_dict["lambdaaii_avg"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["x0ii"],
            zetax,
        )
        Bii_2lambdarii_avg = self._cm.calc_Bkl_eff(
            rho,
            2.0 * self.eos_dict["lambdarii_avg"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["x0ii"],
            zetax,
        )
        Bii_lambdaaii_avglambdarii_avg = self._cm.calc_Bkl_eff(
            rho,
            self.eos_dict["lambdaaii_avg"] + self.eos_dict["lambdarii_avg"],
            self.eos_dict["Cmol2seg"],
            self.eos_dict["dii_eff"],
            self.eos_dict["epsilonii_avg"],
            self.eos_dict["x0ii"],
            zetax,
        )

        eKC2 = np.einsum(
            "i,j->ij",
            KHS / rho / self.eos_dict["Cmol2seg"],
            self.eos_dict["epsilonii_avg"] * (Cii ** 2),
        )

        g2MCA = (
            1.0
            / (
                2.0
                * np.pi
                * (self.eos_dict["epsilonii_avg"] ** 2)
                * self.eos_dict["dii_eff"] ** 3
                * constants.molecule_per_nm3 ** 2
            )
        ) * (
            (3.0 * da2iidrhos)
            - (
                eKC2
                * self.eos_dict["lambdarii_avg"]
                * (self.eos_dict["x0ii"] ** (2.0 * self.eos_dict["lambdarii_avg"]))
            )
            * (a1sii_2lambdarii_avg + Bii_2lambdarii_avg)
            + eKC2
            * (self.eos_dict["lambdarii_avg"] + self.eos_dict["lambdaaii_avg"])
            * (
                self.eos_dict["x0ii"]
                ** (self.eos_dict["lambdarii_avg"] + self.eos_dict["lambdaaii_avg"])
            )
            * (a1sii_lambdarii_avglambdaaii_avg + Bii_lambdaaii_avglambdarii_avg)
            - eKC2
            * self.eos_dict["lambdaaii_avg"]
            * (self.eos_dict["x0ii"] ** (2.0 * self.eos_dict["lambdaaii_avg"]))
            * (a1sii_2lambdaaii_avg + Bii_2lambdaaii_avg)
        )

        g2 = (1.0 + gammacii) * g2MCA

        return g2

    def Achain(self, rho, T, xi):
        r"""
        Outputs the chain term for the Helmholtz energy, :math:`A^{chain}/Nk_{B}T`.
    
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
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        zetax = stb.calc_zetax(
            rho, self.eos_dict["Cmol2seg"], self.eos_dict["xskl"], self.eos_dict["dkl"]
        )
        gdHS = self.gdHS(rho, T, xi, zetax=zetax)
        g1 = self.g1(rho, T, xi, zetax=zetax)
        g2 = self.g2(rho, T, xi, zetax=zetax)

        gii = gdHS * np.exp(
            (self.eos_dict["epsilonii_avg"] * g1 / (T * gdHS))
            + (((self.eos_dict["epsilonii_avg"] / T) ** 2) * g2 / gdHS)
        )

        Achain = 0.0
        for i in range(self.ncomp):
            beadsum = -1.0 + self.eos_dict["num_rings"][i]
            for k in range(self.nbeads):
                beadsum += (
                    self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["Vks"][k]
                    * self.eos_dict["Sk"][k]
                )
            Achain -= xi[i] * beadsum * np.log(gii[:, i])

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

        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack

        max_density = (
            maxpack
            * 6.0
            / (
                self.eos_dict["Cmol2seg"]
                * np.pi
                * np.sum(self.eos_dict["xskl"] * (self.eos_dict["dkl"] ** 3))
            )
            / constants.molecule_per_nm3
        )

        return max_density

    @staticmethod
    def calc_fm(alphakl, mlist):
        r"""
        Calculate list of coefficients used to compute the correction term for :math:`A_{2}/Nk_{B}T` which is related to the fluctuations of attractive energy. where a list of m values are specified in mlist eq. 39
        
        Parameters
        ----------
        alphakl : numpy.ndarray
            (Ngroup,Ngroup) "A dimensionless form of the integrated vdW energy of the Mie potential" eq. 33
        mlist : numpy.ndarray
            (number of m values) an array of integers used in the calculation of :math:`A^{mono}`
        
        Returns
        -------
        fmlist : numpy.ndarray
            List of coefficients used to compute the correction term for :math:`A_{2}` which is related to the fluctuations of attractive energy.
        """

        if np.size(np.shape(alphakl)) == 2:
            fmlist = np.zeros(
                (np.size(mlist), np.size(alphakl, axis=0), np.size(alphakl, axis=0))
            )
        elif np.size(np.shape(alphakl)) == 1:
            fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0)))
        else:
            logger.error("Unexpected shape in calc_fm")
        mlist = mlist - 1

        phimn = np.array(
            [
                [
                    7.53655570e00,
                    -3.76046300e01,
                    7.17459530e01,
                    -4.68355200e01,
                    -2.46798200e00,
                    -5.02720000e-01,
                    8.09568830e00,
                ],
                [
                    -3.59440000e02,
                    1.82560000e03,
                    -3.16800000e03,
                    1.88420000e03,
                    -8.23760000e-01,
                    -3.19350000e00,
                    3.70900000e00,
                ],
                [
                    1.55090000e03,
                    -5.07010000e03,
                    6.53460000e03,
                    -3.28870000e03,
                    -2.71710000e00,
                    2.08830000e00,
                    0.00000000e00,
                ],
                [
                    -1.19932000e00,
                    9.06363200e00,
                    -1.79482000e01,
                    1.13402700e01,
                    2.05214200e01,
                    -5.66377000e01,
                    4.05368300e01,
                ],
                [
                    -1.91128000e03,
                    2.13901750e04,
                    -5.13207000e04,
                    3.70645400e04,
                    1.10374200e03,
                    -3.26461000e03,
                    2.55618100e03,
                ],
                [
                    9.23690000e03,
                    -1.29430000e05,
                    3.57230000e05,
                    -3.15530000e05,
                    1.39020000e03,
                    -4.51820000e03,
                    4.24160000e03,
                ],
                [
                    1.00000000e01,
                    1.00000000e01,
                    5.70000000e-01,
                    -6.70000000e00,
                    -8.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
            ]
        )

        for i, m in enumerate(mlist):
            for n in range(4):
                fmlist[i] += phimn[m, n] * (alphakl ** n)
            dum = np.ones_like(fmlist[i])
            for n in range(4, 7):
                dum += phimn[m, n] * (alphakl ** (n - 3.0))
            fmlist[i] = fmlist[i] / dum

        return fmlist

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
            Indicates which radial distribution function to return

            - 'ijklab': The bonding volume was calculated from self.calc_Kijklab, return gHS_dij)
            - 'klab': The bonding volume was provided to saft.py so use temperature-density polynomial correlation
    
        Returns
        -------
        Iij : numpy.ndarray
            A temperature-density polynomial correlation of the association integral for a Lennard-Jones monomer. This matrix is (len(rho) x Ncomp x Ncomp)
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if Ktype == "klab":
            gr = calc_Iij(
                rho,
                T,
                xi,
                self.eos_dict["epsilonii_avg"],
                self.eos_dict["sigmaii_avg"],
                self.eos_dict["sigmakl"],
                self.eos_dict["xskl"],
            )
        elif Ktype == "ijklab":
            gr = self.calc_gdHS_assoc(rho, T, xi)
        else:
            raise ValueError(
                "Ktype does not indicate a known gr_assoc for this saft type."
            )

        return gr

    def calc_gdHS_assoc(self, rho, T, xi):
        r"""
            
        Radial distribution function at contact.

        Papaioannou J. Chem. Phys. 140, 054107 (2014)
        
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
        gr : numpy.ndarray
            This matrix is (len(rho) x Ncomp x Ncomp)
        """

        rho = self._check_density(rho)
        self._check_temperature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        eta = np.zeros((np.size(rho), 2))
        for m in range(2, 4):
            eta[:, m] = (
                rho
                * constants.molecule_per_nm3
                * self.eos_dict["Cmol2seg"]
                * (
                    np.sum(
                        np.sqrt(np.diag(self.eos_dict["xskl"]))
                        * (np.diag(self.eos_dict["dkl"]) ** m)
                    )
                    * (np.pi / 6.0)
                )
            )

        gr = np.zeros((len(rho), self.ncomp, self.ncomp))
        tmp0 = 1 / (1 - eta[:, 1])
        tmp1 = eta[:, 0] / (1 - eta[:, 1]) ** 2
        tmp2 = eta[:, 0] ** 2 / (1 - eta[:, 1]) ** 3
        for i in range(ncomp):
            for j in range(ncomp):
                tmp = (
                    self.eos_dict["dii_eff"][i]
                    * self.eos_dict["dii_eff"][j]
                    / (self.eos_dict["dii_eff"][i] + self.eos_dict["dii_eff"][j])
                )
                gr[:, i, j] = tmp0 + 3 * tmp * tmp1 + 2 * tmp ** 2 * tmp2

        return gr

    def calc_Kijklab(self, T, rc_klab, rd_klab=None, reduction_ratio=0.25):
        r"""
            
        Calculation of association site bonding volume, dependent on molecule in addition to group

        Papaioannou J. Chem. Phys. 140, 054107 (2014)
        
        Parameters
        ----------
        T : float
            Temperature of the system [K]
    
        Returns
        -------
        gr : numpy.ndarray
            This matrix is (len(rho) x Ncomp x Ncomp)
        """

        self._check_temperature_dependent_parameters(T)

        dij_bar = np.zeros((self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                dij_bar[i, j] = np.mean(
                    [self.eos_dict["dii_eff"][i], self.eos_dict["dii_eff"][j]]
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
        alphakl : np.array
            (Ngroup,Ngroup) "A dimensionless form of the integrated vdW energy of the Mie potential" eq. 33
        eos_dict : dict
            The following entries are updated:

            - Ckl (numpy.ndarray) - Matrix of Mie potential prefactors between beads  (l,k)
            - epsilonkl (numpy.ndarray) - Matrix of Mie potential well depths for groups (k,l)
            - sigmakl (numpy.ndarray) - Matrix of bead diameters (k,l)
            - lambdarkl (numpy.ndarray) - Matrix of repulsive Mie exponent for groups (k,l)
            - lambdaakl (numpy.ndarray) - Matrix of attractive Mie exponent for groups (k,l)
            - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.
            - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l

        """

        self.bead_library.update(bead_library)
        self.cross_library.update(cross_library)

        self.eos_dict["Sk"] = tb.extract_property(
            "Sk", self.bead_library, self.beads, default=1.0
        )

        output = tb.cross_interaction_from_dict(
            self.beads,
            self.bead_library,
            self.combining_rules,
            cross_library=self.cross_library,
        )
        self.eos_dict["sigmakl"] = output["sigma"]
        self.eos_dict["epsilonkl"] = output["epsilon"]
        self.eos_dict["lambdaakl"] = output["lambdaa"]
        self.eos_dict["lambdarkl"] = output["lambdar"]

        # Update Non bonded matrices
        if not np.isnan(self.T) and self.T != None:
            self._check_temperature_dependent_parameters(self.T)
        else:
            self._check_temperature_dependent_parameters(298)

        # Initiate average interaction terms
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

        self.eos_dict["Ckl"] = prefactor(
            self.eos_dict["lambdarkl"], self.eos_dict["lambdaakl"]
        )
        self.eos_dict["alphakl"] = self.eos_dict["Ckl"] * (
            (1.0 / (self.eos_dict["lambdaakl"] - 3.0))
            - (1.0 / (self.eos_dict["lambdarkl"] - 3.0))
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
            raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.0):
            raise ValueError("Density values cannot be negative.")

        return rho

    def _check_temperature_dependent_parameters(self, T):
        r"""
        This function checks that the temperature dependent parameters are computed for the correct value. If not, they are recomputed.
        
        Parameters
        ----------
        T : float
            Temperature of the system [K]
            
        Attributes
        ---------
        T : float
            Updated temperature value
        eos_dict : dict
            The following entries are updated:

            - dkl (numpy.ndarray) - Matrix of hard sphere equivalent for each bead and interaction between them (l,k)
            - x0kl (numpy.ndarray) - Matrix of sigmakl/dkl, sigmakl is the Mie radius for groups (k,l)
            - etc. Other matrices will also be updated if temperature dependent multipole mixing rules are used.

        """

        if self.T != T:
            self.T = T
            # Check for temperature dependent mixing rule
            if self._mixing_temp_dependence == None:
                self._mixing_temp_dependence = False
                for key, value in self.combining_rules.items():
                    if "temperature" in value:
                        self._mixing_temp_dependence = True
                        if "additional_outputs" in value:
                            for params in value["additional_outputs"]:
                                self.combining_rules[params]["function"] = "None"
                        self.combining_rules[key]["temperature"] = T
            else:
                for key, value in self.combining_rules.items():
                    if "temperature" in value:
                        self.combining_rules[key]["temperature"] = T

            if self._mixing_temp_dependence:
                output = tb.cross_interaction_from_dict(
                    self.beads,
                    self.bead_library,
                    self.combining_rules,
                    cross_library=self.cross_library,
                )
                self.eos_dict["sigmakl"] = output["sigma"]
                self.eos_dict["epsilonkl"] = output["epsilon"]
                self.eos_dict["lambdaakl"] = output["lambdaa"]
                self.eos_dict["lambdarkl"] = output["lambdar"]

                # compute alphakl eq. 33
                self.eos_dict["Ckl"] = prefactor(
                    self.eos_dict["lambdarkl"], self.eos_dict["lambdaakl"]
                )
                self.eos_dict["alphakl"] = self.eos_dict["Ckl"] * (
                    (1.0 / (self.eos_dict["lambdaakl"] - 3.0))
                    - (1.0 / (self.eos_dict["lambdarkl"] - 3.0))
                )
                self.calc_component_averaged_properties()

            self.eos_dict["dkl"], self.eos_dict[
                "x0kl"
            ] = stb.calc_hard_sphere_matricies(
                T, self.eos_dict["sigmakl"], self.bead_library, self.beads, prefactor
            )
            self._update_chain_temperature_dependent_variables(T)

    def _check_composition_dependent_parameters(self, xi):
        r"""
        This function checks that the composition dependent parameters are computed for the correct value. If not, they are recomputed.
        
        Parameters
        ----------
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Attributes
        ---------
        xi : numpy.ndarray
            Component mole fractions are updated
        eos_dict : dict
            The following entries are updated:

            - Cmol2seg (float) - Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`.
            - xskl (numpy.ndarray) - Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l

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

    def _update_chain_temperature_dependent_variables(self, T):
        r"""
        This function checks that the temperature dependent parameters for the chain contribution are computed for the correct value. If not, they are recomputed.
        
        Parameters
        ----------
        T : float
            Temperature of the system [K]
        
        Attributes
        ---------
        T : float
            Updated temperature value
        eos_dict : dict
            The following entries are updated:

            - dii_eff (numpy.ndarray) - Matrix of mole averaged hard sphere equivalent for each bead and interaction between them (i.j)
            - x0ii (numpy.ndarray) - Matrix of sigmaii_avg/dii_eff, sigmaii_avg is the average molecular Mie radius and dii_eff the average molecular hard sphere diameter

        """

        zki = np.zeros((self.ncomp, self.nbeads), float)
        zkinorm = np.zeros(self.ncomp, float)
        dii_eff = np.zeros((self.ncomp), float)
        # compute zki
        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = (
                    self.eos_dict["molecular_composition"][i, k]
                    * self.eos_dict["Vks"][k]
                    * self.eos_dict["Sk"][k]
                )
                zkinorm[i] += zki[i, k]

        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = zki[i, k] / zkinorm[i]

        for i in range(self.ncomp):
            for k in range(self.nbeads):
                for l in range(self.nbeads):
                    dii_eff[i] += (
                        zki[i, k] * zki[i, l] * self.eos_dict["dkl"][k, l] ** 3
                    )
            dii_eff[i] = dii_eff[i] ** (1 / 3.0)
        self.eos_dict["dii_eff"] = dii_eff

        # compute x0ii
        self.eos_dict["x0ii"] = self.eos_dict["sigmaii_avg"] / dii_eff

    def __str__(self):

        string = "Beads: {}".format(self.beads)
        return string
