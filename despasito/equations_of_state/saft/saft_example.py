# -- coding: utf8 --
r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import os
import sys
#np.set_printoptions(threshold=sys.maxsize)


import despasito.equations_of_state.eos_toolbox as tb
from despasito.equations_of_state import constants
import despasito.equations_of_state.saft.saft_toolbox as stb

logger = logging.getLogger(__name__)

from despasito.main import method_stat
if method_stat.disable_cython and method_stat.disable_numba:
    pass
elif not method_stat.disable_cython:
    logger.warning("saft.gamma_sw does not use cython.")
elif not method_stat.disable_numba:
    logger.warning("saft.gamma_sw does not use numba.")

class saft_example():

    r"""
    This heavily annotated, nonfunctional version of the SAFT-gamma-SW class can be used as a template for a new SAFT EOS version. Note, you must add it to the factory in saft.py before it can be used.
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters
    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.
        
    Attributes
    ----------
    eos_dict : dict
        A dictionary that packages all the relevant parameters
    
    """

    def __init__(self, kwargs):

        ####### The Following lines are *MANDATORY* but can be edited as needed

        # Standard attributes of the EosTemplate are 'density_max' and 'parameter_refresh', while all saft version require 'calc_gr_assoc' for calculating association sites
    
        # This keyword is used by the saft.py class and subsequently the `Aideal_contribution` function. If a user desired a different default method of calulating the ideal contribution, add a new function to Aideal.py
        self.Aideal_method = "Abroglie"

        # This list of strings are the bead types that the parameter update method can expect.
        # The bead library should use these same parameter names.
        # Parameter names should not contain '-' or '_'
        self.parameter_types = ["epsilon", "epsilonHB", "sigma", "Sk", "K"]

        # This dictionary will contain the feasible bounds of the parameters listed in 'parameter_types'. Bounds are necessary for each type.
        self.parameter_bound_extreme = {"epsilon":[0.,1000.], "sigma":[0.,1.0], "Sk":[0.,1.], "epsilonHB":[0.,5000.], "K":[0.,10000.]}    

        # This list contained the Helmholtz energy contributions contained in this class below. The class in saft.py will add these as as it's own attributes to calculate the total helmholtz energy. 
        self.residual_helmholtz_contributions = ["Amonomer","Achain"]
        # Note that these strings must represent methods below, but if the method is too verbose or repeditive, other methods may be added to this object. If a function is to be optimized with Cython or Numba, an extentions library can be added to the 'compiled_modules' directory. A python version with the same function names should also be present aswell. The import "if-structure" at the top is then used to import the desired form, see gamma-mie.py for an example.
        # When deciding whether to include an additional function as a class method, or if it should simply be imported from a library. Think about whether accessing that function would be nice when handling an EOS object in a python script. For instance, the 'reduced density', 'effiective packing fraction', 'Ahard_sphere', 'A1'... methods would be nice to have as attributes.

        # When calculating the cross-interaction term between two beads, the parameter name and the mixing type should be listed here. The mixing rule keyword can be any that are supported in saft_toolbox.mixing_rules. As of right now, mixing rules that use other parameters are not supported by this function, so a custom mixing rule method is added as a method in this class. Even if all your mixing rules are handled internally, this attribute must exist.
        self.mixing_rules = {"sigma": "mean"}
    
        # Now we start processing the given variables. The following three attributes are always needs for the saft.py class. If other inputs are needed for the specific SAFT type at hand, feel free to add them to this list.
        self.eos_dict = {}
        needed_attributes = ['nui','beads','beadlibrary']
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                self.eos_dict[key] = kwargs[key]

        if 'crosslibrary' not in kwargs:
            self.eos_dict['crosslibrary'] = {}
        else:
            self.eos_dict['crosslibrary'] = kwargs['crosslibrary']

        ###### The following lines are *OPTIONAL* and are completely for internal use for this specific SAFT type and aren't mandatory 

        # This is an optional line that processes the mass of beads for the Aideal method, Abroglie
        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # The following terms are used by SAFT-gamma variations 
        if not hasattr(self, 'Vks'):
            self.eos_dict['Vks'] = tb.extract_property("Vks",self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Sk'):
            self.eos_dict['Sk'] = tb.extract_property("Sk",self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # Initialize composition attribute. This is for composition dependent properties. By recording this, we can avoid recalculating those parameters unnecessarily. In saft-gamma-mie we also have temperature dependent parameters and so self.T is included.
        if not hasattr(self, 'xi'):
            self.xi = np.nan

        # These are initialized for loops used in some of the methods below. Depending on how you choose to break things up, these might not be needed.
        if not hasattr(self, 'nbeads'):
            self.ncomp, self.nbeads = np.shape(self.eos_dict['nui'])

        # Intiate cross interaction terms, as mentioned above, some mixing rules use a particular combination of parameters and these are handled here.
        output = tb.cross_interaction_from_dict( self.eos_dict['beads'], self.eos_dict['beadlibrary'], self.mixing_rules, crosslibrary=self.eos_dict['crosslibrary'])
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.calc_sw_cross_interaction_parameters()

        # This optional keyword can be passed in an input file as "eos_num_rings". If your chosen version of SAFT needs special keyward passed that don't fall under a catagory above. Make a note in the doc string to pass it as "eos_somekeyword"
        if "num_rings" in kwargs:
            self.eos_dict['num_rings'] = kwargs['num_rings']
            logger.info("Accepted component ring structure: {}".format(kwargs["num_rings"]))
        else:
            self.eos_dict['num_rings'] = np.zeros(len(self.eos_dict['nui']))

        # Initiate average interaction terms
        self.calc_component_averaged_properties()
        self.alphakl = 2.0*np.pi/3.0*self.eos_dict['epsilon_kl']*self.eos_dict['sigma_kl']**3*(self.eos_dict['lambda_kl']**3 - 1.0)

    def calc_sw_cross_interaction_parameters(self, mode="normal"): # Defined above as *MANDATORY*
        r""" Calculate mixed energy parameter. This optional method was added to handle specific mixing rules that require multiple parameters.
  
        Parameters
        ----------
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        """

        # The guts of this function are irrelevant to our purpose. See gamma_sw.py for more detail

        self.eos_dict["lambda_ij"] = np.diagflat(lambdaii)
        self.eos_dict["epsilon_ij"] = np.diagflat(epsilonii)
        self.calc_sw_cross_interaction_parameters(mode="effective")

    def reduced_density(self, rho, xi): # *OPTIONAL*
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
            Reduced density (len(rho), 4)
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        # The guts of this function are irrelevant to our purpose. See gamma_sw.py for more detail


        return reduced_density

    def effective_packing_fraction(self, rho, xi, zetax=None, mode="normal"): # *OPTIONAL*
        r"""
        Effective packing fraction for SAFT-gamma with a square-wave potential
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        
        Returns
        -------
        zeta_eff : numpy.ndarray
            Effective packing fraction (len(rho), Nbeads, Nbeads)
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        # The guts of this function are irrelevant to our purpose. See gamma_sw.py for more detail

        return zetakl

    def _dzetaeff_dzetax(self, rho, xi, zetax=None, mode="normal"): # *OPTIONAL*
        r"""
        Effective packing fraction for SAFT-gamma with a square-wave potential
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        
        Returns
        -------
        zeta_eff : numpy.ndarray
            Effective packing fraction (len(rho), Nbeads, Nbeads)
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        # The guts of this function are irrelevant to our purpose. See gamma_sw.py for more detail

        return dzetakl

        
    def Ahard_sphere(self,rho, T, xi): # *OPTIONAL*
        r"""
        Outputs :math:`A^{HS}`.
        
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

        # The guts of this function are irrelevant to our purpose. See gamma_sw.py for more detail

        return AHS
    
    def Afirst_order(self,rho, T, xi, zetax=None):  # *OPTIONAL*
        r"""
        Outputs :math:`A^{1st order}`. This is the first order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        
        Returns
        -------
        Afirst_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        g0HS = self.calc_g0HS(rho, xi, zetax=zetax)
        a1kl_tmp = np.tensordot(rho * constants.molecule_per_nm3, self.eos_dict['xskl']*self.alphakl, 0)
        A1 = -(self.eos_dict['Cmol2seg']**2 / T) * np.sum(a1kl_tmp * g0HS, axis=(1,2)) # Units of K

        #print("A1",A1)

        return A1

    def Asecond_order(self, rho, T, xi, zetax=None, KHS=None):  # *OPTIONAL*
        r"""
        Outputs :math:`A^{2nd order}`. This is the second order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        KHS : numpy.ndarray, Optional, default: None
            (length of densities) isothermal compressibility of system with packing fraction zetax
        
        Returns
        -------
        Asecond_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """
        
        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]
        # Note that zetax = zeta3

        if KHS is None:
            KHS = stb.calc_KHS(zetax)
        
        dzetakl = self._dzetaeff_dzetax(rho, xi, zetax=zetax)
        zeta_eff = self.effective_packing_fraction(rho, xi, zetax=zetax)
        g0HS = self.calc_g0HS(rho, xi, zetax=zetax)

        rho2 = self.eos_dict['Cmol2seg'] * rho * constants.molecule_per_nm3

        tmp1 = KHS * rho2 / 2.0
        tmp2 = self.eos_dict['epsilon_kl'] * self.alphakl * self.eos_dict['xskl']
        a2kl_tmp = np.tensordot( tmp1, tmp2, 0)
        a2 = a2kl_tmp*(g0HS + zetax[:,np.newaxis,np.newaxis]*dzetakl*(2.5 - zeta_eff)/(1-zeta_eff)**4)

        # NoteHere: this negative sign is in the final expression for A2 but not in any of the components
        A2 = (self.eos_dict['Cmol2seg'] / (T**2)) * np.sum(a2, axis=(1,2))

        #print("A2",A2)

        return A2
    
    def Amonomer(self,rho, T, xi): # Defined above as *MANDATORY*
        r"""
        Outputs :math:`A^{mono.}`. This is composed
        
        Outputs :math:`A^{HS}, A_1, A_2`, and :math:`A_3` (number of densities) :math:`A^{mono.}` components as well as some related quantities. Note these quantities are normalized by NkbT. Eta is really zeta
    
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
            raise ValueError("Density values should not all be greater than {}, or calc_Amono will fail in log calculation.".format(self.density_max(xi, T)))

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zetax = self.reduced_density(rho, xi)[:,3]

        Amonomer = self.Ahard_sphere(rho, T, xi) + self.Afirst_order(rho, T, xi, zetax=zetax) + self.Asecond_order(rho, T, xi, zetax=zetax)

        return Amonomer

    def calc_g0HS(self, rho, xi, zetax=None, mode="normal"):  # *OPTIONAL*
        r"""
        The contact value of the pair correlation function of a hypothetical pure fluid of diameter sigmax evaluated at an effective packing fraction, zeta_eff.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective", where normal used bead interaction matricies, and effective uses component averaged parameters.
        
        Returns
        -------
        g0HS : numpy.ndarray
            The contact value of the pair correlation function of a hypothetical pure fluid
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        zeta_eff = self.effective_packing_fraction(rho, xi, mode=mode, zetax=zetax)

        g0HS = (1.0 - zeta_eff/2.0) / (1.0 - zeta_eff)**3

        return g0HS

    def calc_gHS(self, rho, xi):  # *OPTIONAL*
        r"""
        
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        
        Returns
        -------
        Afirst_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zetam = self.reduced_density(rho, xi)

        tmp1 = 1.0 / (1.0 - zetam[:,3])
        tmp2 = zetam[:,2] / (1.0 - zetam[:,3])**2
        tmp3 = zetam[:,2]**2 / (1.0 - zetam[:,3])**3

        gHS = np.zeros((np.size(rho), self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(self.ncomp):
                tmp = constants.molecule_per_nm3 * self.eos_dict['sigma_ij'][i,i]*self.eos_dict['sigma_ij'][j,j]/(self.eos_dict['sigma_ij'][i,i]+self.eos_dict['sigma_ij'][j,j])
                gHS[:,i,j] = tmp1 + 3*tmp*tmp2 + 2*tmp**2*tmp3

        return gHS

    def calc_gSW(self, rho, T, xi, zetax=None):  # *OPTIONAL*
        r"""
        
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        
        Returns
        -------
        Afirst_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        rho = self._check_density(rho)
        self._check_composition_dependent_parameters(xi)
        kT = T * constants.kb

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        g0HS = self.calc_g0HS(rho, xi, zetax=zetax, mode="effective")
        gHS = self.calc_gHS(rho, xi)
        zeta_eff = self.effective_packing_fraction(rho, xi, mode="effective", zetax=zetax)
        dg0HSdzetaeff = (2.5 - zeta_eff)/(1.0 - zeta_eff)**4

        ncomp = len(xi)
        dckl_coef = np.array([[-1.50349, 0.249434],[1.40049, -0.827739],[-15.0427, 5.30827]])
        zetax_pow = np.transpose(np.array([zetax, zetax**2, zetax**3]))
        dzetaijdlambda = np.zeros((np.size(rho), ncomp, ncomp))
        for i in range(ncomp):
            for j in range(ncomp):
                cikl = np.dot(dckl_coef, np.array([1.0, (2*self.eos_dict['lambda_ij'][i, j])]))
                dzetaijdlambda[:, i, j] = np.dot( zetax_pow, cikl)

        dzetaijdzetax = self._dzetaeff_dzetax(rho, xi, zetax=zetax, mode="effective")
        dzetaeff = self.eos_dict['lambda_ij'][np.newaxis,:,:]/3.0*dzetaijdlambda - zetax[:,np.newaxis,np.newaxis]*dzetaijdzetax
    
        gSW = gHS + self.eos_dict['epsilon_ij'][np.newaxis,:,:]/ T * (g0HS + (self.eos_dict['lambda_ij'][np.newaxis,:,:]**3-1.0)*dg0HSdzetaeff*dzetaeff)

        return gSW

    def Achain(self, rho, T, xi): # Defined above as *MANDATORY*
        r"""
        Outputs :math:`A^{chain}`.
    
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
   
        #print("gii", gii)
        
        Achain = 0.0
        for i in range(self.ncomp):
            beadsum = -1.0 + self.eos_dict['num_rings'][i]
            for k in range(self.nbeads):
                beadsum += (self.eos_dict['nui'][i, k] * self.eos_dict["Vks"][k] * self.eos_dict["Sk"][k])
            Achain -= xi[i] * beadsum * np.log(gii[:, i,i])

        if np.any(np.isnan(Achain)):
            logger.error("Some Helmholtz values are NaN, check energy parameters.")

        #print("Achain",Achain)

        return Achain

    def density_max(self, xi, T, maxpack=0.65): ############## *MANDATORY*

        """
        Estimate the maximum density based on the hard sphere packing fraction.
        
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]
        maxpack : float, Optional, default: 0.65
            Maximum packing fraction
        
        Returns
        -------
        maxrho : float
            Maximum molar density [mol/m^3]
        """

        self._check_composition_dependent_parameters(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        maxrho = maxpack * 6.0 / (self.eos_dict['Cmol2seg'] * np.pi * np.sum(self.eos_dict['xskl'] * (self.eos_dict['sigma_kl']**3))) / constants.molecule_per_nm3

        return maxrho

    def calc_gr_assoc(self, rho, T, xi): ############## *MANDATORY*
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
    
        Returns
        -------
        Iij : numpy.ndarray
            A temperature-density polynomial correlation of the association integral for a Lennard-Jones monomer. This matrix is (len(rho) x Ncomp x Ncomp)
        """
    
        rho = self._check_density(rho)
        gSW = self.calc_gSW(rho, T, xi)

        return gSW

    def parameter_refresh(self, beadlibrary, crosslibrary): ############## *MANDATORY*
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on _beadlibrary and _crosslibrary attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        self.eos_dict["beadlibrary"].update(beadlibrary)
        self.eos_dict["crosslibrary"].update(crosslibrary)

        # Update Non bonded matrices
        output = tb.cross_interaction_from_dict( self.eos_dict['beads'], self.eos_dict['beadlibrary'], self.mixing_rules, crosslibrary=self.eos_dict['crosslibrary'])
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.calc_sw_cross_interaction_parameters()
        self.calc_component_averaged_properties()

        if not np.isnan(self.xi):
            self.eos_dict['Cmol2seg'], self.eos_dict['xskl'] = stb.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])

    def _check_density(self, rho): # *OPTIONAL*
        r"""
        This function checks the attritutes of the density array
        
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
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

        return rho

    def _check_composition_dependent_parameters(self, xi): # *OPTIONAL*
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Atributes
        ---------
        eos_dict : dict
            The following entries are updated: Cmol2seg, xskl
        """
        xi = np.array(xi)
        if not np.all(self.xi == xi):
            self.eos_dict['Cmol2seg'], self.eos_dict['xskl'] = stb.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])
            self.xi = xi

    def __str__(self):

        string = "Beads: {}".format(self.eos_dict['beads'])
        return string
