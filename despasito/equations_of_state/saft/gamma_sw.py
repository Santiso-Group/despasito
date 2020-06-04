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


import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state import constants
import despasito.equations_of_state.saft.saft_toolbox as stb

logger = logging.getLogger(__name__)

ckl_coef = np.array([[2.25855, -1.50349, 0.249434], [-0.669270, 1.40049, -0.827739], [10.1576, -15.0427, 5.30827]])

class gamma_sw():

    r"""
    
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    
        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.
        
        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
        - l_a: :math:`\lambda^{a}_{k,l}`, Exponent of attractive term between groups of type k and l
    
    """

    def __init__(self, kwargs):
    
        self.Aideal_method = "Abroglie"
        self.parameter_types = ["epsilon", "epsilonHB", "sigma", "Sk", "K"]
        self.parameter_bound_extreme = {"epsilon":[0.,1000.], "sigma":[0.,9e-9], "Sk":[0.,1.], "epsilonHB":[0.,5000.], "K":[0.,10000.]}    
        self.mixing_rules = {"sigma": "mean"}
        self.residual_helmholtz_contributions = ["Amonomer","Achain"]
    
        if not hasattr(self, 'eos_dict'):
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

        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Vks'):
            self.eos_dict['Vks'] = tb.extract_property("Vks",self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Sk'):
            self.eos_dict['Sk'] = tb.extract_property("Sk",self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # Initialize temperature attribute
        if not hasattr(self, 'xi'):
            self.xi = np.nan

        if not hasattr(self, 'nbeads'):
            self.ncomp, self.nbeads = np.shape(self.eos_dict['nui'])
        # Intiate cross interaction terms
        output = stb.cross_interaction_from_dict( self.eos_dict['beads'], self.eos_dict['beadlibrary'], self.mixing_rules, crosslibrary=self.eos_dict['crosslibrary'])
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.calc_sw_cross_interaction_parameters()
        # Initiate average interaction terms
        self.calc_component_averaged_properties()
        self.alphakl = 2.0*np.pi/3.0*self.eos_dict['epsilon_kl']*self.eos_dict['sigma_kl']**3*(self.eos_dict['lambda_kl']**3 - 1.0)

    def calc_sw_cross_interaction_parameters(self, mode="normal"):
        r""" Calculate mixed energy parameter
  
        Parameters
        ----------
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective"
        """

        if mode == "normal":
            tmp = "_kl"
            lx = self.nbeads
        elif mode == "effective":
            tmp = "_ij"
            lx = self.ncomp
        else:
            raise ValueError("The input, {}, is not a supported mode for cross interaction calculations".format(mode))

        # self-interaction
        if mode == "normal":
            epsilon = np.zeros((lx,lx))
            lambda_tmp = np.zeros((lx,lx))
            for k in range(lx):
                bead1 = self.eos_dict['beads'][k]
                epsilon[k,k] = self.eos_dict["beadlibrary"][bead1]["epsilon"]
                lambda_tmp[k,k] = self.eos_dict["beadlibrary"][bead1]["lambda"]
        elif mode == "effective":
            epsilon = self.eos_dict["epsilon_ij"]
            lambda_tmp = self.eos_dict["lambda_ij"]
     
        for k in range(lx):
            for l in range(k,lx):
                 tmp1 = self.eos_dict["sigma"+tmp][k,k]*lambda_tmp[k,k] + self.eos_dict["sigma"+tmp][l,l]*lambda_tmp[l,l]
                 tmp2 = (self.eos_dict["sigma"+tmp][k,k] + self.eos_dict["sigma"+tmp][l,l])
                 lambda_tmp[k,l] = tmp1/tmp2
                 lambda_tmp[l,k] = lambda_tmp[k,l]

                 tmp1 = self.eos_dict["sigma"+tmp][k,k]**3*(lambda_tmp[k,k]**3 - 1.0)*epsilon[k,k]
                 tmp2 = self.eos_dict["sigma"+tmp][l,l]**3*(lambda_tmp[l,l]**3 - 1.0)*epsilon[l,l]
                 tmp3 = self.eos_dict["sigma"+tmp][k,l]**3*(lambda_tmp[k,l]**3 - 1.0)
                 epsilon[k,l] = np.sqrt(tmp1*tmp2)/tmp3
                 epsilon[l,k] = epsilon[k,l]

        # testing if crosslibrary is empty ie not specified
        if mode == "normal" and self.eos_dict["crosslibrary"]:
            # find any cross terms in the cross term library
            crosslist = []
            beads = self.eos_dict['beads']
            crosslibrary = self.eos_dict["crosslibrary"]
            for (i, beadname) in enumerate(beads):
                if beadname in crosslibrary:
                    for (j, beadname2) in enumerate(beads):
                        if beadname2 in crosslibrary[beadname]:
                            crosslist.append([i, j])

            for i in range(np.size(crosslist, axis=0)):
                a = crosslist[i][0]
                b = crosslist[i][1]
                if beads[a] in crosslibrary:
                    if beads[b] in crosslibrary[beads[a]]:
                        if "epsilon" in crosslibrary[beads[a]][beads[b]]:
                            epsilon[a, b] = crosslibrary[beads[a]][beads[b]]["epsilon"]
                            epsilon[b, a] = epsilon[a, b]
                        if "lambda" in crosslibrary[beads[a]][beads[b]]:
                            lambda_tmp[a, b] = crosslibrary[beads[a]][beads[b]]["lambda"]
                            lambda_tmp[b, a] = epsilon[a, b]

        self.eos_dict["epsilon"+tmp] = epsilon 
        self.eos_dict["lambda"+tmp] = lambda_tmp

    def calc_component_averaged_properties(self):
        r"""
        
        Attributes
        ----------
        output : dict
            Dictionary of outputs, the following possibilities aer calculated if all relevant beads have those properties.
    
            - epsilon_ij : numpy.ndarray, Matrix of well depths for groups (k,l)
            - sigma_ij : numpy.ndarray, Matrix of Mie diameter for groups (k,l)
            - lambda_ij : numpy.ndarray, Matrix of Mie potential attractive exponents for k,l groups
    
        """
    
        ncomp, nbeads = np.shape(self.eos_dict['nui'])
        zki = np.zeros((ncomp, nbeads), float)
        zkinorm = np.zeros(ncomp, float)
    
        epsilonii = np.zeros(ncomp, float)
        sigmaii = np.zeros(ncomp, float)
        lambdaii = np.zeros(ncomp, float)
    
        #compute zki
        for i in range(ncomp):
            for k in range(nbeads):
                zki[i, k] = self.eos_dict['nui'][i, k] * self.eos_dict['Vks'][k] * self.eos_dict['Sk'][k]
                zkinorm[i] += zki[i, k]
    
        for i in range(ncomp):
            for k in range(nbeads):
                zki[i, k] = zki[i, k] / zkinorm[i]
    
        for i in range(ncomp):
            for k in range(nbeads):
                sigmaii[i] += zki[i, k] * self.eos_dict['sigma_kl'][k, k]**3
                for l in range(nbeads):

                    epsilonii[i] += zki[i, k] * zki[i, l] * self.eos_dict['epsilon_kl'][k, l]
                    lambdaii[i] += zki[i, k] * zki[i, l] * self.eos_dict['lambda_kl'][k, l]
            sigmaii[i] = sigmaii[i]**(1.0/3.0)

        input_dict = {"sigma": sigmaii}
        output_dict = stb.cross_interaction_from_array( input_dict, self.mixing_rules)
        self.eos_dict["sigma_ij"] = output_dict['sigma']

        self.eos_dict["lambda_ij"] = np.diagflat(lambdaii)
        self.eos_dict["epsilon_ij"] = np.diagflat(epsilonii)
        self.calc_sw_cross_interaction_parameters(mode="effective")

    def calc_Kklab(self, r_klab, ratio=0.25):
        r"""
        Calculate available bonding site volume from the square-well cut-off distance.
 
        Parameters
        ----------
        r_ij_klab : numpy.ndarray
            Matrix of cutt-off values (ncomp x ncomp x nbeads x nbeads)
        ratio : float, Optional, default: 0.25
            This sets the reduced density of the sites, r_d/sigma_ij (effective sigma)

        Returns
        -------
        Kklab : numpy.ndarray
            Bonding volume for association sites. (ncomp x ncomp x nbeads x nbeads) 
        """

    #    sigma = self.eos_dict['sigma_ij']
    #    Kklab = np.zeros((self.ncomp,self.ncomp,self.nbeads,self.nbeads))
    #    for i in range(self.ncomp):
    #        for j in range(self.ncomp):
    #            r_d = ratio*sigma[i,j]
    #            for k in range(self.nbeads):
    #                for l in range(self.nbeads):
    #                    tmp = 4*np.pi*sigma[i,j]**2/(72.0*r_d**2)
    #                    tmp11 = np.log(r_klab[i,j,k,l])
    #                    tmp12 = 
    #                    tmp21 = 
    #                    tmp22 = 
    #                    Kklab[i,j,k,l] = tmp*(tmp11*tmp12+tmp21*tmp22)

    #    return Kklab

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
            Reduced density (len(rho), 4)
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        rho2 = rho * constants.molecule_per_nm3 * self.eos_dict['Cmol2seg']

        reduced_density = np.zeros((np.size(rho), 4))
        for m in range(4):
            reduced_density[:, m] = rho2 * (np.sum(np.sqrt(np.diag(self.eos_dict['xskl'])) * (np.diag(self.eos_dict['sigma_kl'])**m)) * (np.pi / 6.0))

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

        if mode == "normal":
            lambdakl = self.eos_dict["lambda_kl"]
        elif mode == "effective":
            lambdakl = self.eos_dict["lambda_ij"]
        lx = len(lambdakl) # lx is nbeads for normal and ncomp for effective

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        zetax_pow = np.zeros((np.size(rho), 3))
        zetax_pow[:, 0] = zetax
        for i in range(1, 3):
            zetax_pow[:, i] = zetax_pow[:, i - 1] * zetax_pow[:, 0]

        zetakl = np.zeros((np.size(rho), lx, lx))
        for k in range(lx):
            for l in range(lx):
                if lambdakl[k, l] != 0.0:
                    cikl = np.dot(ckl_coef, np.array( (1.0, lambdakl[k, l], lambdakl[k, l]**2), dtype=ckl_coef.dtype ))
                    zetakl[:, k, l] = np.dot( zetax_pow, cikl)

        return zetakl

    def _dzetaeff_dzetax(self, rho, xi, zetax=None, mode="normal"):
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

        if mode == "normal":
            lambdakl = self.eos_dict["lambda_kl"]
        elif mode == "effective":
            lambdakl = self.eos_dict["lambda_ij"]
        lx = len(lambdakl) # lx is nbeads for normal and ncomp for effective

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        zetax_pow = np.transpose(np.array([np.ones(len(rho)), 2*zetax, 3*zetax**2]))

        # check if you have more than 1 bead types
        dzetakl = np.zeros((np.size(rho), lx, lx))
        for k in range(lx):
            for l in range(lx):
                if lambdakl[k, l] != 0.0:
                    cikl = np.dot(ckl_coef, np.array( (1.0, lambdakl[k, l], lambdakl[k, l]**2), dtype=ckl_coef.dtype ))
                    dzetakl[:, k, l] = np.dot( zetax_pow, cikl)

        return dzetakl

        
    def Ahard_sphere(self,rho, T, xi):
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

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zeta = self.reduced_density(rho, xi)

        tmp = (6.0 / (np.pi * rho * constants.molecule_per_nm3))
        tmp1 = np.log1p(-zeta[:, 3]) * (zeta[:, 2]**3 / (zeta[:, 3]**2) - zeta[:, 0])
        tmp2 = 3.0 * zeta[:, 2] / (1 - zeta[:, 3]) * zeta[:, 1]
        tmp3 = zeta[:, 2]**3 / (zeta[:, 3] * ((1.0 - zeta[:, 3])**2))
        AHS = tmp*(tmp1 + tmp2 + tmp3)

        #print("AHS",AHS)

        return AHS
    
    def Afirst_order(self,rho, T, xi, zetax=None):
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

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        g0HS = self.calc_g0HS(rho, xi, zetax=zetax)
        a1kl_tmp = np.tensordot(rho * constants.molecule_per_nm3, self.eos_dict['xskl']*self.alphakl, 0)
        A1 = -(self.eos_dict['Cmol2seg']**2 / T) * np.sum(a1kl_tmp * g0HS, axis=(1,2)) # Units of K

        #print("A1",A1)

        return A1

    def Asecond_order(self, rho, T, xi, zetax=None, KHS=None):
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
        
        self._check_density(rho)
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
    
    def Amonomer(self,rho, T, xi):
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

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        zetax = self.reduced_density(rho, xi)[:,3]

        Amonomer = self.Ahard_sphere(rho, T, xi) + self.Afirst_order(rho, T, xi, zetax=zetax) + self.Asecond_order(rho, T, xi, zetax=zetax)

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
        zetax : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
        mode : str, Optional, default: "normal"
            This indicates whether group or effective component parameters are used. Options include: "normal" and "effective", where normal used bead interaction matricies, and effective uses component averaged parameters.
        
        Returns
        -------
        g0HS : numpy.ndarray
            The contact value of the pair correlation function of a hypothetical pure fluid
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = self.reduced_density(rho, xi)[:,3]

        zeta_eff = self.effective_packing_fraction(rho, xi, mode=mode, zetax=zetax)

        g0HS = (1.0 - zeta_eff/2.0) / (1.0 - zeta_eff)**3

        return g0HS

    def calc_gHS(self, rho, xi):
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

        self._check_density(rho)
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

    def calc_gSW(self, rho, T, xi, zetax=None):
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

        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        self._check_density(rho)
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

    def Achain(self, rho, T, xi):
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

        gii = self.calc_gSW(rho, T, xi)
   
        #print("gii", gii)
        
        Achain = 0.0
        for i in range(self.ncomp):
            beadsum = -1.0
            for k in range(self.nbeads):
                beadsum += (self.eos_dict['nui'][i, k] * self.eos_dict["Vks"][k] * self.eos_dict["Sk"][k])
            Achain -= xi[i] * beadsum * np.log(gii[:, i,i])

        if np.any(np.isnan(Achain)):
            logger.error("Some Helmholtz values are NaN, check energy parameters.")

        #print("Achain",Achain)

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

    def calc_gr_assoc(self, rho, T, xi):
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
            A temperature-density polynomial correlation of the association integral for a Lennard−Jones monomer. This matrix is (len(rho) x Ncomp x Ncomp)
        """
        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)
    
        gSW = self.calc_gSW(rho, T, xi)

        return gSW

    def parameter_refresh(self, beadlibrary, crosslibrary):
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on _beadlibrary and _crosslibrary attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        self.eos_dict["beadlibrary"].update(beadlibrary)
        self.eos_dict["crosslibrary"].update(crosslibrary)

        # Update Non bonded matrices
        output = stb.cross_interaction_from_dict( self.eos_dict['beads'], self.eos_dict['beadlibrary'], self.mixing_rules, crosslibrary=self.eos_dict['crosslibrary'])
        self.eos_dict["sigma_kl"] = output["sigma"]
        self.calc_sw_cross_interaction_parameters()
        self.calc_component_averaged_properties()

        if not np.isnan(self.xi):
            self.eos_dict['Cmol2seg'], self.eos_dict['xskl'] = stb.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])

    def _check_density(self, rho):
        r"""
        This function checks the attritutes of the density array
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        """
        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
                raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

    def _check_composition_dependent_parameters(self, xi):
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
