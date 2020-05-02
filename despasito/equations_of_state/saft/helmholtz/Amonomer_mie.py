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

logger = logging.getLogger(__name__)

# Check for Numba
if 'NUMBA_DISABLE_JIT' in os.environ:
    disable_jit = os.environ['NUMBA_DISABLE_JIT']
else:
    from ... import jit_stat
    disable_jit = jit_stat.disable_jit

if disable_jit:
    from ..compiled_modules.nojit_exts import calc_a1s, calc_a1ii, calc_Bkl, prefactor
else:
    from ..compiled_modules.jit_exts import calc_a1s
    from ..compiled_modules.nojit_exts import calc_a1ii, calc_Bkl, prefactor

# Check for cython
from ... import cython_stat
disable_cython = cython_stat.disable_cython
if not disable_cython:
    if not disable_jit:
        logger.warning("Flag for Numba and Cython were given. Using Numba")
    else:
        from ..compiled_modules.c_exts import calc_a1s
        from ..compiled_modules.nojit_exts import calc_a1ii, calc_Bkl, prefactor

class Amonomer():

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
        
        needed_attributes = ['nui','beads','beadlibrary']
        
        if not hasattr(self, 'eos_dict'):
            self.eos_dict = {}
        
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                self.eos_dict[key] = kwargs[key]

        if 'crosslibrary' not in kwargs:
            self.eos_dict['crosslibrary'] = {}
        else:
            self.eos_dict['crosslibrary'] = kwargs['crosslibrary']

#################
        for key, value in self.eos_dict['beadlibrary'].items():
            self.eos_dict['beadlibrary'][key]["sigma"] = value["sigma"]*1e+9
#################

        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # Initialize temperature attribute
        if not hasattr(self, 'T'):
            self.T = np.nan
        if not hasattr(self, 'xi'):
            self.xi = np.nan

        if not hasattr(self, 'nbeads'):
            self.ncomp, self.nbeads = np.shape(self.eos_dict['nui'])
        
        # Intiate cross interaction terms
        if not all([hasattr(self, key) for key in ['epsilonkl','sigmakl','l_akl','l_rkl']]):
            self.eos_dict.update(tb.calc_interaction_matrices(self.eos_dict['beads'],self.eos_dict['beadlibrary'],crosslibrary=self.eos_dict['crosslibrary']))

        # compute alphakl eq. 33
        self.eos_dict['Ckl'] = prefactor(self.eos_dict['l_rkl'], self.eos_dict['l_akl'])
        self.eos_dict['alphakl'] = self.eos_dict['Ckl'] * ((1.0 / (self.eos_dict['l_akl'] - 3.0)) - (1.0 / (self.eos_dict['l_rkl'] - 3.0)))
    
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
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        eta = np.zeros((np.size(rho), 4))
        for m in range(4):
            eta[:, m] = rho * constants.molecule_per_nm3 * self.eos_dict['Cmol2seg'] * (np.sum(np.sqrt(np.diag(self.eos_dict['xskl'])) * (np.diag(self.eos_dict['dkl'])**m)) * (np.pi / 6.0))

        tmp = (6.0 / (np.pi * rho * constants.molecule_per_nm3))
        tmp1 = np.log1p(-eta[:, 3]) * (eta[:, 2]**3 / (eta[:, 3]**2) - eta[:, 0])
        tmp2 = 3.0 * eta[:, 2] / (1 - eta[:, 3]) * eta[:, 1]
        tmp3 = eta[:, 2]**3 / (eta[:, 3] * ((1.0 - eta[:, 3])**2))
        AHS = tmp*(tmp1 + tmp2 + tmp3)

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
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])

        # compute components of eq. 19
        a1kl = calc_a1ii(rho, self.eos_dict['Cmol2seg'], self.eos_dict['dkl'], self.eos_dict['l_akl'], self.eos_dict['l_rkl'], self.eos_dict['x0kl'], self.eos_dict['epsilonkl'], zetax)
    
        # eq. 18
        a1 = np.einsum("ijk,jk->i", a1kl, self.eos_dict['xskl'])
        A1 = (self.eos_dict['Cmol2seg'] / T) * a1 # Units of K

        return A1

    def Asecond_order(self, rho, T, xi, zetaxstar=None, zetax=None, KHS=None):
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
        zetaxstar : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on sigma for groups (k,l)
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
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])

        if zetaxstar is None:
            zetaxstar = tb.calc_zetaxstar(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['sigmakl'])

        if KHS is None:
            KHS = tb.calc_KHS(zetax)
        
        ## compute a2kl, eq. 30 #####
        
        # compute f1, f2, and f3 for eq. 32
        fmlist123 = self.calc_fm(self.eos_dict['alphakl'], np.array([1, 2, 3]))
    
        chikl = np.einsum("i,jk", zetaxstar, fmlist123[0]) + np.einsum("i,jk", zetaxstar**5, fmlist123[1]) + np.einsum("i,jk", zetaxstar**8, fmlist123[2])
        
        a1s_2la = calc_a1s(rho, self.eos_dict['Cmol2seg'], 2.0 * self.eos_dict['l_akl'], zetax, self.eos_dict['epsilonkl'], self.eos_dict['dkl'])
        a1s_2lr = calc_a1s(rho, self.eos_dict['Cmol2seg'], 2.0 * self.eos_dict['l_rkl'], zetax, self.eos_dict['epsilonkl'], self.eos_dict['dkl'])
        a1s_lalr = calc_a1s(rho, self.eos_dict['Cmol2seg'], self.eos_dict['l_akl'] + self.eos_dict['l_rkl'], zetax, self.eos_dict['epsilonkl'], self.eos_dict['dkl'])
        B_2la = calc_Bkl(rho, 2.0 * self.eos_dict['l_akl'], self.eos_dict['Cmol2seg'], self.eos_dict['dkl'], self.eos_dict['epsilonkl'], self.eos_dict['x0kl'], zetax)
        B_2lr = calc_Bkl(rho, 2.0 * self.eos_dict['l_rkl'], self.eos_dict['Cmol2seg'], self.eos_dict['dkl'], self.eos_dict['epsilonkl'], self.eos_dict['x0kl'], zetax)
        B_lalr = calc_Bkl(rho, self.eos_dict['l_akl'] + self.eos_dict['l_rkl'], self.eos_dict['Cmol2seg'], self.eos_dict['dkl'], self.eos_dict['epsilonkl'], self.eos_dict['x0kl'], zetax)
        
        a2kl = (self.eos_dict['x0kl']**(2.0 * self.eos_dict['l_akl'])) * (a1s_2la + B_2la) / constants.molecule_per_nm3 - ((2.0 * self.eos_dict['x0kl']**(self.eos_dict['l_akl'] + self.eos_dict['l_rkl'])) * (a1s_lalr + B_lalr) / constants.molecule_per_nm3) + ((self.eos_dict['x0kl']**(2.0 * self.eos_dict['l_rkl'])) * (a1s_2lr + B_2lr) / constants.molecule_per_nm3)
        a2kl *= (1.0 + chikl) * self.eos_dict['epsilonkl'] * (self.eos_dict['Ckl']**2)  # *(KHS/2.0)
        a2kl = np.einsum("i,ijk->ijk", KHS / 2.0, a2kl)
        
        # eq. 29
        a2 = np.einsum("ijk,jk->i", a2kl, self.eos_dict['xskl'])
        A2 = (self.eos_dict['Cmol2seg'] / (T**2)) * a2
                
        return A2
    
    def Athird_order(self,rho, T, xi, zetaxstar=None):
        r"""
        Outputs :math:`A^{3rd order}`. This is the third order term in the high-temperature perturbation expansion
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        zetaxstar : numpy.ndarray, Optional, default: None
            Matrix of hypothetical packing fraction based on sigma for groups (k,l)
        
        Returns
        -------
        Athird_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        if zetaxstar is None:
            zetaxstar = tb.calc_zetaxstar(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['sigmakl'])
        
        # compute a3kl
        fmlist456 = self.calc_fm(self.eos_dict['alphakl'], np.array([4, 5, 6]))

        a3kl = np.einsum("i,jk", zetaxstar, -(self.eos_dict['epsilonkl']**3) * fmlist456[0]) * np.exp( np.einsum("i,jk", zetaxstar, fmlist456[1]) + np.einsum("i,jk", zetaxstar**2, fmlist456[2])) # a3kl=-(epsilonkl**3)*fmlist456[0]*zetaxstar*np.exp((fmlist456[1]*zetaxstar)+(fmlist456[2]*(zetaxstar**2)))

        # eq. 37
        a3 = np.einsum("ijk,jk->i", a3kl, self.eos_dict['xskl'])
        A3 = (self.eos_dict['Cmol2seg'] / (T**3)) * a3
                
        return A3
    
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
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])
        zetaxstar = tb.calc_zetaxstar(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['sigmakl'])
        
        Amonomer = self.Ahard_sphere(rho, T, xi) + self.Afirst_order(rho, T, xi, zetax=zetax) + self.Asecond_order(rho, T, xi, zetax=zetax, zetaxstar=zetaxstar) + self.Athird_order(rho, T, xi, zetaxstar=zetaxstar)
        
        return Amonomer

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

        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        maxrho = maxpack * 6.0 / (self.eos_dict['Cmol2seg'] * np.pi * np.sum(self.eos_dict['xskl'] * (self.eos_dict['dkl']**3))) / constants.molecule_per_nm3
        return maxrho

    @staticmethod
    def calc_fm(alphakl, mlist):
        r"""
        Calculate fm(alphakl) where a list of m values are specified in mlist eq. 39
        
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
            fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0), np.size(alphakl, axis=0)))
        elif np.size(np.shape(alphakl)) == 1:
            fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0)))
        else:
            logger.error('Unexpected shape in calc_fm')
        mlist = mlist - 1

        phimn=np.array([[  7.53655570e+00,  -3.76046300e+01,   7.17459530e+01,  -4.68355200e+01, -2.46798200e+00,  -5.02720000e-01,   8.09568830e+00],\
                [ -3.59440000e+02,   1.82560000e+03,  -3.16800000e+03,   1.88420000e+03, -8.23760000e-01,  -3.19350000e+00,   3.70900000e+00],\
                [  1.55090000e+03,  -5.07010000e+03,   6.53460000e+03,  -3.28870000e+03, -2.71710000e+00,   2.08830000e+00,   0.00000000e+00],\
                [ -1.19932000e+00,   9.06363200e+00,  -1.79482000e+01,   1.13402700e+01,  2.05214200e+01,  -5.66377000e+01,   4.05368300e+01],\
                [ -1.91128000e+03,   2.13901750e+04,  -5.13207000e+04,   3.70645400e+04,  1.10374200e+03,  -3.26461000e+03,   2.55618100e+03],\
                [  9.23690000e+03,  -1.29430000e+05,   3.57230000e+05,  -3.15530000e+05,  1.39020000e+03,  -4.51820000e+03,   4.24160000e+03],\
                [  1.00000000e+01,   1.00000000e+01,   5.70000000e-01,  -6.70000000e+00, -8.00000000e+00,   0.00000000e+00,   0.00000000e+00]])
        
        for i, m in enumerate(mlist):
            for n in range(4):
                fmlist[i] += phimn[m, n] * (alphakl**n)
            dum = np.ones_like(fmlist[i])
            for n in range(4, 7):
                dum += phimn[m, n] * (alphakl**(n - 3.0))
            fmlist[i] = fmlist[i] / dum
    
        return fmlist
    
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

    def _check_temerature_dependent_parameters(self, T):
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        T : float
            Temperature of the system [K]
            
        Atributes
        ---------
        eos_dict : dict
            The following entries are updated: dkl, x0kl
        """
        if self.T != T:
            self.eos_dict['dkl'], self.eos_dict['x0kl'] = tb.calc_hard_sphere_matricies(T, self.eos_dict['sigmakl'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])
            self.T = T

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
        
            self.eos_dict['Cmol2seg'], self.eos_dict['xskl'] = tb.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])
            self.xi = xi

    def __str__(self):

        string = "Beads: {}".format(self.eos_dict['beads'])
        return string


