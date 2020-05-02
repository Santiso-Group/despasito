# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging

import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)

import logging # NoteHere
import numpy as np
from scipy import misc
from scipy import integrate
import scipy.optimize as spo
#import matplotlib.pyplot as plt
import os
import sys
#np.set_printoptions(threshold=sys.maxsize)

logger = logging.getLogger(__name__)

# Check for Numba
if 'NUMBA_DISABLE_JIT' in os.environ:
    disable_jit = os.environ['NUMBA_DISABLE_JIT']
else:
    from ... import jit_stat
    disable_jit = jit_stat.disable_jit

if disable_jit:
    from ..compiled_modules.nojit_exts import calc_a1s, calc_Bkl, calc_da1iidrhos, calc_da2ii_1pchi_drhos, prefactor
else:
    from ..compiled_modules.jit_exts import calc_a1s
    from ..compiled_modules.nojit_exts import calc_Bkl, calc_da1iidrhos, calc_da2ii_1pchi_drhos, prefactor

# Check for cython
from ... import cython_stat
disable_cython = cython_stat.disable_cython
if not disable_cython:
    if not disable_jit:
        logger.warning("Flag for Numba and Cython were given. Using Numba")
    else:
        from ..compiled_modules.c_exts import calc_a1s
        from ..compiled_modules.nojit_exts import calc_Bkl, calc_da1iidrhos, calc_da2ii_1pchi_drhos, prefactor

class Achain():

    r"""
    
    
    Parameters
    ----------

        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
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

        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Vks'):
            self.eos_dict['Vks'] = tb.extract_property("Vks",self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Sk'):
            self.eos_dict['Sk'] = tb.extract_property("Sk",self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # Initialize temperature attribute
        if not hasattr(self, 'T'):
            self.T = np.nan
        if not hasattr(self, 'xi'):
            self.xi = np.nan

        if not hasattr(self, 'nbeads'):
            self.ncomp, self.nbeads = np.shape(self.eos_dict['nui']) 

        # Initiate cross interaction terms
        if not all([hasattr(self, key) for key in ['epsilonkl','sigmakl','l_akl','l_rkl']]):
            self.eos_dict.update(tb.calc_interaction_matrices(self.eos_dict['beads'],self.eos_dict['beadlibrary'],crosslibrary=self.eos_dict['crosslibrary']))

        # Initiate average interaction terms
        if not all([hasattr(self, key) for key in ['sigmaii_avg', 'epsilonii_avg', 'l_rii_avg', 'l_aii_avg']]):
            self.eos_dict.update(tb.calc_component_averaged_properties(self.eos_dict['nui'], self.eos_dict['Vks'],self.eos_dict['Sk'],epsilonkl=self.eos_dict['epsilonkl'], sigmakl=self.eos_dict['sigmakl'], l_akl=self.eos_dict['l_akl'], l_rkl=self.eos_dict['l_rkl']))

    def gdHS(self, rho, T, xi, zetax=None):
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

        self._check_density(rho)
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])

        km = np.zeros((np.size(rho), 4))
        gdHS = np.zeros((np.size(rho), np.size(xi)))
        
        km[:, 0] = -np.log(1.0 - zetax) + (42.0 * zetax - 39.0 * zetax**2 + 9.0 * zetax**3 - 2.0 * zetax**4) / (6.0 * (1.0 - zetax)**3)
        km[:, 1] = (zetax**4 + 6.0 * zetax**2 - 12.0 * zetax) / (2.0 * (1.0 - zetax)**3)
        km[:, 2] = -3.0 * zetax**2 / (8.0 * (1.0 - zetax)**2)
        km[:, 3] = (-zetax**4 + 3.0 * zetax**2 + 3.0 * zetax) / (6.0 * (1.0 - zetax)**3)

        for i in range(self.ncomp):
            gdHS[:, i] = np.exp(km[:, 0] + km[:, 1] * self.eos_dict['x0ii'][i] + km[:, 2] * self.eos_dict['x0ii'][i]**2 + km[:, 3] * self.eos_dict['x0ii'][i]**3)
        
        return gdHS

    def g1(self, rho, T, xi, zetax=None):
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
        Asecond_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        self._check_density(rho)
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])

        da1iidrhos = calc_da1iidrhos(rho, self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['l_aii_avg'], self.eos_dict['l_rii_avg'], self.eos_dict['x0ii'], self.eos_dict['epsilonii_avg'], zetax)

        a1sii_l_aii_avg = calc_a1s(rho, self.eos_dict['Cmol2seg'], self.eos_dict['l_aii_avg'], zetax, self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'])
        a1sii_l_rii_avg = calc_a1s(rho, self.eos_dict['Cmol2seg'], self.eos_dict['l_rii_avg'], zetax, self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'])

        Bii_l_aii_avg = calc_Bkl(rho, self.eos_dict['l_aii_avg'], self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['epsilonii_avg'], self.eos_dict['x0ii'], zetax)
        Bii_l_rii_avg = calc_Bkl(rho, self.eos_dict['l_rii_avg'], self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['epsilonii_avg'], self.eos_dict['x0ii'], zetax)
    
        Cii = prefactor(self.eos_dict['l_rii_avg'], self.eos_dict['l_aii_avg'])

        g1 = (1.0 / (2.0 * np.pi * self.eos_dict['epsilonii_avg'] * self.eos_dict['dii_eff']**3 * constants.molecule_per_nm3**2)) * (3.0 * da1iidrhos - Cii * self.eos_dict['l_aii_avg'] * (self.eos_dict['x0ii']**self.eos_dict['l_aii_avg']) * np.einsum("ij,i->ij", (a1sii_l_aii_avg + Bii_l_aii_avg), 1.0 / (rho * self.eos_dict['Cmol2seg'])) + (Cii * self.eos_dict['l_rii_avg'] *  (self.eos_dict['x0ii']**self.eos_dict['l_rii_avg'])) * np.einsum("ij,i->ij", (a1sii_l_rii_avg + Bii_l_rii_avg), 1.0 / (rho * self.eos_dict['Cmol2seg'])))

        return g1
    
    def g2(self, rho, T, xi, zetax=None):
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
        Athird_order : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """

        self._check_density(rho)
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)

        if zetax is None:
            zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])
        zetaxstar = tb.calc_zetaxstar(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['sigmakl'])
        KHS = tb.calc_KHS(zetax)
        
        Cii = prefactor(self.eos_dict['l_rii_avg'], self.eos_dict['l_aii_avg'])
        
        phi7 = np.array([10.0, 10.0, 0.57, -6.7, -8.0])
        alphaii = Cii * ((1.0 / (self.eos_dict['l_aii_avg'] - 3.0)) - (1.0 / (self.eos_dict['l_rii_avg'] - 3.0)))
        theta = np.exp(self.eos_dict['epsilonii_avg'] / constants.kb / T) - 1.0
        
        gammacii = np.zeros((np.size(rho), np.size(xi)))
        for i in range(self.ncomp):
            gammacii[:, i] = phi7[0] * (-np.tanh(phi7[1] * (phi7[2] - alphaii[i])) + 1.0) * zetaxstar * theta[i] * np.exp(phi7[3] * zetaxstar + phi7[4] * (zetaxstar**2))
        
        da2iidrhos = calc_da2ii_1pchi_drhos(rho, self.eos_dict['Cmol2seg'], self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'], self.eos_dict['x0ii'], self.eos_dict['l_rii_avg'], self.eos_dict['l_aii_avg'], zetax)
        
        a1sii_2l_aii_avg = calc_a1s(rho, self.eos_dict['Cmol2seg'], 2.0 * self.eos_dict['l_aii_avg'], zetax, self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'])
        a1sii_2l_rii_avg = calc_a1s(rho, self.eos_dict['Cmol2seg'], 2.0 * self.eos_dict['l_rii_avg'], zetax, self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'])
        a1sii_l_rii_avgl_aii_avg = calc_a1s(rho, self.eos_dict['Cmol2seg'], self.eos_dict['l_aii_avg'] + self.eos_dict['l_rii_avg'], zetax, self.eos_dict['epsilonii_avg'], self.eos_dict['dii_eff'])
        
        Bii_2l_aii_avg = calc_Bkl(rho, 2.0 * self.eos_dict['l_aii_avg'], self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['epsilonii_avg'], self.eos_dict['x0ii'], zetax)
        Bii_2l_rii_avg = calc_Bkl(rho, 2.0 * self.eos_dict['l_rii_avg'], self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['epsilonii_avg'], self.eos_dict['x0ii'], zetax)
        Bii_l_aii_avgl_rii_avg = calc_Bkl(rho, self.eos_dict['l_aii_avg'] + self.eos_dict['l_rii_avg'], self.eos_dict['Cmol2seg'], self.eos_dict['dii_eff'], self.eos_dict['epsilonii_avg'], self.eos_dict['x0ii'], zetax)
        
        eKC2 = np.einsum("i,j->ij", KHS / rho / self.eos_dict['Cmol2seg'], self.eos_dict['epsilonii_avg'] * (Cii**2))
        
        g2MCA = (1.0 / (2.0 * np.pi * (self.eos_dict['epsilonii_avg']**2) * self.eos_dict['dii_eff']**3 * constants.molecule_per_nm3**2)) * ((3.0 * da2iidrhos) \
                - (eKC2 * self.eos_dict['l_rii_avg'] * (self.eos_dict['x0ii']**(2.0 * self.eos_dict['l_rii_avg']))) \
                   * (a1sii_2l_rii_avg + Bii_2l_rii_avg) \
                + eKC2 * (self.eos_dict['l_rii_avg'] + self.eos_dict['l_aii_avg']) \
                   * (self.eos_dict['x0ii']**(self.eos_dict['l_rii_avg'] + self.eos_dict['l_aii_avg'])) * (a1sii_l_rii_avgl_aii_avg + \
                Bii_l_aii_avgl_rii_avg) \
                - eKC2 * self.eos_dict['l_aii_avg'] * (self.eos_dict['x0ii']**(2.0 * self.eos_dict['l_aii_avg'])) \
                   * (a1sii_2l_aii_avg + Bii_2l_aii_avg))

        g2 = (1.0 + gammacii) * g2MCA
        
        return g2
    
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

        self._check_density(rho)
        self._check_temerature_dependent_parameters(T)
        self._check_composition_dependent_parameters(xi)
        kT = T * constants.kb

        zetax = tb.calc_zetax(rho, self.eos_dict['Cmol2seg'], self.eos_dict['xskl'], self.eos_dict['dkl'])
        gdHS = self.gdHS(rho, T, xi, zetax=zetax)
        g1 = self.g1(rho, T, xi, zetax=zetax)
        g2 = self.g2(rho, T, xi, zetax=zetax)

        gii = gdHS * np.exp((self.eos_dict['epsilonii_avg'] * g1 / (kT * gdHS)) + (((self.eos_dict['epsilonii_avg'] / kT)**2) * g2 / gdHS))
        
        Achain = 0.0
        for i in range(self.ncomp):
            beadsum = -1.0
            for k in range(self.nbeads):
                beadsum += (self.eos_dict['nui'][i, k] * self.eos_dict["Vks"][k] * self.eos_dict["Sk"][k])
            Achain -= xi[i] * beadsum * np.log(gii[:, i])
        
        if np.any(np.isnan(Achain)):
            logger.error("Some Helmholtz values are NaN, check energy parameters.")

        return Achain
    
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
            self._update_chain_temperature_dependent_variables(T)
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

    def _update_chain_temperature_dependent_variables(self, T):
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        T : float
            Temperature of the system [K]
        
        Atributes
        ---------
        eos_dict : dict
            The following entries are updated: dii_eff, x0ii
        """
        
        zki = np.zeros((self.ncomp, self.nbeads), float)
        zkinorm = np.zeros(self.ncomp, float)
        dii_eff = np.zeros((self.ncomp), float)
        #compute zki
        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = self.eos_dict['nui'][i, k] * self.eos_dict['Vks'][k] * self.eos_dict['Sk'][k]
                zkinorm[i] += zki[i, k]

        for i in range(self.ncomp):
            for k in range(self.nbeads):
                zki[i, k] = zki[i, k] / zkinorm[i]
        
        for i in range(self.ncomp):
            for k in range(self.nbeads):
                for l in range(self.nbeads):
                    dii_eff[i] += zki[i, k] * zki[i, l] * self.eos_dict['dkl'][k, l]**3
            dii_eff[i] = dii_eff[i]**(1/3.0)
        self.eos_dict['dii_eff'] = dii_eff

        #compute x0ii
        self.eos_dict['x0ii'] = self.eos_dict['sigmaii_avg']/dii_eff

    def __str__(self):

        string = "Beads: {}".format(self.eos_dict['beads'])
        return string


