# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import sys


from despasito.equations_of_state import constants
import despasito.equations_of_state.toolbox as tb

logger = logging.getLogger(__name__)

class Aideal():

    r"""
    Object for ideal contribution of helmholtz energy.
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
        - mass: Bead mass [kg/mol]
        
    Attributes
    ----------
    eos_dict : dict, default: keys = ['beadlibrary', 'beads', 'nui', 'massi']
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
        
        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])
    
    def Aideal(self, rho, T, xi):
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
    
        Returns
        -------
        Aideal : numpy.ndarray
            Helmholtz energy of ideal gas for each density given.
        """
        
        self._check_density(rho)

        rho2 = rho*constants.molecule_per_nm3
        
        xi_tmp, massi_tmp = tb.remove_insignificant_components( xi, self.eos_dict['massi'])
        
        # rhoi: (number of components,number of densities) number density of each component for each density
        
        rhoi = np.outer(rho2, xi_tmp)
        Lambda = np.sqrt( (constants.h*constants.Nav * constants.m2nm) * (constants.h / constants.kb * constants.m2nm) / (2.0 * np.pi * massi_tmp * T))
        log_broglie3_rho = np.log(Lambda**3*rhoi)

        #    if not any(np.sum(xi_tmp * np.log(Aideal_tmp), axis=1)):
        if np.isnan(np.sum(np.sum(xi_tmp * log_broglie3_rho, axis=1))):
            raise ValueError("Aideal has values of zero when taking the log. All mole fraction values should be nonzero. Mole fraction: {}".format(xi_tmp))
        else:
            Aideal = np.sum(xi_tmp * log_broglie3_rho, axis=1) - 1.0

        return Aideal
    
    @staticmethod
    def _check_density(rho):
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
        """
        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
            raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

    def __str__(self):

        string = "Beads: {}, Masses: {} kg/mol".format(self.eos_dict['beads'],eos_dict['massi'])
        return string


