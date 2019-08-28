"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import sys
import numpy as np

from . import constants
from . import gamma_mie_funcs as funcs
# Later this line will be in an abstract class file in this directory, and all versions of SAFT will reference it
from despasito.equations_of_state.interface import EOStemplate

# ________________ Saft Family ______________
# NoteHere: Insert SAFT family abstract class in this directory to clean up


class saft_gamma_mie(EOStemplate):

    """
    Initialize EOS with system component parameters so that methods may be used by thermodynamic calculations. All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    xi : list[float]
        Mole fraction of component, only relevant for parameter fitting
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        * epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        * sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        * mass: Bead mass [kg/mol]
        * l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        * l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        * Sk: :math:`S_{k}`, Shape parameter of group k
        * Vks: :math:`V_{k,s}`, Number of groups, k, in component
    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        * epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        * l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
    sitenames : list[str], Optional, default: []
        List of unique association sites used among components
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):

        # Self interaction parameters
        xi = kwargs['xi']
        self._nui = kwargs['nui']
        self._beads = kwargs['beads']
        self._beadlibrary = kwargs['beadlibrary']

        massi = np.zeros_like(xi)
        for i in range(np.size(xi)):
            for k in range(np.size(self._beads)):
                massi[i] += self._nui[i, k] * self._beadlibrary[self._beads[k]]["mass"]
        self._massi = massi

        # Cross interaction parameters
        if 'crosslibrary' in kwargs:
            crosslibrary = kwargs['crosslibrary']
        else:
            crosslibrary = {}

        epsilonkl, sigmakl, l_akl, l_rkl, Ckl = funcs.calc_interaction_matrices(self._beads,
                                                                                self._beadlibrary,
                                                                                crosslibrary=crosslibrary)

        self._epsilonkl = epsilonkl
        self._sigmakl = sigmakl
        self._l_akl = l_akl
        self._l_rkl = l_rkl
        self._Ckl = Ckl

        # Association sites
        if 'sitenames' in kwargs:
            self._sitenames = kwargs['sitenames']
        else:
            self._sitenames = []

        epsilonHB, Kklab, nk = funcs.calc_assoc_matrices(self._beads,
                                                         self._beadlibrary,
                                                         sitenames=self._sitenames,
                                                         crosslibrary=crosslibrary)
        self._epsilonHB = epsilonHB
        self._Kklab = Kklab
        self._nk = nk
        
        # Initialize temperature attribute
        self.T = np.nan

    def _temp_dependent_variables(self, T):

        """
        Temperature dependent variables are initialized or updated according to the provided temperature.
    
        Parameters
        ----------
        T : float, default: numpy.nan
            Temperature of the system [K]
    
        Attributes
        ----------
        T : float, default: numpy.nan
            Temperature of the system
        """

        dkk, dkl, x0kl = funcs.calc_hard_sphere_matricies(self._beads, self._beadlibrary, self._sigmakl, T)
        self.T = T
        self._dkk = dkk
        self._dkl = dkl
        self._x0kl = x0kl

    def _xi_dependent_variables(self, xi):

        """
        Variables dependent on composition are initialized or updated according to the provided mole fractions.
    
        Parameters
        ----------
        xi : list[float]
            Mole fraction of component, only relevant for parameter fitting

        """

        Cmol2seg, xsk, xskl = funcs.calc_composition_dependent_variables(xi, self._nui, self._beads, self._beadlibrary)
        self._Cmol2seg = Cmol2seg
        self._xsk = xsk
        self._xskl = xskl

    def P(self, rho, T, xi):
        """
        Compute pressure given system information
       
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [molecules/m^3]
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
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)
        
        if type(rho) != np.ndarray:
            rho = np.array(rho)

        step = np.sqrt(np.finfo(float).eps) * rho * 10000.0
        # Decreasing step size by 2 orders of magnitude didn't reduce noise in P values
        nrho = np.size(rho)

        # computer rho+step and rho-step for better a bit better performance
        A = funcs.calc_A(np.append(rho + step, rho - step), xi, T, self._beads, self._beadlibrary, self._massi, self._nui, self._Cmol2seg, self._xsk, self._xskl, self._dkk, self._epsilonkl, self._sigmakl, self._dkl, self._l_akl, self._l_rkl, self._Ckl,self._x0kl, self._epsilonHB, self._Kklab, self._nk)
        
        P_tmp = (A[:nrho]-A[nrho:])*((constants.kb*T)/(2.0*step))*(rho**2)
        
        return P_tmp

    def chemicalpotential(self, P, rho, xi, T):
        """
        Compute pressure given system information
      
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
        mui : numpy.ndarray
            Array of chemical potential values for each component
        """

        if T != self.T:
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)

        daresdxi = np.zeros_like(xi)
        mui = np.zeros_like(xi)
        nmol = 1.0
        dnmol = 1.0E-4

        # compute mui
        for i in range(np.size(mui)):
            dAres = np.zeros(2)
            ares = funcs.calc_Ares(rho * constants.Nav, xi, T, self._beads, self._beadlibrary, self._massi, self._nui, self._Cmol2seg, self._xsk, self._xskl,self._dkk, self._epsilonkl, self._sigmakl, self._dkl, self._l_akl, self._l_rkl, self._Ckl, self._x0kl, self._epsilonHB, self._Kklab, self._nk)
            for j, delta in enumerate((dnmol, -dnmol)):
                xi_temp = np.copy(xi)
                xi_temp[i] += delta
                Cmol2seg_tmp, xsk_tmp, xskl_tmp = funcs.calc_composition_dependent_variables(xi_temp, self._nui, self._beads, self._beadlibrary)
                # xi_temp/=(nmol+delta)
                dAres[j] = funcs.calc_Ares(rho * constants.Nav, xi_temp, T, self._beads, self._beadlibrary, self._massi, self._nui, Cmol2seg_tmp, xsk_tmp, xskl_tmp, self._dkk, self._epsilonkl, self._sigmakl, self._dkl, self._l_akl, self._l_rkl, self._Ckl, self._x0kl, self._epsilonHB, self._Kklab, self._nk)
            daresdxi[i] = (dAres[0] - dAres[1]) / (2.0 * dnmol)

        # compute Z
        Z = P / (rho * T * constants.Nav * constants.kb)
        xjdaresdxj = np.sum(xi * daresdxi)
        for i in range(np.size(mui)):
            mui[i] = ares + Z - 1.0 + daresdxi[i] - xjdaresdxj - np.log(Z)
        return mui

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

        if T != self.T:
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        maxrho = maxpack * 6.0 / (self._Cmol2seg * np.pi * np.sum(self._xskl * (self._dkl**3))) / constants.Nav
        return maxrho

    def __str__(self):

        string = "Beads:" + str(self._beads) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string

