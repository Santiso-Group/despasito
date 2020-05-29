r"""
    Routines for calculating the Helmholtz energy for the SAFT-gamma equation of state.
    
    Equations referenced in this code are from V. Papaioannou et al. J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import sys

from despasito.equations_of_state import constants
#from profilehooks import profile

logger = logging.getLogger(__name__)

#@profile
def calc_a1s(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return a1s,kl(rho*Cmol2seg,l_kl) in K as defined in eq. 25.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_kl : numpy.ndarray
        Matrix of mie potential exponents for k,l groups
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)

    Returns
    -------
    a1s : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

    nbeads = np.size(dkl, axis=0)
    zetax_pow = np.zeros((np.size(rho), 4))
    zetax_pow[:, 0] = zetax
    for i in range(1, 4):
        zetax_pow[:, i] = zetax_pow[:, i - 1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                cikl = np.dot(constants.ckl_coef, np.array( (1.0, 1.0/l_kl[k, l], 1.0/l_kl[k, l]**2, 1.0/l_kl[k, l]**3), dtype=constants.ckl_coef.dtype ))
                etakl[:, k, l] = np.dot( zetax_pow, cikl)
        a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0))
        
        # a1s is 4D matrix
        a1s = np.transpose(np.transpose(a1s) * rho) # {BottleNeck}

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl[:, k] = np.dot(zetax_pow, cikl)
        a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0))

        # a1s is 3D matrix
        a1s = np.transpose(np.transpose(a1s) * rho) # {BottleNeck}
    else:
        print('Error in calc_a1s, unexpected array size')

    return a1s

#@profile
def calc_da1sii_drhos(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return da1s,kl(rho*Cmol2seg,l_kl)/rhos in K.
    
    Used in the calculation of :math:`A_chain` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_kl : numpy.ndarray
        Matrix of mie potential exponents for k,l groups
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)

    Returns
    -------
    da1s_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

    nbeads = np.size(dkl, axis=0)
    zetax_pow = np.zeros((np.size(rho), 4))
    zetax_pow[:, 0] = zetax
    for i in range(1, 4):
        zetax_pow[:, i] = zetax_pow[:, i - 1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                # Constants to calculate eta_eff
                cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3])))
                etakl[:, k, l] = np.dot(zetax_pow, cikl)
                rhos_detakl_drhos[:, k, l] = np.dot(zetax_pow, cikl*np.array([1.0,2.0,3.0,4.0]))
  # NoteHere
        da1s_drhos = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos, -2.0 * np.pi * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0)))
        sys.exit()

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl[:, k] = np.dot(zetax_pow, cikl)
            rhos_detakl_drhos[:, k] = np.dot(zetax_pow, cikl*np.array([1.,2.,3.,4.]))

        tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
        tmp2 = -2.0 * np.pi * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0))
        da1s_drhos = tmp1 * tmp2
    else:
        print('Error in calc_da1s_drhos, unexpected array size')

    return da1s_drhos

def calc_Bkl(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax):
    r"""
    Return Bkl(rho*Cmol2seg,l_kl) in K as defined in eq. 20.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    l_kl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential exponents for k,l groups
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, ratio of Mie radius for groups scaled by hard sphere interaction (k,l)
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    Bkl : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is rho x l_kl.shape
    
    """
    
    rhos = Cmol2seg * rho
    
    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))

    if len(np.shape(l_kl)) == 2:
        Bkl = np.tensordot(rhos * (2.0 * np.pi), (dkl**3 * constants.molecule_per_nm3**2) * epsilonkl, 0) *(np.tensordot( (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl, 0) - np.tensordot((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3)), Jkl, 0))
    else:
        Bkl = np.tensordot(rhos * (2.0 * np.pi), (dkl**3 * constants.molecule_per_nm3**2) * epsilonkl, 0) *(np.tensordot( (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl, 0) - np.tensordot((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3)), Jkl, 0))
#        Bkl = rhos * (2.0 * np.pi) * (dkl**3 * constants.molecule_per_nm3**2) * epsilonkl *((1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3) * Ikl - (9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))*Jkl)

    return Bkl

def calc_dBkl_drhos(l_kl, dkl, epsilonkl, x0kl, zetax):
    r"""
    Return derivative of Bkl(rho*Cmol2seg,l_kl) with respect to :math:`\rho_S`.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.
    
    Parameters
    ----------
    l_kl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential exponents for k,l groups
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, ratio of Mie radius for groups scaled by hard sphere interaction (k,l)
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    dBkl_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is rho x l_kl.shape
    
    """

    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))

    tmp = 2.0 * np.pi * dkl**3 * epsilonkl
    tmp1 = np.einsum("i,j", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,j", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl)
    tmp2 = np.einsum("i,j", (5.0 - 2.0*zetax) * zetax / (2*(1.0 - zetax)**4), Ikl) - np.einsum("i,j", ((9.0 * zetax * (zetax**2 + 4.0*zetax + 1)) / (2.0 * ((1 - zetax)**4))), Jkl)
    dBkl_drhos = tmp*(tmp1 + tmp2) * constants.molecule_per_nm3**2

    return dBkl_drhos

def calc_a1ii(rho, Cmol2seg, dii_eff, l_aii_avg, l_rii_avg, x0ii, epsilonii_avg, zetax):
    r"""
    Calculate effective first-order perturbation term :math:`\bar{a}_{1,ii}`.
    
    Used for the contribution of the monomeric interactions to the free energy per segment.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dii_eff : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    l_aii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    l_rii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    x0ii : numpy.ndarray
        Matrix of sigmaii_avg/dii_eff
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    a1ii : numpy.ndarray
        Matrix used in the calculation of the radial distribution function of a hypothetical one-fluid Mie system.
    """
    
    Cii = prefactor(l_rii_avg, l_aii_avg)
    
    Bii_r = calc_Bkl(rho, l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_a = calc_Bkl(rho, l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    a1s_r = calc_a1s(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1s_a = calc_a1s(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)
    
    return (Cii * (((x0ii**l_aii_avg) / constants.molecule_per_nm3 * (a1s_a + Bii_a)) - ((x0ii**l_rii_avg) / constants.molecule_per_nm3 * (a1s_r + Bii_r))))

def calc_da1iidrhos(rho, Cmol2seg, dii_eff, l_aii_avg, l_rii_avg, x0ii, epsilonii_avg, zetax):
    
    r"""
    Compute derivative of the term, :math:`\bar{a}_{1,ii}` with respect to :math:`\rho_s`
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dii_eff : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    l_aii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    l_rii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    x0ii : numpy.ndarray
        Matrix of sigmaii_avg/dii_eff
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    da1iidrhos : numpy.ndarray
        Derivative of term with respect to segment density
    """
    
    Cii = prefactor(l_rii_avg, l_aii_avg)
    
    das1_drhos_a = calc_da1sii_drhos(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)
    das1_drhos_r = calc_da1sii_drhos(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    
    dB_drhos_a = calc_dBkl_drhos(l_aii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dB_drhos_r = calc_dBkl_drhos(l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)

    da1iidrhos = (Cii * (((x0ii**l_aii_avg) * (das1_drhos_a + dB_drhos_a)) - ((x0ii**l_rii_avg) * (das1_drhos_r + dB_drhos_r))))

    return da1iidrhos

def calc_da2ii_1pchi_drhos(rho, Cmol2seg, epsilonii_avg, dii_eff, x0ii, l_rii_avg, l_aii_avg, zetax):
    
    r"""
    Compute derivative of the term, :math:`\frac{\bar{a}_{2,ii}}{1+\bar{\chi}_{ii}}` with respect to :math:`\rho_s`.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    dii_eff : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    x0ii : numpy.ndarray
        Matrix of sigmaii_avg/dii_eff
    l_rii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    l_aii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    da2ii_1pchi_drhos : numpy.ndarray
        Term used in the calculation of the second-order term from the macroscopic compressibility
    
    """
    
    # NoteHere g2mca der_a2kl

    # Calculate terms and derivatives used in derivative chain rule
    KHS = ((1.0 - zetax)**4) / (1.0 + (4.0 * zetax) + (4.0 * (zetax**2)) - (4.0 * (zetax**3)) + (zetax**4))
    dKHS_drhos = (4.0*(zetax**2 - 5.0*zetax - 2.0)*(1.0 - zetax)**3)/(zetax**4 - 4.0*zetax**3 + 4.0*zetax**2 + 4.0*zetax + 1.0)**2 *(zetax/(rho*Cmol2seg))
    
    a1sii_2l_aii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)
    
    Bii_2l_aii_avg = calc_Bkl(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    
    da1sii_2l_aii_avg = calc_da1sii_drhos(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    da1sii_2l_rii_avg = calc_da1sii_drhos(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    da1sii_l_rii_avgl_aii_avg = calc_da1sii_drhos(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)
    
    dBii_2l_aii_avg = calc_dBkl_drhos(2.0 * l_aii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dBii_2l_rii_avg = calc_dBkl_drhos(2.0 * l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dBii_l_aii_avgl_rii_avg = calc_dBkl_drhos(l_aii_avg + l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    
    # Calculate Derivative
    Cii = prefactor(l_rii_avg, l_aii_avg)
    
    B = x0ii**(2.0 * l_aii_avg) * (a1sii_2l_aii_avg + Bii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (a1sii_l_rii_avgl_aii_avg + Bii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (a1sii_2l_rii_avg + Bii_2l_rii_avg)
    dA_B = np.transpose(np.transpose(0.5*epsilonii_avg*Cii**2 * B)*dKHS_drhos)
    
    dB = x0ii**(2.0 * l_aii_avg) * (da1sii_2l_aii_avg + dBii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (da1sii_l_rii_avgl_aii_avg + dBii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (da1sii_2l_rii_avg + dBii_2l_rii_avg)
    A_dB = np.transpose(np.transpose(0.5*epsilonii_avg*Cii**2 * dB)*KHS)
    
    da2ii_1pchi_drhos = A_dB + dA_B
    
    return da2ii_1pchi_drhos

def prefactor(l_r, l_a):
    r"""
    Calculations C, the Mie potential prefactor, defined in eq. 2
    
    Parameters
    ----------
    l_a : float
        Mie potential attractive exponent
    l_r : float
        Mie potential attractive exponent
    
    Returns
    -------
    C : float
        Mie potential prefactor
    """
    if np.any(l_r == 0.0):
        sys.exit("end")

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))

def calc_Iij(rho, T, xi, epsilonii_avg, sigmaii_avg, sigmakl, xskl):
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
    epsilonii_avg : numpy.ndarray
        Array of average Mie diameters for component i
    sigmaii_avg : numpy.ndarray
        Array of average Mie diameters for component i
    sigmakl : numpy.ndarray
        Matrix of Mie diameter for groups (k,l)
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
    
    Returns
    -------
    Iij : numpy.ndarray
        A temperature-density polynomial correlation of the association integral for a Lennard−Jones monomer. This matrix is (len(rho) x Ncomp x Ncomp)
    
    """
    
    ncomp = len(xi)
    kT = T * constants.kb

    # compute epsilonij
    epsilonij = np.zeros((ncomp, ncomp))
    for i in range(ncomp):
        for j in range(i, ncomp):
            epsilonij[i, j] = np.sqrt(sigmaii_avg[i] * sigmaii_avg[j])**3.0 * np.sqrt(epsilonii_avg[i] * epsilonii_avg[j]) / (((sigmaii_avg[i] + sigmaii_avg[j]) / 2.0)**3)
            epsilonij[j, i] = epsilonij[i, j]

    sigmax3 = np.sum(xskl * (sigmakl**3 * constants.molecule_per_nm3))

    cij=np.array([[7.56425183020431e-02, -1.28667137050961e-01, 1.28350632316055e-01, -7.25321780970292e-02, 2.57782547511452e-02, -6.01170055221687e-03, 9.33363147191978e-04, -9.55607377143667e-05, 6.19576039900837e-06, -2.30466608213628e-07, 3.74605718435540e-09],\
                [1.34228218276565e-01, -1.82682168504886e-01, 7.71662412959262e-02, -7.17458641164565e-04, -8.72427344283170e-03, 2.97971836051287e-03, -4.84863997651451e-04, 4.35262491516424e-05, -2.07789181640066e-06, 4.13749349344802e-08, 0.00000000000000e+00],\
                [-5.65116428942893e-01, 1.00930692226792e+00, -6.60166945915607e-01, 2.14492212294301e-01, -3.88462990166792e-02, 4.06016982985030e-03, -2.39515566373142e-04, 7.25488368831468e-06, -8.58904640281928e-08, 0.00000000000000e+00, 0.00000000000000e+00],\
                [-3.87336382687019e-01, -2.11614570109503e-01, 4.50442894490509e-01, -1.76931752538907e-01, 3.17171522104923e-02, -2.91368915845693e-03, 1.30193710011706e-04, -2.14505500786531e-06, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [2.13713180911797e+00, -2.02798460133021e+00, 3.36709255682693e-01, 1.18106507393722e-03, -6.00058423301506e-03, 6.26343952584415e-04, -2.03636395699819e-05, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [-3.00527494795524e-01, 2.89920714512243e+00, -5.67134839686498e-01, 5.18085125423494e-02, -2.39326776760414e-03, 4.15107362643844e-05, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [-6.21028065719194e+00, -1.92883360342573e+00, 2.84109761066570e-01, -1.57606767372364e-02, 3.68599073256615e-04, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [1.16083532818029e+01, 7.42215544511197e-01, -8.23976531246117e-02, 1.86167650098254e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [-1.02632535542427e+01, -1.25035689035085e-01, 1.14299144831867e-02, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [4.65297446837297e+00, -1.92518067137033e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                [-8.67296219639940e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00]])

    Iij = np.zeros((np.size(rho), ncomp, ncomp))
    for p in range(11):
        for q in range(11 - p):
            #Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3 * rho)**p), ((kT / epsilonij)**q))
            if p == 0:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * np.ones(len(rho)), ((kT / epsilonij)**q))
            elif p == 1:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3 * rho)), ((kT / epsilonij)**q))
            elif p == 2:
                rho2 = rho**2
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho2)), ((kT / epsilonij)**q))
            elif p == 3:
                rho3 = rho2*rho
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho3)), ((kT / epsilonij)**q))
            elif p == 4:
                rho4 = rho2**2
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho4)), ((kT / epsilonij)**q))
            elif p == 5:
                rho5 = rho2*rho3
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho5)), ((kT / epsilonij)**q))
            elif p == 6:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho*rho5)), ((kT / epsilonij)**q))
            elif p == 7:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho2*rho5)), ((kT / epsilonij)**q))
            elif p == 8:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho3*rho5)), ((kT / epsilonij)**q))
            elif p == 9:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho4*rho5)), ((kT / epsilonij)**q))
            elif p == 10:
                Iij += np.einsum("i,jk->ijk", cij[p, q] * ((sigmax3**p * rho5*rho5)), ((kT / epsilonij)**q))

    return Iij

