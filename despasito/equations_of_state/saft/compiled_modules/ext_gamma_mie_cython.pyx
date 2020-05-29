
import numpy as np
cimport numpy as np
import logging
import os

from despasito.equations_of_state import constants

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
        print("l_r should not be zero.")

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))

def calc_a1s(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" wrapper function for calling 2d/3d versions of calc_a1s ... this is done for stupid Numba 
    """
    if len(l_kl.shape) == 2:
        output = calc_a1s_2d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)
    elif len(l_kl.shape) == 1:
        output = calc_a1s_1d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)

    return output

def calc_a1s_2d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return a1s,kl(rho*Cmol2seg,l_kl) in K as defined in eq. 25, used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_kl : numpy.ndarray
        Matrix of mie potential exponents for k,l groups
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)

    Returns
    -------
    numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups

    :note: output seems to be a tensor of size (N x Ngroups x Ngroups)
    """
    # Andrew: why is the 4 hard-coded here?
    nbeads = len(dkl)
    zetax_pow = np.empty((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.empty((len(rho), nbeads, nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        for l in range(nbeads):
            tmp = np.dot(constants.ckl_coef, np.array( (1.0, 1.0/l_kl[k, l], 1.0/l_kl[k, l]**2, 1.0/l_kl[k, l]**3), dtype=constants.ckl_coef.dtype ))
            etakl[:, k, l] = np.dot( zetax_pow, tmp )

    a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0))
    return np.transpose(np.transpose(a1s) * rho)

def calc_a1s_1d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return a1s,kl(rho*Cmol2seg,l_kl) in K as defined in eq. 25, used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_kl : numpy.ndarray
        Matrix of mie potential exponents for k,l groups
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)

    Returns
    -------
    numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """
    nbeads = len(dkl)
    zetax_pow = np.empty((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.empty((len(rho), nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        tmp = np.dot(constants.ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=constants.ckl_coef.dtype ) )
        etakl[:, k] = np.dot( zetax_pow, tmp )

    a1s = - (1.0 - (etakl / 2.0)) / (1.0 - etakl)**3 * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0) )

    return np.transpose(np.transpose(a1s) * rho)

#@profile
def calc_Bkl(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax):
    r""" Wrapper function for calling 2d/3d versions of calc_Bkl (this is done for Numba)
    """
    print("numba",np.shape(l_kl))

    if len(l_kl.shape) == 2:
        output = calc_Bkl_2d(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)
    elif len(l_kl.shape) == 1:
        output = calc_Bkl_1d(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)

    return output

def calc_Bkl_1d(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax):
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
    nbeads = len(dkl)

    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))

    tmp11 = rhos * (2.0 * np.pi)
    tmp12 = (dkl**3 * constants.molecule_per_nm3**2) * epsilonkl
    tmp2 = (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3)
    tmp3 = (9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))

    Bkl = np.zeros((len(rho), nbeads))
    for k in np.arange(nbeads):
        Bkl[:,k] = tmp11*tmp12[k]*(tmp2*Ikl[k] - tmp3*Jkl[k])
  #  Bkl = tmp11*tmp12*(tmp2*Ikl - tmp3*Jkl)

    return Bkl

def calc_Bkl_2d(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax):
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

    tmp11 = rhos * (2.0 * np.pi)
    tmp12 = (dkl**3 * constants.molecule_per_nm3**2) * epsilonkl
    tmp2 = (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3)
    tmp3 = (9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))

    lx = len(Ikl)
    Bkl = np.zeros((len(rho), lx, lx))
    for i in np.arange(lx):
        for j in np.arange(lx):
            Bkl[:,i,j] = tmp11*tmp12[i,j]*(tmp2*Ikl[i,j] - tmp3*Jkl[i,j])

    return Bkl

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

    Bii_r = calc_Bkl_2d(rho, l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_a = calc_Bkl_2d(rho, l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    a1s_r = calc_a1s_2d(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1s_a = calc_a1s_2d(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)

    return (Cii * (((x0ii**l_aii_avg) / constants.molecule_per_nm3 * (a1s_a + Bii_a)) - ((x0ii**l_rii_avg) / constants.molecule_per_nm3 * (a1s_r + Bii_r))))

def calc_da1sii_drhos(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return a1s,kl(rho*Cmol2seg,l_kl) in K as defined in eq. 25.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_kl : numpy.ndarray
        Matrix of mie potential exponents for k,l groups
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)

    Returns
    -------
    numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """
    nbeads = len(dkl)
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.zeros((len(rho), nbeads), dtype=rho.dtype)
    rhos_detakl_drhos = np.zeros((len(rho), nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        tmp = np.dot(constants.ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=constants.ckl_coef.dtype ) )
        tmp_dr = np.dot(constants.ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=constants.ckl_coef.dtype ) )*np.array((1.0,2.0,3.0,4.0))
        etakl[:, k] = np.dot( zetax_pow, tmp )
        rhos_detakl_drhos[:, k] = np.dot( zetax_pow, tmp_dr )

    tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
    tmp2 = - 2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    da1s_drhos = tmp1*tmp2

    #da1s_drhos = - 2.0 * np.pi * ((1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0 - 2.0*etakl)/(2.0*(1.0-etakl)**4)) * rhos_detakl_drhos * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    return da1s_drhos

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
    nbeads = len(dkl)
    nrho = len(zetax)

    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))

    tmp = 2.0 * np.pi * dkl**3 * epsilonkl
    tmp1 = np.zeros((nrho,nbeads))
    tmp2 = np.zeros((nrho,nbeads))
    for k in np.arange(nbeads):
        tmp1[:,k] = ((1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3) * Ikl[k]) - (((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))) * Jkl[k])
        tmp2[:,k] = ((5.0 - 2.0*zetax) * zetax / (2*(1.0 - zetax)**4) * Ikl[k]) - (((9.0 * zetax * (zetax**2 + 4.0*zetax + 1)) / (2.0 * ((1 - zetax)**4))) * Jkl[k])
    dBkl_drhos = tmp*(tmp1 + tmp2) * constants.molecule_per_nm3**2

    return dBkl_drhos

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

    a1sii_2l_aii_avg = calc_a1s_1d(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s_1d(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s_1d(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)

    Bii_2l_aii_avg = calc_Bkl_1d(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl_1d(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl_1d(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)

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
#    dA_B = np.einsum("i,ij->ij", dKHS_drhos, 0.5*epsilonii_avg*Cii**2 * B)

    dB = x0ii**(2.0 * l_aii_avg) * (da1sii_2l_aii_avg + dBii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (da1sii_l_rii_avgl_aii_avg + dBii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (da1sii_2l_rii_avg + dBii_2l_rii_avg)

    A_dB = np.transpose(np.transpose(0.5*epsilonii_avg*Cii**2 * dB)*KHS)
    #A_dB = np.einsum("i,ij->ij", KHS, 0.5*epsilonii_avg*Cii**2 * dB)

    da2ii_1pchi_drhos = A_dB + dA_B

    return da2ii_1pchi_drhos


