
import numpy as np
cimport numpy as np
import logging
import os

from despasito.equations_of_state import constants

# For Numba, ckl_coef cannot be encapsulated
ckl_coef = np.array([[0.81096, 1.7888, -37.578, 92.284], [1.0205, -19.341, 151.26, -463.50],
                     [-1.9057, 22.845, -228.14, 973.92], [1.0885, -6.1962, 106.98, -677.64]])


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
        Matrix of hardsphere diameters for groups (k,l)

    Returns
    -------
    a1s : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Nrho x Ngroups x Ngroups
    """

    nbeads = len(dkl)
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.zeros((len(rho), nbeads, nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        for l in range(nbeads):
            tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k, l], 1.0/l_kl[k, l]**2, 1.0/l_kl[k, l]**3), dtype=ckl_coef.dtype ))
            etakl[:, k, l] = np.dot( zetax_pow, tmp )
    a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.molecule_per_nm3**2)) / (l_kl - 3.0))

    output = np.transpose(np.transpose(a1s) * rho)

    return output

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

def calc_a1ii(rho, Cmol2seg, dkl, l_akl, l_rkl, x0kl, epsilonkl, zetax):
    r"""
    Calculate effective first-order perturbation term :math:`\bar{a}_{1,ii}`.
    
    Used for the contribution of the monomeric interactions to the free energy per segment.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)
    l_rkl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential repulsive exponents for k,l groups
    l_akl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential attractive exponents for k,l groups 
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, ratio of Mie radius for groups scaled by hard sphere interaction (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    a1ii : numpy.ndarray
        Matrix used in the calculation of the radial distribution function of a hypothetical one-fluid Mie system.
    """
    Cii = (l_rkl / (l_rkl - l_akl)) * (l_rkl / l_akl)**(l_akl / (l_rkl - l_akl))

    Bii_r = calc_Bkl(rho, l_rkl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)
    Bii_a = calc_Bkl(rho, l_akl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)
    a1s_r = calc_a1s(rho, Cmol2seg, l_rkl, zetax, epsilonkl, dkl)
    a1s_a = calc_a1s(rho, Cmol2seg, l_akl, zetax, epsilonkl, dkl)

    return (Cii * (((x0kl**l_akl) / constants.molecule_per_nm3 * (a1s_a + Bii_a)) - ((x0kl**l_rkl) / constants.molecule_per_nm3 * (a1s_r + Bii_r))))

def calc_a1s_eff(rho, Cmol2seg, l_ii_avg, zetax, epsilonii_avg, dii_avg):
    r"""
    Return a1s,(rho*Cmol2seg,l_ij) in K
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_ii_avg : numpy.ndarray
        Average bead (i.e. group or segment) exponent in component (i.e. molecule) i.
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    dii_avg : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    
    Returns
    -------
    a1s_eff : numpy.ndarray
        Term used in the used in the calculation of the effective averaged molecular radial distribution function of a hypothetical one-fluid Mie system.
    """
    
    ncomp = len(dii_avg)
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]
    
    # check if you have more than 1 bead types
    etaii_avg = np.zeros((len(rho), ncomp), dtype=rho.dtype)

    for k in range(ncomp):
        tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_ii_avg[k], 1.0/l_ii_avg[k]**2, 1.0/l_ii_avg[k]**3), dtype=ckl_coef.dtype) )
        etaii_avg[:, k] = np.dot( zetax_pow, tmp )
    
    a1s = - (1.0 - (etaii_avg / 2.0)) / (1.0 - etaii_avg)**3 * 2.0 * np.pi * Cmol2seg * ((epsilonii_avg * (dii_avg**3 * constants.molecule_per_nm3**2)) / (l_ii_avg - 3.0) )
    
    output = np.transpose(np.transpose(a1s) * rho)
    
    return output

def calc_Bkl_eff(rho, l_ii_avg, Cmol2seg, dii_avg, epsilonii_avg, x0ii, zetax):
    r"""
    Return Bii_avg(rho*Cmol2seg,l_ii_avg) in K as defined in eq. 20.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    l_ii_avg : numpy.ndarray
        Average bead (i.e. group or segment) exponent in component (i.e. molecule) i.
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dii_avg : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    x0ii : numpy.ndarray
        Matrix of sigmaii_avg/dii_eff
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    Bii_avg : numpy.ndarray
        Bii_avg(rho*Cmol2seg,l_ii_avg) in K as defined in eq. 20.
    
    """
    
    rhos = Cmol2seg * rho
    ncomp = len(dii_avg)
    
    # compute Iii_avg(l_ii_avg), eq. 23
    Iii_avg = (1.0 - (x0ii**(3.0 - l_ii_avg))) / (l_ii_avg - 3.0)
    # compute Jii_avg(l_ii_avg), eq. 24
    Jii_avg = (1.0 - ((x0ii**(4.0 - l_ii_avg)) * (l_ii_avg - 3.0)) + ((x0ii**(3.0 - l_ii_avg)) * (l_ii_avg - 4.0))) / ((l_ii_avg - 3.0) * (l_ii_avg - 4.0))
    
    tmp11 = rhos * (2.0 * np.pi)
    tmp12 = (dii_avg**3 * constants.molecule_per_nm3**2) * epsilonii_avg
    tmp2 = (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3)
    tmp3 = (9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))
    
    Bii_avg = np.zeros((len(rho), ncomp))
    for k in np.arange(ncomp):
        Bii_avg[:,k] = tmp11*tmp12[k]*(tmp2*Iii_avg[k] - tmp3*Jii_avg[k])
    #  Bii_avg = tmp11*tmp12*(tmp2*Iii_avg - tmp3*Jii_avg)
    
    return Bii_avg

def calc_da1sii_drhos(rho, Cmol2seg, l_ii_avg, zetax, epsilonii_avg, dii_avg):
    r""" 
    Return a1s,ii_avg(rho*Cmol2seg,l_ii_avg) in K as defined in eq. 25.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    l_ii_avg : numpy.ndarray
        Average bead (i.e. group or segment) exponent in component (i.e. molecule) i.
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    dii_avg : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.

    Returns
    -------
    calc_da1sii_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """
    ncomp = len(dii_avg)
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etaii_avg = np.zeros((len(rho), ncomp), dtype=rho.dtype)
    rhos_detaii_avg_drhos = np.zeros((len(rho), ncomp), dtype=rho.dtype)

    for k in range(ncomp):
        ciii_avg = np.dot(ckl_coef, np.array( (1.0, 1.0/l_ii_avg[k], 1.0/l_ii_avg[k]**2, 1.0/l_ii_avg[k]**3), dtype=ckl_coef.dtype ) )
        etaii_avg[:, k] = np.dot( zetax_pow, ciii_avg)
        rhos_detaii_avg_drhos[:, k] = np.dot(zetax_pow, ciii_avg*np.array([1.,2.,3.,4.]))

    tmp1 = (1.0 - (etaii_avg / 2.0)) / ((1.0 - etaii_avg)**3) + (5.0-2.0*etaii_avg)/(2.0*(1.0-etaii_avg)**4)*rhos_detaii_avg_drhos
    tmp2 = - 2.0 * np.pi * ((epsilonii_avg * (dii_avg**3 * constants.molecule_per_nm3**2)) / (l_ii_avg - 3.0))

    da1s_drhos = tmp1*tmp2

    #da1s_drhos = - 2.0 * np.pi * ((1.0 - (etaii_avg / 2.0)) / ((1.0 - etaii_avg)**3) + (5.0 - 2.0*etaii_avg)/(2.0*(1.0-etaii_avg)**4)) * rhos_detaii_avg_drhos * ((epsilonii_avg * (dii_avg**3)) / (l_ii_avg - 3.0))
    return da1s_drhos

def calc_dBkl_drhos(l_ii_avg, dii_avg, epsilonii_avg, x0ii, zetax):
    r"""
    Return derivative of Bkl(rho*Cmol2seg,l_ii_avg) with respect to :math:`\rho_S`.
    
    Used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.
    
    Parameters
    ----------
    l_aii_avg : numpy.ndarray
        Average bead (i.e. group or segment) attractive exponent in component (i.e. molecule) i.
    dii_avg : numpy.ndarray
        Effective hard sphere diameter of the beads (i.e. groups or segments) in component (i.e. molecule) i.
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    x0ii : numpy.ndarray
        Matrix of sigmaii_avg/dii_eff
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    dBkl_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is rho x l_ii_avg.shape
    
    """
    ncomp = len(dii_avg)
    nrho = len(zetax)

    # compute Iii_avg(l_ii_avg), eq. 23
    Iii_avg = (1.0 - (x0ii**(3.0 - l_ii_avg))) / (l_ii_avg - 3.0)
    # compute Jii_avg(l_ii_avg), eq. 24
    Jii_avg = (1.0 - ((x0ii**(4.0 - l_ii_avg)) * (l_ii_avg - 3.0)) + ((x0ii**(3.0 - l_ii_avg)) * (l_ii_avg - 4.0))) / ((l_ii_avg - 3.0) * (l_ii_avg - 4.0))

    tmp = 2.0 * np.pi * dii_avg**3 * epsilonii_avg
    tmp1 = np.zeros((nrho,ncomp))
    tmp2 = np.zeros((nrho,ncomp))
    for k in np.arange(ncomp):
        tmp1[:,k] = ((1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3) * Iii_avg[k]) - (((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))) * Jii_avg[k])
        tmp2[:,k] = ((5.0 - 2.0*zetax) * zetax / (2*(1.0 - zetax)**4) * Iii_avg[k]) - (((9.0 * zetax * (zetax**2 + 4.0*zetax + 1)) / (2.0 * ((1 - zetax)**4))) * Jii_avg[k])
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

    a1sii_2l_aii_avg = calc_a1s_eff(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s_eff(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s_eff(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)

    Bii_2l_aii_avg = calc_Bkl_eff(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl_eff(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl_eff(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)

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

