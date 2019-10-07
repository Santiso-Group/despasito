import numpy as np
cimport numpy as np
import logging
import os

from .constants import ckl_coef

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
            tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k, l], 1.0/l_kl[k, l]**2, 1.0/l_kl[k, l]**3), dtype=ckl_coef.dtype ))
            etakl[:, k, l] = np.dot( zetax_pow, tmp )

    a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
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
        tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=ckl_coef.dtype ) )
        etakl[:, k] = np.dot( zetax_pow, tmp )

    a1s = - (1.0 - (etakl / 2.0)) / (1.0 - etakl)**3 * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0) )

    return np.transpose(np.transpose(a1s) * rho)
