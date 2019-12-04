import numpy as np
import os
#from timeit import default_timer as timer

if 'NUMBA_DISABLE_JIT' in os.environ:
    disable_jit = os.environ['NUMBA_DISABLE_JIT']
else:
    from .. import jit_stat
    disable_jit = jit_stat.disable_jit

if disable_jit:
    os.environ['NUMBA_DISABLE_JIT'] = '1'

import numba

# For Numba, ckl_coef cannot be encapsulated
from .constants import ckl_coef
from profilehooks import profile

@profile
def calc_a1s(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" wrapper function for calling 2d/3d versions of calc_a1s ... this is done for stupid Numba 
    """
    if len(l_kl.shape) == 2:
        output = calc_a1s_2d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)
    elif len(l_kl.shape) == 1:
        output = calc_a1s_1d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)

    return output

@numba.njit(numba.f8[:,:,:](numba.f8[:], numba.f8, numba.f8[:,:], numba.f8[:], numba.f8[:,:], numba.f8[:,:]))
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

    a1s = - (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    return np.transpose(np.transpose(a1s) * rho)

@numba.njit(numba.f8[:,:](numba.f8[:], numba.f8, numba.f8[:], numba.f8[:], numba.f8[:], numba.f8[:]))
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
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.zeros((len(rho), nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=ckl_coef.dtype ) )
        etakl[:, k] = np.dot( zetax_pow, tmp )

    a1s = - (1.0 - (etakl / 2.0)) / (1.0 - etakl)**3 * 2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0) )

    return np.transpose(np.transpose(a1s) * rho)

@profile
def calc_da1sii_drhos(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" wrapper function for calling 2d/3d versions of calc_a1s ... this is done for stupid Numba 
    """
    if len(l_kl.shape) == 2:
        output = calc_da1sii_drhos_2d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)
    elif len(l_kl.shape) == 1:
        output = calc_da1sii_drhos_1d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl)

    return output

@numba.njit(numba.f8[:,:,:](numba.f8[:], numba.f8, numba.f8[:,:], numba.f8[:], numba.f8[:,:], numba.f8[:,:]))
def calc_da1sii_drhos_2d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
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
    da1sii_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups

    :note: output seems to be a tensor of size (N x Ngroups x Ngroups)
    """

    nbeads = len(dkl)
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.zeros((len(rho), nbeads, nbeads), dtype=rho.dtype)
    rhos_detakl_drhos = np.zeros((len(rho), nbeads, nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        for l in range(nbeads):
            tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k, l], 1.0/l_kl[k, l]**2, 1.0/l_kl[k, l]**3), dtype=ckl_coef.dtype ))
            tmp_dr = np.dot(ckl_coef, np.array( (1.0, 2.0/l_kl[k, l], 3.0/l_kl[k, l]**2, 4.0/l_kl[k, l]**3), dtype=ckl_coef.dtype ))
            etakl[:, k, l] = np.dot( zetax_pow, tmp )
            rhos_detakl_drhos[:, k, l] = np.dot( zetax_pow, tmp_dr)

    tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
    tmp2 = - 2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    da1s_drhos = tmp1*tmp2

    return da1s_drhos

@numba.njit(numba.f8[:,:](numba.f8[:], numba.f8, numba.f8[:], numba.f8[:], numba.f8[:], numba.f8[:]))
def calc_da1sii_drhos_1d(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
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
    zetax_pow = np.zeros((len(rho), 4), dtype=rho.dtype)
    zetax_pow[:, 0] = zetax
    for i in range(1,4):
        zetax_pow[:, i] = zetax_pow[:, i-1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    etakl = np.zeros((len(rho), nbeads), dtype=rho.dtype)
    rhos_detakl_drhos = np.zeros((len(rho), nbeads), dtype=rho.dtype)

    for k in range(nbeads):
        tmp = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=ckl_coef.dtype ) )
        tmp_dr = np.dot(ckl_coef, np.array( (1.0, 1.0/l_kl[k], 1.0/l_kl[k]**2, 1.0/l_kl[k]**3), dtype=ckl_coef.dtype ) )*np.array((1.0,2.0,3.0,4.0))
        etakl[:, k] = np.dot( zetax_pow, tmp )
        rhos_detakl_drhos[:, k] = np.dot( zetax_pow, tmp_dr )

    tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
    tmp2 = - 2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    da1s_drhos = tmp1*tmp2

    #da1s_drhos = - 2.0 * np.pi * ((1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0 - 2.0*etakl)/(2.0*(1.0-etakl)**4)) * rhos_detakl_drhos * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
    return da1s_drhos

@numba.njit(numba.types.Tuple((numba.f8[:,:,:,:], numba.f8[:]))(numba.i8[:,:], numba.f8[:], numba.f8[:], numba.f8[:,:], numba.f8[:,:], numba.f8[:,:,:,:], numba.f8[:,:,:,:], numba.f8[:,:,:])) # , numba.i8, numba.f8, numba.f8
def calc_Xika(indices, rho, xi, nui, nk, Fklab, Kklab, Iij): # , maxiter=500, tol=1e-12, damp=.1
    r""" 
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion.

    Parameters
    ----------
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    nk : numpy.ndarray
        For each bead the number of each type of site
    delta : numpy.ndarray
        The association strength between a site of type a on a group of type k of component i and a site of type b on a group of type l of component j. eq. 66
    maxiter : int, Optional, default: 500
        Maximum number of iteration for minimization
    tol : float, Optional, default: 1e-12
        Once matrix converges.
    damp : float, Optional, default: 0.1
        Only add a fraction of the new matrix values to update the guess

    Returns
    -------
    Xika : numpy.ndarray
        NoteHere
    err_array : numpy.ndarray
        Of the same length of rho, is a list in the error of the total error Xika for each point. 
    """

    maxiter=500
    tol=1e-12
    damp=.1

    nbeads    = nui.shape[1]
    ncomp     = len(xi)
    nsitesmax = nk.shape[1]
    nrho      = len(rho)
    l_ind = len(indices)


    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    err_array   = np.zeros(nrho)

    # Parallelize here, wrt rho!
    Xika_elements = .5*np.ones(len(indices))
    for r in range(nrho):
        for knd in range(maxiter):

            Xika_elements_new = np.ones(len(Xika_elements))
            ind = 0
            for iind in range(l_ind):
                i, k, a = indices[iind]
                jnd = 0
                for jjnd in range(l_ind):
                    j, l, b = indices[jjnd]
                    delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * Iij[r,i, j]
                    Xika_elements_new[ind] += rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[jnd] * delta
                    jnd += 1
                ind += 1
            Xika_elements_new = 1./Xika_elements_new
            obj = np.sum(np.abs(Xika_elements_new - Xika_elements))

            if obj < tol:
                break
            else:
                if obj/max(Xika_elements) > 1e+3:
                    Xika_elements = Xika_elements + damp*(Xika_elements_new - Xika_elements)
                else:
                    Xika_elements = Xika_elements_new

      #  if knd == maxiter-1:
      #      print("Didn't find Xika within {} iterations, error: {}".format(maxiter,obj))

        err_array[r] = obj

        for jjnd in range(l_ind):
            i,k,a = indices[jjnd]
            Xika_final[r,i,k,a] = Xika_elements[jjnd]

    return Xika_final, err_array

