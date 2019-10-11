r"""
    
    Routines for calculating the Helmholtz energy for the SAFT-gamma equation of state.
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
#from profilehooks import profile

from . import constants

#@profile
def calc_a1s(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
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
    a1s : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

    logger = logging.getLogger(__name__)

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
                cikl = np.inner(constants.ckl_coef, np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3]).T)
                etakl[:, k, l] = np.einsum("ij,j", zetax_pow, cikl)
        a1s = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        # a1s is 4D matrix
        a1s = np.einsum("i,ijk->ijk", rho, a1s)  # {BottleNeck}

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3]).T)
            etakl[:, k] = np.einsum("ij,j", zetax_pow, cikl)
        a1s = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        # a1s is 3D matrix
        a1s = np.einsum("i,ij->ij", rho, a1s)
    else:
        print('Error in calc_a1s, unexpected array size')

    return a1s

#@profile
def calc_Xika(rho, xi, nui, nk, delta, maxiter=500, tol=1e-12):
    r""" 
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion.

    Parameters
    ----------
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

    Returns
    -------
    Xika : numpy.ndarray
        NoteHere
    """

    nbeads    = nui.shape[1]
    ncomp     = np.size(xi)
    nsitesmax = np.size(nk, axis=1)
    nrho      = len(rho)


    Xika_final = []
    err_array   = np.zeros(nrho)
    Xika = np.ones((ncomp, nbeads, nsitesmax))

    # Parallelize here, wrt rho!
    for r in range(nrho):
        for ind in range(maxiter):
            Xika0 = Xika
            # start fortran calc
            Xika = np.ones((ncomp, nbeads, nsitesmax))
            for i in range(ncomp):
                for k in range(nbeads):
                    for a in range(nsitesmax):

                        for j in range(ncomp):
                            for l in range(nbeads):
                                for b in range(nsitesmax):
                                    Xika[i,k,a] += (rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika0[j,l,b] * delta[r,i,j,k,l,a,b])
            Xika = 1./Xika
            err = np.amax(abs(Xika-Xika0))
            # end fortran calc
            print("    Xika: {}, Error: {}".format(Xika,err))
            if len(rho) == 2*28460:
                print("    Xika: {}, Error: {}".format(Xika,err))
            if err < tol:
                break
                print("Found Xika")
        if len(rho) == 2*28460:
            print("Density: {}, Error: {}".format(rho[r],err))
        err_array[r] = err
        Xika_final.append(Xika)

    Xika_final = np.array(Xika_final)
    return Xika_final, err_array


