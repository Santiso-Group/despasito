# -- coding: utf8 --

import numpy as np
import logging
import os

from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)


def calc_Xika(
    indices,
    rho,
    xi,
    molecular_composition,
    nk,
    Fklab,
    Kklab,
    gr_assoc,
    maxiter=500,
    tol=1e-12,
    damp=0.1,
):
    r""" 
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k.

    Parameters
    ----------
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    molecular_composition : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    nk : numpy.ndarray
        For each bead the number of each type of site
    Fklab : numpy.ndarray
        The association strength between a site of type a on a group of type k of component i and a site of type b on a group of type l of component j., known as the Mayer f-function.
    gr_assoc : numpy.ndarray
        Reference fluid pair correlation function used in calculating association sites, (len(rho) x Ncomp x Ncomp)
    maxiter : int, Optional, default=500
        Maximum number of iteration for minimization
    tol : float, Optional, default=1e-12
        Once matrix converges.
    damp : float, Optional, default=0.1
        Only add a fraction of the new matrix values to update the guess

    Returns
    -------
    Xika : numpy.ndarray
        The fraction of molecules of component i that are not bonded at a site of type a on group k. Matrix (len(rho) x Ncomp x Nbeads x len(sitenames))
    err_array : numpy.ndarray
        Of the same length of rho, is a list in the error of the total error Xika for each point.
    """

    # ncomp, nbeads = np.shape(molecular_composition)
    # nsitesmax = np.shape(nk)[1]
    nrho = len(rho)
    l_ind = len(indices)
    l_K = len(np.shape(Kklab))

    #    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    Xika_final = np.ones((nrho, len(indices)))
    err_array = np.zeros(nrho)

    # Parallelize here, with respect to rho!
    Xika_elements_old = 0.5 * np.ones(len(indices))
    for r in range(nrho):
        for knd in range(maxiter):
            Xika_elements_new = np.ones(len(Xika_elements_old))
            for ind in range(l_ind):
                i, k, a = indices[ind]
                for jnd in range(l_ind):
                    j, l, b = indices[jnd]
                    if l_K == 4:
                        delta = (
                            Fklab[k, l, a, b] * Kklab[k, l, a, b] * gr_assoc[r, i, j]
                        )
                    elif l_K == 6:
                        delta = (
                            Fklab[k, l, a, b]
                            * Kklab[i, j, k, l, a, b]
                            * gr_assoc[r, i, j]
                        )
                    Xika_elements_new[ind] += (
                        constants.molecule_per_nm3
                        * rho[r]
                        * xi[j]
                        * molecular_composition[j, l]
                        * nk[l, b]
                        * Xika_elements_old[jnd]
                        * delta
                    )
            Xika_elements_new = 1.0 / Xika_elements_new
            obj = np.sum(np.abs(Xika_elements_new - Xika_elements_old))

            if obj < tol:
                break
            else:
                if obj / max(Xika_elements_old) > 1e3:
                    Xika_elements_old = Xika_elements_old + damp * (
                        Xika_elements_new - Xika_elements_old
                    )
                else:
                    Xika_elements_old = Xika_elements_new

        err_array[r] = obj

        Xika_final[r, :] = Xika_elements_old
        # for jjnd in range(l_ind):
        #    i,k,a = indices[jjnd]
        #    Xika_final[r,i,k,a] = Xika_elements[jjnd]

    return Xika_final, err_array
