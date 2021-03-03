import numpy as np
import numba

from despasito.equations_of_state import constants


def calc_Xika(indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc):
    r""" 
    A wrapper to calculate the fraction of molecules of component i that are not bonded at a site of type a on group k. Switched between functions for different Kklab

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
    Kklab : numpy.ndarray
        Bonding volume between each association site
    gr_assoc : numpy.ndarray
        Reference fluid pair correlation function used in calculating association sites, (len(rho) x Ncomp x Ncomp)

    Returns
    -------
    Xika : numpy.ndarray
        The fraction of molecules of component i that are not bonded at a site of type a on group k. Matrix (len(rho) x Ncomp x Nbeads x len(sitenames))
    err_array : numpy.ndarray
        Of the same length of rho, is a list in the error of the total error Xika for each point. 
    """

    l_K = len(np.shape(Kklab))

    # Ensure all inputs are numpy arrays
    tmp_array = [rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc]
    for i,tmp in enumerate(tmp_array):
        print(np.shape(tmp))
        if np.shape(tmp):
            print(type(tmp[0]))      
 
        if np.shape(tmp):
            tmp_array[i] = np.array(tmp)
        else:
            tmp_array[i] = np.array([tmp])
    rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc = tmp_array

    if l_K == 4:
        Xika_final, err_array = calc_Xika_4(
            indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc
        )
    if l_K == 6:
        Xika_final, err_array = calc_Xika_6(
            indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc
        )

    return Xika_final, err_array


#@numba.njit(
#    numba.types.Tuple((numba.f8[:, :], numba.f8[:]))(
#        numba.i8[:, :],
#        numba.f8[:],
#        numba.f8[:],
#        numba.f8[:, :],
#        numba.i8[:, :],
#        numba.f8[:, :, :, :],
#        numba.f8[:, :, :, :],
#        numba.f8[:, :, :],
#    )
#)
@numba.njit()
def calc_Xika_4(
    indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc
):  # , maxiter=500, tol=1e-12, damp=.1
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
    Kklab : numpy.ndarray
        Bonding volume between each association site
    gr_assoc : numpy.ndarray
        Reference fluid pair correlation function used in calculating association sites, (len(rho) x Ncomp x Ncomp)

    Returns
    -------
    Xika : numpy.ndarray
        The fraction of molecules of component i that are not bonded at a site of type a on group k. Matrix (len(rho) x Ncomp x Nbeads x len(sitenames))
    err_array : numpy.ndarray
        Of the same length of rho, is a list in the error of the total error Xika for each point. 
    """

    maxiter = 500
    tol = 1e-12
    damp = 0.1

    ncomp, nbeads = molecular_composition.shape
    nsitesmax = nk.shape[1]
    nrho = len(rho)
    l_ind = len(indices)

    #    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    Xika_final = np.ones((nrho, len(indices)))
    err_array = np.zeros(nrho)

    # Parallelize here, wrt rho!
    Xika_elements = 0.5 * np.ones(len(indices))
    for r in range(nrho):
        for knd in range(maxiter):

            Xika_elements_new = np.ones(len(Xika_elements))
            ind = 0
            for iind in range(l_ind):
                i, k, a = indices[iind]
                jnd = 0
                for jjnd in range(l_ind):
                    j, l, b = indices[jjnd]
                    delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * gr_assoc[r, i, j]
                    Xika_elements_new[ind] += (
                        constants.molecule_per_nm3
                        * rho[r]
                        * xi[j]
                        * molecular_composition[j, l]
                        * nk[l, b]
                        * Xika_elements[jnd]
                        * delta
                    )
                    jnd += 1
                ind += 1
            Xika_elements_new = 1.0 / Xika_elements_new
            obj = np.sum(np.abs(Xika_elements_new - Xika_elements))

            if obj < tol:
                break
            else:
                if obj / np.max(Xika_elements) > 1e3:
                    Xika_elements = Xika_elements + damp * (
                        Xika_elements_new - Xika_elements
                    )
                else:
                    Xika_elements = Xika_elements_new

        err_array[r] = obj

        Xika_final[r, :] = Xika_elements
        # for jjnd in range(l_ind):
        #    i,k,a = indices[jjnd]
        #    Xika_final[r,i,k,a] = Xika_elements[jjnd]

    return Xika_final, err_array


#@numba.njit(
#    numba.types.Tuple((numba.f8[:, :], numba.f8[:]))(
#        numba.i8[:, :],
#        numba.f8[:],
#        numba.f8[:],
#        numba.f8[:, :],
#        numba.i8[:, :],
#        numba.f8[:, :, :, :],
#        numba.f8[:, :, :, :, :, :],
#        numba.f8[:, :, :],
#    )
#)
@numba.njit()
def calc_Xika_6(
    indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc
):  # , maxiter=500, tol=1e-12, damp=.1
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
    Kklab : numpy.ndarray
        Bonding volume between each association site
    gr_assoc : numpy.ndarray
        Reference fluid pair correlation function used in calculating association sites, (len(rho) x Ncomp x Ncomp)

    Returns
    -------
    Xika : numpy.ndarray
        The fraction of molecules of component i that are not bonded at a site of type a on group k. Matrix (len(rho) x Ncomp x Nbeads x len(sitenames))
    err_array : numpy.ndarray
        Of the same length of rho, is a list in the error of the total error Xika for each point. 
    """

    maxiter = 500
    tol = 1e-12
    damp = 0.1

    ncomp, nbeads = molecular_composition.shape
    nsitesmax = nk.shape[1]
    nrho = len(rho)
    l_ind = len(indices)

    #    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    Xika_final = np.ones((nrho, len(indices)))
    err_array = np.zeros(nrho)

    # Parallelize here, wrt rho!
    Xika_elements = 0.5 * np.ones(len(indices))
    for r in range(nrho):
        for knd in range(maxiter):

            Xika_elements_new = np.ones(len(Xika_elements))
            ind = 0
            for iind in range(l_ind):
                i, k, a = indices[iind]
                jnd = 0
                for jjnd in range(l_ind):
                    j, l, b = indices[jjnd]
                    delta = (
                        Fklab[k, l, a, b] * Kklab[i, j, k, l, a, b] * gr_assoc[r, i, j]
                    )
                    Xika_elements_new[ind] += (
                        constants.molecule_per_nm3
                        * rho[r]
                        * xi[j]
                        * molecular_composition[j, l]
                        * nk[l, b]
                        * Xika_elements[jnd]
                        * delta
                    )
                    jnd += 1
                ind += 1
            Xika_elements_new = 1.0 / Xika_elements_new
            obj = np.sum(np.abs(Xika_elements_new - Xika_elements))

            if obj < tol:
                break
            else:
                if obj / np.max(Xika_elements) > 1e3:
                    Xika_elements = Xika_elements + damp * (
                        Xika_elements_new - Xika_elements
                    )
                else:
                    Xika_elements = Xika_elements_new

        err_array[r] = obj

        Xika_final[r, :] = Xika_elements
        # for jjnd in range(l_ind):
        #    i,k,a = indices[jjnd]
        #    Xika_final[r,i,k,a] = Xika_elements[jjnd]

    return Xika_final, err_array
