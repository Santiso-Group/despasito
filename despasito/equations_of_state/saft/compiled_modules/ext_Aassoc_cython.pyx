# cython: profile=True,  boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.stdio cimport printf

FTYPE = np.float
ctypedef np.float_t FTYPE_t

cdef double const_molecule_per_nm3 = 6.02214086e-4 # mol/m^3 to molecules/nm^3 

def calc_Xika(indices, rho, xi, nui, nk, Fklab, Kklab, gr_assoc): # , maxiter=500, tol=1e-12, damp=.1
    r""" 
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k.

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
    Xika_init = 0.5*np.ones(len(indices), dtype=np.float_)

    indices = indices.astype(np.intc)
    rho = rho.astype(np.float_)
    xi = xi.astype(np.float_)
    nui = nui.astype(np.float_)
    nk = nk.astype(np.float_)
    Fklab = Fklab.astype(np.float_)
    Kklab = Kklab.astype(np.float_)
    gr_assoc = gr_assoc.astype(np.float_)

    l_K = len(np.shape(Kklab))
    if l_K == 4:
        Xika_final, err_array = calc_Xika_4(indices, rho, xi, nui, nk, Fklab, Kklab, gr_assoc, Xika_init)
    if l_K == 6:
        Xika_final, err_array = calc_Xika_6(indices, rho, xi, nui, nk, Fklab, Kklab, gr_assoc, Xika_init)

    return Xika_final, err_array

cdef calc_Xika_4(int[:,:] indices, double[:] rho, double[:] xi, double[:,:] nui, double[:,:] nk, double[:,:,:,:] Fklab, double[:,:,:,:] Kklab, double[:,:,:] gr_assoc, double[:] Xika_elements):

    cdef int  maxiter=500
    cdef double  tol=1e-12
    cdef double  damp=0.1

    cdef int nrho = rho.shape[0]
    cdef int l_ind = indices.shape[0]

    cdef int r, knd, ind, i, k, a, jnd, j, l, b, z
    cdef double delta, obj, Xika_max
    cdef double[:] Xika_elements_new = np.empty(l_ind, dtype=np.float_)

    cdef np.ndarray[np.float_t, ndim=2] Xika_final = np.empty((nrho, l_ind), dtype=np.float_)
    cdef np.ndarray[np.float_t, ndim=1] err_array = np.empty(nrho, dtype=np.float_)

    for r in range(nrho):
        for knd in range(maxiter):
            Xika_elements_new[:] = 1.0
            for ind in range(l_ind):
                i = indices[ind][0]
                k = indices[ind][1]
                a = indices[ind][2]
                for jnd in range(l_ind):
                    j = indices[jnd][0] 
                    l = indices[jnd][1]
                    b = indices[jnd][2]
                    delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * gr_assoc[r,i, j]
                    Xika_elements_new[ind] = Xika_elements_new[ind] + const_molecule_per_nm3 * rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[jnd] * delta

            obj = 0
            Xika_max = 0
            for z in range(l_ind):
                Xika_elements_new[z] = 1.0/Xika_elements_new[z]
                obj += abs(Xika_elements_new[z]-Xika_elements[z])
                if Xika_elements_new[z] > Xika_max:
                    Xika_max = Xika_elements_new[z]

            if obj < tol:
                break
            else:
                if obj/Xika_max > 1e+3:
                    for z in range(l_ind):
                        Xika_elements[z] = Xika_elements[z] + damp*(Xika_elements_new[z] - Xika_elements[z])
                else:
                    Xika_elements = Xika_elements_new

        err_array[r] = obj
        Xika_final[r,:] = Xika_elements

    return Xika_final, err_array

cdef calc_Xika_6(int[:,:] indices, double[:] rho, double[:] xi, double[:,:] nui, double[:,:] nk, double[:,:,:,:] Fklab, double[:,:,:,:,:,:] Kklab, double[:,:,:] gr_assoc, double[:] Xika_elements):

    cdef int  maxiter=500
    cdef double  tol=1e-12
    cdef double  damp=0.1

    cdef int nrho = rho.shape[0]
    cdef int l_ind = indices.shape[0]

    cdef int r, knd, ind, i, k, a, jnd, j, l, b, z
    cdef double delta, obj, Xika_max
    cdef double[:] Xika_elements_new = np.empty(l_ind, dtype=np.float_)

    cdef np.ndarray[np.float_t, ndim=2] Xika_final = np.empty((nrho, l_ind), dtype=np.float_)
    cdef np.ndarray[np.float_t, ndim=1] err_array = np.empty(nrho, dtype=np.float_)

    for r in range(nrho):
        for knd in range(maxiter):
            Xika_elements_new[:] = 1.0
            for ind in range(l_ind):
                i = indices[ind][0]
                k = indices[ind][1]
                a = indices[ind][2]
                for jnd in range(l_ind):
                    j = indices[jnd][0]
                    l = indices[jnd][1]
                    b = indices[jnd][2]
                    delta = Fklab[k, l, a, b] * Kklab[i, j, k, l, a, b] * gr_assoc[r,i, j]
                    Xika_elements_new[ind] = Xika_elements_new[ind] + const_molecule_per_nm3 * rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[jnd] * delta

            obj = 0
            Xika_max = 0
            for z in range(l_ind):
                Xika_elements_new[z] = 1.0/Xika_elements_new[z]
                obj += abs(Xika_elements_new[z]-Xika_elements[z])
                if Xika_elements_new[z] > Xika_max:
                    Xika_max = Xika_elements_new[z]

            if obj < tol:
                break
            else:
                if obj/Xika_max > 1e+3:
                    for z in range(l_ind):
                        Xika_elements[z] = Xika_elements[z] + damp*(Xika_elements_new[z] - Xika_elements[z])
                else:
                    Xika_elements = Xika_elements_new

        err_array[r] = obj
        Xika_final[r,:] = Xika_elements

    return Xika_final, err_array

