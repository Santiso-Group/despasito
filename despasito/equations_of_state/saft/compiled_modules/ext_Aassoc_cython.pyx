# cython: profile=True
import numpy as np
cimport numpy as np

cdef double const_molecule_per_nm3 = 6.02214086e-4 # mol/m^3 to molecules/nm^3 

def calc_Xika( indices, rho, xi, nui, nk, Fklab, Kklab, gr_assoc): # , maxiter=500, tol=1e-12, damp=.1
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

    cdef int  maxiter=500
    cdef double  tol=1e-12
    cdef double  damp=0.1
    
    ncomp, nbeads = np.shape(nui)
    nsitesmax = np.shape(nk)[1]
    nrho = len(rho)
    l_ind = len(indices)

#    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    Xika_final = np.ones((nrho, len(indices)))
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
                    delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * gr_assoc[r,i, j]
                    Xika_elements_new[ind] += const_molecule_per_nm3 * rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[jnd] * delta
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

        err_array[r] = obj

        Xika_final[r,:] = Xika_elements
        #for jjnd in range(l_ind):
        #    i,k,a = indices[jjnd]
        #    Xika_final[r,i,k,a] = Xika_elements[jjnd]

    return Xika_final, err_array
