import numpy as np

const_molecule_per_nm3 = 6.02214086e-4 # mol/m^3 to molecules/nm^3 

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

    maxiter = 500
    damp = 0.1
    tol = 1e-12

    nrho = len(rho)
    l_ind = len(indices)
    Xika_elements_old = 0.5*np.ones(len(indices))

    Xika_elements0 = np.ones(l_ind)
    Xika_elements_new = np.empty(l_ind)

    Xika_final = np.empty((nrho, l_ind))
    err_array = np.empty(nrho)

    for r in range(nrho):
        for knd in range(maxiter):
            Xika_elements_new = np.copy(Xika_elements0)
            for ind in range(l_ind):
                i = indices[ind][0]
                k = indices[ind][1]
                a = indices[ind][2]
                for jnd in range(l_ind):
                    j = indices[jnd][0] 
                    l = indices[jnd][1]
                    b = indices[jnd][2]
                    delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * gr_assoc[r,i, j]
                    Xika_elements_new[ind] = Xika_elements_new[ind] + const_molecule_per_nm3 * rho[r] * xi[j] * nui[j,l] * nk[l,b] * Xika_elements_old[jnd] * delta
            Xika_elements_new = np.reciprocal(Xika_elements_new)
            obj = np.sum(np.abs(np.subtract(Xika_elements_new,Xika_elements_old)))

            if obj < tol:
                break
            else:
                if obj/max(Xika_elements_old) > 1e+3:
                    Xika_elements_old = np.add(Xika_elements_old, damp*np.subtract(Xika_elements_new,Xika_elements_old))
                else:
                    Xika_elements_old = Xika_elements_new

        err_array[r] = obj
        Xika_final[r,:] = Xika_elements_old

    return Xika_final, err_array
