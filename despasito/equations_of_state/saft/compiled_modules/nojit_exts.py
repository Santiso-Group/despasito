r"""
    Routines for calculating the Helmholtz energy for the SAFT-gamma equation of state.
    
    Equations referenced in this code are from V. Papaioannou et al. J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import sys

from despasito.equations_of_state import constants
#from profilehooks import profile

logger = logging.getLogger(__name__)

ckl_coef = np.array([[0.81096, 1.7888, -37.578, 92.284], [1.0205, -19.341, 151.26, -463.50],
                     [-1.9057, 22.845, -228.14, 973.92], [1.0885, -6.1962, 106.98, -677.64]])

#@profile
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
        Matrix of hard sphere diameters for groups (k,l)

    Returns
    -------
    a1s : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

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
                cikl = np.inner(ckl_coef, np.transpose(np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3])))
                etakl[:, k, l] = np.einsum("ij,j", zetax_pow, cikl)
        a1s = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.Nav)) / (l_kl - 3.0)))
        # a1s is 4D matrix
        a1s = np.einsum("i,ijk->ijk", rho, a1s)  # {BottleNeck}

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl[:, k] = np.einsum("ij,j", zetax_pow, cikl)
        a1s = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3 * constants.Nav)) / (l_kl - 3.0)))
        # a1s is 3D matrix
        a1s = np.einsum("i,ij->ij", rho, a1s)
    else:
        print('Error in calc_a1s, unexpected array size')

    return a1s

#@profile
def calc_da1sii_drhos(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return da1s,kl(rho*Cmol2seg,l_kl)/rhos in K.
    
    Used in the calculation of :math:`A_chain` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

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
        Matrix of hard sphere diameters for groups (k,l)

    Returns
    -------
    da1s_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

    nbeads = np.size(dkl, axis=0)
    zetax_pow = np.zeros((np.size(rho), 4))
    zetax_pow[:, 0] = zetax
    for i in range(1, 4):
        zetax_pow[:, i] = zetax_pow[:, i - 1] * zetax_pow[:, 0]

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                # Constants to calculate eta_eff
                cikl = np.inner(ckl_coef, np.transpose(np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3])))
                etakl[:, k, l] = np.einsum("ij,j", zetax_pow, cikl)
                rhos_detakl_drhos[:, k, l] = np.einsum("ij,j", zetax_pow, cikl*np.array([1.0,2.0,3.0,4.0]))
        da1s_drhos = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos,
                        -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl[:, k] = np.einsum("ij,j", zetax_pow, cikl)
            rhos_detakl_drhos[:, k] = np.einsum("ij,j", zetax_pow, cikl*np.array([1.,2.,3.,4.]))

        tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
        tmp2 = -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
        da1s_drhos = np.einsum("ij,j->ij",tmp1,tmp2)
#        da1s_drhos = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos, -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
    else:
        print('Error in calc_da1s_drhos, unexpected array size')

    return da1s_drhos

#@profile
def calc_Xika(indices, rho, xi, nui, nk, Fklab, Kklab, Iij, maxiter=500, tol=1e-12, damp=.1):
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
    """

    nbeads    = nui.shape[1]
    ncomp     = np.size(xi)
    nsitesmax = np.size(nk, axis=1)
    nrho      = len(rho)
    l_ind = len(indices)


    Xika_final = np.ones((nrho,ncomp, nbeads, nsitesmax))
    err_array   = np.zeros(nrho)

    # Parallelize here, with respect to rho!
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
                    Xika_elements_new[ind] += constants.Nav * rho[r] * xi[j] * nui[j,l] * nk[l,b] *    Xika_elements[jnd] * delta
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

        for jjnd in range(l_ind):
            i,k,a = indices[jjnd]
            Xika_final[r,i,k,a] = Xika_elements[jjnd]
    
    return Xika_final, err_array

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
    
    if np.size(np.shape(l_kl)) == 2:
        # Bkl=np.zeros((np.size(rho),np.size(l_kl,axis=0),np.size(l_kl,axis=0)))
        Bkl = np.einsum("i,jk", rhos * (2.0 * np.pi),
                        (dkl**3 * constants.Nav) * epsilonkl) * (np.einsum("i,jk", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,jk", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl))
    elif np.size(np.shape(l_kl)) == 1:
        Bkl = np.einsum("i,j", rhos * (2.0 * np.pi),
                        (dkl**3 * constants.Nav) * epsilonkl) * (np.einsum("i,j", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,j", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl))
    else:
        logger.warning('Error unexpected l_kl shape in Bkl')
    
    return Bkl

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
    
    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))
    
    if np.size(np.shape(l_kl)) == 2:
        tmp1 = np.einsum("i,jk", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,jk", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl)
        tmp2 = np.einsum("i,jk", (5.0 - 2.0*zetax) / (2*(1.0 - zetax)**4), Ikl) - np.einsum("i,jk", ((9.0 * (zetax**2 + 4.0*zetax + 1)) / (2.0 * ((1 - zetax)**4))), Jkl)
        dBkl_drhos = (2.0 * np.pi)*(dkl**3) * epsilonkl * (tmp1 + np.einsum("i,jk", zetax, tmp2))
    elif np.size(np.shape(l_kl)) == 1:
        tmp1 = np.einsum("i,j", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,j", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl)
        tmp2 = np.einsum("i,j", (5.0 - 2.0*zetax) / (2*(1.0 - zetax)**4), Ikl) - np.einsum("i,j", ((9.0 * (zetax**2 + 4.0*zetax + 1)) / (2.0 * ((1 - zetax)**4))), Jkl)
        dBkl_drhos = np.einsum( "i,j", 2.0*np.pi*np.ones_like(zetax), (dkl**3)*epsilonkl)*tmp1 + np.einsum("i,j", zetax, (dkl**3)*epsilonkl)*tmp2
    
    return dBkl_drhos

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
    
    Bii_r = calc_Bkl(rho, l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_a = calc_Bkl(rho, l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    a1s_r = calc_a1s(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1s_a = calc_a1s(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)
    
    return (Cii * (((x0ii**l_aii_avg) * (a1s_a + Bii_a)) - ((x0ii**l_rii_avg) * (a1s_r + Bii_r))))

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
    
    das1_drhos_r = calc_da1sii_drhos(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    das1_drhos_a = calc_da1sii_drhos(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)
    
    dB_drhos_r = calc_dBkl_drhos(l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dB_drhos_a = calc_dBkl_drhos(l_aii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    
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
    
    a1sii_2l_aii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)
    
    Bii_2l_aii_avg = calc_Bkl(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    
    da1sii_2l_aii_avg = calc_da1sii_drhos(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    da1sii_2l_rii_avg = calc_da1sii_drhos(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    da1sii_l_rii_avgl_aii_avg = calc_da1sii_drhos(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)
    
    dBii_2l_aii_avg = calc_dBkl_drhos(2.0 * l_aii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dBii_2l_rii_avg = calc_dBkl_drhos(2.0 * l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    dBii_l_aii_avgl_rii_avg = calc_dBkl_drhos(l_aii_avg + l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax)
    
    # Calculate Derivative
    Cii = prefactor(l_rii_avg, l_aii_avg)
    
    B = x0ii**(2.0 * l_aii_avg) * (a1sii_2l_aii_avg + Bii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (a1sii_l_rii_avgl_aii_avg + Bii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (a1sii_2l_rii_avg + Bii_2l_rii_avg)
    dA_B = np.einsum("i,ij->ij", dKHS_drhos, 0.5*epsilonii_avg*Cii**2 * B / constants.Nav)
    
    dB = x0ii**(2.0 * l_aii_avg) * (da1sii_2l_aii_avg + dBii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (da1sii_l_rii_avgl_aii_avg + dBii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (da1sii_2l_rii_avg + dBii_2l_rii_avg)
    A_dB = np.einsum("i,ij->ij", KHS, 0.5*epsilonii_avg*Cii**2 * dB)
    
    da2ii_1pchi_drhos = A_dB + dA_B
    
    return da2ii_1pchi_drhos

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
        sys.exit("end")

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))
