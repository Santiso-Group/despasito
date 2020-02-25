# -- coding: utf8 --

r"""
    
    Routines for calculating the Helmholtz energy for the SAFT-gamma equation of state.
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import logging
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit, vmap
from jax.ops import index, index_add, index_update, index_min, index_max
from jax import lax
# jax.experimental? 
from scipy import integrate # approximate with taylor series?

from . import constants

############################################################
#                                                          #
#                 A Ideal Contribution                     #
#                                                          #
############################################################

def calc_Aideal(xi, rho, massi, T):
    r""" 
    Return a vector of ideal contribution of Helmholtz energy.
    :math:`\frac{A^{ideal}}{N k_{B} T}`
    
    Parameters
    ----------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    massi : numpy.ndarray
        Mass for each component [kg/mol]
    T : float
        Temperature of the system [K]

    Returns
    -------
    Aideal : numpy.ndarray
        Helmholtz energy of ideal gas for each density given.
    """

    logger = logging.getLogger(__name__)

    if any(np.isnan(rho)):
        raise ValueError("NaN was given as a value of density, rho") 
    elif rho.size == 0:
        raise ValueError("No value of density, rho, was given")

    # Check for mole fractions of zero and remove those components
    tol = 1e-32
    l_x = len(xi)
    l_x0 = len(xi[xi<tol])
    if l_x0:
        xi_tmp = np.zeros(l_x-len(ind))
        massi_tmp = np.zeros(l_x-len(ind))

        j = 0
        for i,x in enumerate(xi):
            if x < tol:
                xi_tmp = index_update(xi_tmp, index[j], xi[i])
                massi_tmp = index_update(massi_tmp, index[j], massi[i])
                j = j + 1
    else:
        xi_tmp = xi
        massi_tmp = massi

    # rhoi: (number of components,number of densities) number density of each component for each density
    rhoi = np.outer(rho, xi_tmp)
    Lambda3 = (constants.h / np.sqrt(2.0 * np.pi * (massi_tmp / constants.Nav) * constants.kb * T))**3
    Aideal_tmp = rhoi*Lambda3
    log_broglie3_rho = np.log(Aideal_tmp)

#    if not any(np.sum(xi_tmp * np.log(Aideal_tmp), axis=1)):
    if np.isnan(np.sum(np.sum(xi_tmp * log_broglie3_rho, axis=1))):
        raise ValueError("Aideal has values of zero when taking the log. All mole fraction values should be nonzero. Mole fraction: {}".format(xi_tmp))
    else:
        Aideal = np.sum(xi_tmp * log_broglie3_rho, axis=1) - 1.0

    return Aideal

############################################################
#                                                          #
#            A Monomer (Group) Contribution                #
#                                                          #
############################################################

def _dkk_int(r, Ce_kT, sigma, l_r, l_a):
    r""" 
    Return integrand used to calculate the hard sphere diameter, :math:`d_{k,k}` of a group k. See eq. 10.
    
    Parameters
    ----------
    r : numpy.ndarray
        Bead distance between zero and :math:`sigma_{k,k}` [Å]
    Ce_kT : float
        :math:`C \epsilon_{k,k}/(k_B T)`, Mie prefactor scaled by kT
    sigma : float
        :math:`\sigma_{k,k}`, Size parameter [Å] (or same units as r)
    l_r : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    l_a : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k

    Returns
    -------
    dkk_int_tmp : numpy.ndarray
        Integrand used to calculate the hard sphere diameter
    """

    logger = logging.getLogger(__name__)

    dkk_int_tmp = 1.0 - np.exp(-Ce_kT * (np.power(sigma / r, l_r) - np.power(sigma / r, l_a)))

    return dkk_int_tmp


def calc_dkk(epsilon, sigma, T, l_r, l_a=6.0):
    r""" 
    Calculates hard sphere diameter of a group, :math:`d_{k,k}`. Defined in eq. 10.

    Parameters
    ----------
    epsilon : float
        :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant [K]
    sigma : float
        :math:`\sigma_{k,k}`, Size parameter [Å] (or same units as r)
    T : float
        Temperature of the system [K]
    l_r : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    l_a : float, Optional, default: 6.0
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k

    Returns
    -------
    dkk : float
        Hard sphere diameter of a group [Å] 
    """

    logger = logging.getLogger(__name__)

    Ce_kT = C(l_r, l_a) * epsilon / T
    # calculate integral of dkk_int from 0.0 to sigma
#    results = integrate.quad(lambda r: _dkk_int(r, Ce_kT, sigma, l_r, l_a), 0.0, sigma, epsabs=1.0e-16, epsrel=1.0e-16)
    # 40 point Gauss Lengedre quadrature
    w = np.array([0.077505948, 0.077505948, 0.077039818, 0.077039818, 0.076110362, 0.076110362, 0.074723169, 0.074723169, 0.072886582, 0.072886582, 0.070611647, 0.070611647, 0.067912046, 0.067912046, 0.064804013, 0.064804013, 0.061306242, 0.061306242, 0.057439769, 0.057439769, 0.053227847, 0.053227847, 0.048695808, 0.048695808, 0.043870908, 0.043870908, 0.038782168, 0.038782168, 0.033460195, 0.033460195, 0.027937007, 0.027937007, 0.022245849, 0.022245849, 0.016421058, 0.016421058, 0.010498285, 0.010498285, 0.004521277, 0.004521277])
    x = np.array([-0.038772418, 0.038772418, -0.116084071, 0.116084071, -0.192697581, 0.192697581, -0.268152185, 0.268152185, -0.341994091, 0.341994091, -0.413779204, 0.413779204, -0.483075802, 0.483075802, -0.549467125, 0.549467125, -0.61255389, 0.61255389, -0.671956685, 0.671956685, -0.727318255, 0.727318255, -0.778305651, 0.778305651, -0.824612231, 0.824612231, -0.865959503, 0.865959503, -0.902098807, 0.902098807, -0.932812808, 0.932812808, -0.957916819, 0.957916819, -0.97725995, 0.97725995, -0.990726239, 0.990726239, -0.99823771, 0.99823771])

    r = 0.5*sigma*(x+1)
    results = 0.5*sigma*np.sum(w*_dkk_int(r, Ce_kT, sigma, l_r, l_a))

    return results

def C(l_r, l_a):
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

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))

def calc_interaction_matrices(beads, beadlibrary, crosslibrary={}):
    r"""
    Computes matrices of cross interaction parameters epsilonkl, sigmakl, l_akl, l_rkl (attractive and repulsive exponents), Ckl

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.

        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        - l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.

    Returns
    -------
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    l_akl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    l_rkl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    Ckl : numpy.ndarray
        Matrix of mie potential prefactors for k,l groups
    """

    logger = logging.getLogger(__name__)

    nbeads = len(beads)
    sigmakl = np.zeros((nbeads, nbeads))
    l_rkl = np.zeros((nbeads, nbeads))
    l_akl = np.zeros((nbeads, nbeads))
    epsilonkl = np.zeros((nbeads, nbeads))

    # compute default interaction parameters for beads
    for k in range(nbeads):
        for l in range(nbeads):

            tmp = (beadlibrary[beads[k]]["sigma"] + beadlibrary[beads[l]]["sigma"]) / 2.0
            sigmakl = index_update(sigmakl, index[k, l], tmp)

            tmp = 3 + np.sqrt((beadlibrary[beads[k]]["l_r"] - 3.0) * (beadlibrary[beads[l]]["l_r"] - 3.0))
            l_rkl = index_update(l_rkl, index[k, l], tmp)

            tmp = 3 + np.sqrt((beadlibrary[beads[k]]["l_a"] - 3.0) * (beadlibrary[beads[l]]["l_a"] - 3.0))
            l_akl = index_update(l_akl, index[k, l], tmp)

            tmp = np.sqrt(beadlibrary[beads[k]]["epsilon"] * beadlibrary[beads[l]]["epsilon"]) * \
                            np.sqrt((beadlibrary[beads[k]]["sigma"] ** 3) * \
                            (beadlibrary[beads[l]]["sigma"] ** 3)) / (sigmakl[k, l] ** 3)
            epsilonkl = index_update(epsilonkl, index[k, l], tmp)

    # testing if crosslibrary is empty i.e. not specified
    if crosslibrary:

        for (i, beadname) in enumerate(beads):
            if beadname in crosslibrary:
                for (j, beadname2) in enumerate(beads):
                    if beadname2 in crosslibrary[beadname]:
                        if "epsilon" in crosslibrary[beads[i]][beads[j]]:
                            epsilonkl = index_update(epsilonkl,index[i, j], crosslibrary[beads[i]][beads[j]]["epsilon"])
                            epsilonkl = index_update(epsilonkl,index[j, i],epsilonkl[i, j])
                        if "l_r" in crosslibrary[beads[i]][beads[j]]:
                            l_rkl = index_update(l_rkl,index[i, j], crosslibrary[beads[i]][beads[j]]["l_r"])
                            l_rkl = index_update(l_rkl,index[j, i],l_rkl[i, j])

    Ckl = C(l_rkl, l_akl)

    return epsilonkl, sigmakl, l_akl, l_rkl, Ckl

def calc_composition_dependent_variables(xi, nui, beads, beadlibrary):
    r""" 
    Return conversion factor from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13

    Parameters
    ----------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
        Defined for eq. 11. Note that indices are flipped from definition in reference.
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    Returns
    -------
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xsk : numpy.ndarray
        Mole fraction of each bead (i.e. segment or group), sum(xsk) should equal 1.0
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    """

    logger = logging.getLogger(__name__)

    # compute Conversion factor
    Cmol2seg = 0.0
    for i in range(np.size(xi)):
        for j in range(np.size(beads)):
            Cmol2seg = Cmol2seg + xi[i] * nui[i, j] * beadlibrary[beads[j]]["Vks"] * beadlibrary[beads[j]]["Sk"]

    # initialize variables and arrays
    nbeads = len(beads)
    xsk = np.zeros(nbeads, float)
    # compute xsk
    for k in range(nbeads):
        xsk = index_update(xsk,index[k],np.sum(xi * nui[:, k]) * beadlibrary[beads[k]]["Vks"] * \
                 beadlibrary[beads[k]]["Sk"])
    xsk = xsk/Cmol2seg

    # calculate  xskl matrix
    xskl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            xskl = index_update(xskl,index[k, l],xsk[k] * xsk[l])

    return Cmol2seg, xsk, xskl

def calc_hard_sphere_matricies(beads, beadlibrary, sigmakl, T):
    r"""
    Computes matrix of hard sphere interaction parameters dkk, dkl, and x0kl
    This does not include function specific or association terms

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    T : float
        Temperature of the system [K]
    
    Returns
    -------
    dkk : numpy.ndarray
        Array of hard sphere diameters for each group
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, sigmakl is the mie radius for groups (k,l)
    """

    logger = logging.getLogger(__name__)

    nbeads = len(beads)
    dkk = np.zeros(np.size(beads))
    for i in range(np.size(beads)):
        tmp = calc_dkk(beadlibrary[beads[i]]["epsilon"], beadlibrary[beads[i]]["sigma"], T,
                          beadlibrary[beads[i]]["l_r"], beadlibrary[beads[i]]["l_a"])
        dkk = index_update(dkk,index[i],tmp)

    dkl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            dkl = index_update(dkl,index[k, l],(dkk[k] + dkk[l]) / 2.0)

    x0kl = sigmakl / dkl

    return dkk, dkl, x0kl

def calc_Bkl(rho, l_kl, Cmol2seg, dkl, epsilonkl, x0kl, zetax):
    r""" 
    Return Bkl(rho*Cmol2seg,l_kl) in K as defined in eq. 20, used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    l_kl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential exponents for k,l groups
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
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

    logger = logging.getLogger(__name__)

    rhos = Cmol2seg * rho

    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) * (l_kl - 4.0))

    if np.size(np.shape(l_kl)) == 2:
        # Bkl=np.zeros((np.size(rho),np.size(l_kl,axis=0),np.size(l_kl,axis=0)))

        Bkl = np.einsum("i,jk", rhos * (2.0 * np.pi),
                        (dkl**3) * epsilonkl) * (np.einsum("i,jk", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,jk", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl))
    elif np.size(np.shape(l_kl)) == 1:
        Bkl = np.einsum("i,j", rhos * (2.0 * np.pi),
                        (dkl**3) * epsilonkl) * (np.einsum("i,j", (1.0 - (zetax / 2.0)) / ((1.0 - zetax)**3), Ikl) - np.einsum("i,j", ((9.0 * zetax * (1.0 + zetax)) / (2.0 * ((1 - zetax)**3))), Jkl))
    else:
        logger.warning('Error unexpeced l_kl shape in Bkl')

    return Bkl

def calc_dBkl_drhos(l_kl, dkl, epsilonkl, x0kl, zetax):
    r""" 
    Return Bkl(rho*Cmol2seg,l_kl) in K as defined in eq. 20, used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

    Parameters
    ----------
    l_kl : numpy.ndarray
        :math:`\lambda_{k,l}` Matrix of Mie potential exponents for k,l groups
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
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

    logger = logging.getLogger(__name__)

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

def calc_fm(alphakl, mlist):
    r""" 
    Calculate fm(alphakl) where a list of m values are specified in mlist eq. 39

    Parameters
    ----------
    alphakl : numpy.ndarray
        (Ngroup,Ngroup) "A dimensionless form of the integrated vdW energy of the Mie potential" eq. 33
    mlist : numpy.ndarray
        (number of m values) an array of integers used in the calculation of :math:`A^{mono}`

    Returns
    -------
    fmlist : numpy.ndarray
        List of coefficients used to compute the correction term for :math:`A_{2}` which is related to the fluctuations of attractive energy.
    """

    logger = logging.getLogger(__name__)

    nbeads = np.size(alphakl, axis=0)
    if np.size(np.shape(alphakl)) == 2:
        fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0), np.size(alphakl, axis=0)))
    elif np.size(np.shape(alphakl)) == 1:
        fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0)))
    else:
        logger.warning('Error: unexpected shape in calcfm')
    mlist = mlist - 1

    for i, m in enumerate(mlist):
        for n in range(4):
            tmp = fmlist[i] +constants.phimn[m, n] * (alphakl**n)
            fmlist = index_update(fmlist,index[i],tmp)

        dum = np.ones_like(fmlist[i])
        for n in range(4, 7):
            dum = dum + constants.phimn[m, n] * (alphakl**(n - 3.0))
        fmlist = index_update(fmlist,index[i],fmlist[i] / dum)

    return fmlist

@jit
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
    zetax_pow = index_update(zetax_pow,index[:,0],zetax)
    for i in range(1, 4):
        zetax_pow = index_update(zetax_pow,index[:,i],zetax_pow[:, i - 1] * zetax_pow[:, 0])

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3])))
                etakl = index_update(etakl,index[:, k, l],np.einsum("ij,j", zetax_pow, cikl))

        a1s = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        # a1s is 4D matrix
        a1s = np.einsum("i,ijk->ijk", rho, a1s)  # {BottleNeck}

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl = index_update(etakl,index[:,k],np.einsum("ij,j", zetax_pow, cikl))
        a1s = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Cmol2seg * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        # a1s is 3D matrix
        a1s = np.einsum("i,ij->ij", rho, a1s)
    else:
        print('Error in calc_a1s, unexpected array size')

    return a1s

def calc_Amono(rho, xi, nui, Cmol2seg, xsk, xskl, dkk, T, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl):
    r""" 
    Outputs :math:`A^{HS}, A_1, A_2`, and :math:`A_3` (number of densities) :math:`A^{mono.}` components as well as some related quantities. Note these quantities are normalized by NkbT. Eta is really zeta

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xsk : numpy.ndarray
        Mole fraction of each bead (i.e. segment or group), sum(xsk) should equal 1.0
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    dkk : numpy.ndarray
        Array of hard sphere diameters for each group
    T : float
        Temperature of the system [K]
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
    l_akl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    l_rkl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    Ckl : numpy.ndarray
        Matrix of mie potential prefactors for k,l groups
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, sigmakl is the mie radius for groups (k,l)

    Returns
    -------
    AHS : numpy.ndarray
        Hard sphere contribution of Helmholtz energy, length of array rho
    A1 : numpy.ndarray
        Contribution of first perturbation corresponding to mean attractive energy, length of array rho
    A2 : numpy.ndarray
        Contribution of second perturbation corresponding to fluctuations mean attractive energy, length of array rho
    A3 : numpy.ndarray
        Contribution of third perturbation term of mean attractive energy, length of array rho
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    zetaxstar : numpy.ndarray
        Matrix of hypothetical packing fraction based on sigma
    KHS : numpy.ndarray
        (length of densities) isothermal compressibility of system with packing fraction zetax
    """

    logger = logging.getLogger(__name__)

    # initialize variables
    nbeads = list(nui.shape)[1]  # nbeads is the number of unique groups used by any compnent
    rhos = rho * Cmol2seg

    ##### compute AHS (eq. 16) #####

    # initialize variables for AHS
    eta = np.zeros((np.size(rho), 4))

    # compute eta, eq. 14
    for m in range(4):
        eta = index_update(eta,index[:, m],rhos * (np.sum(xsk * (dkk**m)) * (np.pi / 6.0)))

    if rho.any() == 0.0:
        logger.warning("rho:",rho)
    # compute AHS, eq. 16
    tmp1 = np.log(1.0 - eta[:, 3]) * (((eta[:, 2]**3) / (eta[:, 3]**2)) - eta[:, 0])
    tmp2 = (3.0 * eta[:, 1] * eta[:, 2] / (1 - eta[:, 3]))
    tmp3 = ((eta[:, 2]**3) / (eta[:, 3] * ((1.0 - eta[:, 3])**2)))
    AHS = (6.0 / (np.pi * rho)) * (np.log(1.0 - eta[:, 3]) * (((eta[:, 2]**3) / (eta[:, 3]**2)) - eta[:, 0]) + (3.0 * eta[:, 1] * eta[:, 2] / (1 - eta[:, 3])) + ((eta[:, 2]**3) / (eta[:, 3] * ((1.0 - eta[:, 3])**2))))

    ##### compute a1kl, eq. 19 #####

    # calc zetax eq. 22
    zetax = rhos * ((np.pi / 6.0) * np.sum(xskl * (dkl**3)))

    # compute components of eq. 19
    a1kl = calc_a1ii(rho, Cmol2seg, dkl, l_akl, l_rkl, x0kl, epsilonkl, zetax)

    ##### compute a2kl, eq. 30 #####

    # initialize variables for a2kl
    # a2kl = np.zeros((nbeads,nbeads))
    # alphakl = np.zeros((nbeads,nbeads))

    # compute KHS(rho), eq. 31
    KHS = ((1.0 - zetax)**4) / (1.0 + (4.0 * zetax) + (4.0 * (zetax**2)) - (4.0 * (zetax**3)) + (zetax**4))

    # compute alphakl eq. 33
    alphakl = Ckl * ((1.0 / (l_akl - 3.0)) - (1.0 / (l_rkl - 3.0)))

    # compute zetaxstar eq. 35
    zetaxstar = rhos * ((np.pi / 6.0) * np.sum(xskl * (sigmakl**3)))

    # compute f1, f2, and f3 for eq. 32
    fmlist123 = calc_fm(alphakl, np.array([1, 2, 3]))

    chikl = np.einsum("i,jk", zetaxstar, fmlist123[0]) + np.einsum("i,jk", zetaxstar**5, fmlist123[1]) + np.einsum("i,jk", zetaxstar**8, fmlist123[2])

    a1s_2la = calc_a1s(rho, Cmol2seg, 2.0 * l_akl, zetax, epsilonkl, dkl)
    a1s_2lr = calc_a1s(rho, Cmol2seg, 2.0 * l_rkl, zetax, epsilonkl, dkl)
    a1s_lalr = calc_a1s(rho, Cmol2seg, l_akl + l_rkl, zetax, epsilonkl, dkl)
    B_2la = calc_Bkl(rho, 2.0 * l_akl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)
    B_2lr = calc_Bkl(rho, 2.0 * l_rkl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)
    B_lalr = calc_Bkl(rho, l_akl + l_rkl, Cmol2seg, dkl, epsilonkl, x0kl, zetax)

    a2kl = (x0kl**(2.0 * l_akl)) * (a1s_2la + B_2la) - ((2.0 * x0kl**(l_akl + l_rkl)) *
                                                        (a1s_lalr + B_lalr)) + ((x0kl**(2.0 * l_rkl)) * (a1s_2lr + B_2lr))
    a2kl *= (1.0 + chikl) * epsilonkl * (Ckl**2)  # *(KHS/2.0)
    a2kl = np.einsum("i,ijk->ijk", KHS / 2.0, a2kl)

    ##### compute a3kl #####
    a3kl = np.zeros((nbeads, nbeads))
    fmlist456 = calc_fm(alphakl, np.array([4, 5, 6]))

    a3kl = np.einsum("i,jk", zetaxstar, -(epsilonkl**3) * fmlist456[0]) * np.exp(
        np.einsum("i,jk", zetaxstar, fmlist456[1]) + np.einsum("i,jk", zetaxstar**2, fmlist456[2]))
    # a3kl=-(epsilonkl**3)*fmlist456[0]*zetaxstar*np.exp((fmlist456[1]*zetaxstar)+(fmlist456[2]*(zetaxstar**2)))

    # compute a1, a2, a3 from 18, 29, and 37 respectively
    a1 = np.einsum("ijk,jk->i", a1kl, xskl)
    a2 = np.einsum("ijk,jk->i", a2kl, xskl)
    a3 = np.einsum("ijk,jk->i", a3kl, xskl)

    # compute A1, A2, and A3
    # note that a1, a2, and a3 have units of K, K^2, and K^3 respectively
    A1 = (Cmol2seg / T) * a1
    A2 = (Cmol2seg / (T**2)) * a2
    A3 = (Cmol2seg / (T**3)) * a3


    return AHS, A1, A2, A3, zetax, zetaxstar, KHS

############################################################
#                                                          #
#                  A Chain Contribution                    #
#                                                          #
############################################################

def calc_a1ii(rho, Cmol2seg, dii_eff, l_aii_avg, l_rii_avg, x0ii, epsilonii_avg, zetax):
    r""" 
    Calculate effective first-order perturbation term :math:`\bar{a}_{1,ii}` for the contribution of the monomeric interactions to the free energy per segment.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
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

    logger = logging.getLogger(__name__)

    Cii = C(l_rii_avg, l_aii_avg)

    Bii_r = calc_Bkl(rho, l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_a = calc_Bkl(rho, l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    a1s_r = calc_a1s(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1s_a = calc_a1s(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)

    return (Cii * (((x0ii**l_aii_avg) * (a1s_a + Bii_a)) - ((x0ii**l_rii_avg) * (a1s_r + Bii_r))))

@jit
def calc_da1sii_drhos(rho, Cmol2seg, l_kl, zetax, epsilonkl, dkl):
    r""" 
    Return da1s,kl(rho*Cmol2seg,l_kl)/rhos in K, used in the calculation of :math:`A_chain` the first order term of the perturbation expansion corresponding to the mean-attractive energy.

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
    da1s_drhos : numpy.ndarray
        Matrix used in the calculation of :math:`A_1` the first order term of the perturbation expansion corresponding to the mean-attractive energy, size is the Ngroups by Ngroups
    """

    logger = logging.getLogger(__name__)

    nbeads = np.size(dkl, axis=0)
    zetax_pow = np.zeros((np.size(rho), 4))
    zetax_pow = index_update(zetax_pow,index[:,0],zetax)
    for i in range(1, 4):
        zetax_pow = index_update(zetax_pow,index[:, i],zetax_pow[:, i - 1] * zetax_pow[:, 0])

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                # Constants to calculate eta_eff
                cikl = np.inner(constants.ckl_coef, np.transpose([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3]))
                etakl = index_update(etakl,index[:, k, l],np.einsum("ij,j", zetax_pow, cikl))
                rhos_detakl_drhos = index_update(rhos_detakl_drhos,index[:, k, l],np.einsum("ij,j", zetax_pow, cikl*np.array([1.0,2.0,3.0,4.0])))
        da1s_drhos = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos,
                        -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        rhos_detakl_drhos = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.transpose(np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3])))
            etakl = index_update(etakl,index[:, k],np.einsum("ij,j", zetax_pow, cikl))
            rhos_detakl_drhos = index_update(rhos_detakl_drhos,index[:, k],np.einsum("ij,j", zetax_pow, cikl*np.array([1.,2.,3.,4.])))

        tmp1 = (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos
        tmp2 = -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0))
        da1s_drhos = np.einsum("ij,j->ij",tmp1,tmp2)
#        da1s_drhos = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3) + (5.0-2.0*etakl)/(2.0*(1.0-etakl)**4)*rhos_detakl_drhos, -2.0 * np.pi * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
    else:
        print('Error in calc_da1s_drhos, unexpected array size')

    return da1s_drhos

def calc_da1iidrhos(rho, Cmol2seg, dii_eff, l_aii_avg, l_rii_avg, x0ii, epsilonii_avg, zetax):

    r""" 
    Compute derivative of the term, :math:`\bar{a}_{1,ii}` with respect to :math:`\rho_s`

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
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

    logger = logging.getLogger(__name__)

    Cii = C(l_rii_avg, l_aii_avg)

    das1_drhos_r = calc_da1sii_drhos(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)
    das1_drhos_a = calc_da1sii_drhos(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)

    dB_drhos_r = calc_dBkl_drhos(l_rii_avg, dii_eff, epsilonii_avg, x0ii, zetax) 
    dB_drhos_a = calc_dBkl_drhos(l_aii_avg, dii_eff, epsilonii_avg, x0ii, zetax)

    da1iidrhos = (Cii * (((x0ii**l_aii_avg) * (das1_drhos_a + dB_drhos_a)) - ((x0ii**l_rii_avg) * (das1_drhos_r + dB_drhos_r))))

    return da1iidrhos

def calc_a2ii_1pchi(rho, Cmol2seg, epsilonii_avg, dii_eff, x0ii, l_rii_avg, l_aii_avg, zetax):

    r""" 
    Calculate the term, :math:`\frac{\bar{a}_{2,ii}}{1+\bar{\chi}_{ii}}`, used in the calculation of the second-order term from the macroscopic compressibility approximation based on the fluctuation term of the Sutherland potential.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
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
    a2ii_1pchi : nump.ndarray
        Term used in the calculation of the second-order term from the macroscopic compressibility
        
    """

    logger = logging.getLogger(__name__)

    KHS = ((1.0 - zetax)**4) / (1.0 + (4.0 * zetax) + (4.0 * (zetax**2)) - (4.0 * (zetax**3)) + (zetax**4))
    Cii = C(l_rii_avg, l_aii_avg)

    a1sii_2l_aii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)

    Bii_2l_aii_avg = calc_Bkl(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)

    a2ii_1pchi = 0.5 * epsilonii_avg * (Cii**2) * ((x0ii**(2.0 * l_aii_avg)) * (a1sii_2l_aii_avg + Bii_2l_aii_avg) - (2.0 * (x0ii**(l_aii_avg + l_rii_avg))) * (a1sii_l_rii_avgl_aii_avg + Bii_l_aii_avgl_rii_avg) +
 (x0ii**(2.0 * l_rii_avg)) * (a1sii_2l_rii_avg + Bii_2l_rii_avg))

    a2ii_1pchi = np.einsum("i,ij->ij", KHS, a2ii_1pchi)
    return a2ii_1pchi

def calc_da2ii_1pchi_drhos(rho, Cmol2seg, epsilonii_avg, dii_eff, x0ii, l_rii_avg, l_aii_avg, zetax):

    r""" 
    Compute derivative of the term, :math:`\frac{\bar{a}_{2,ii}}{1+\bar{\chi}_{ii}}` with respect to :math:`\rho_s`

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
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
    da2ii_1pchi_drhos : nump.ndarray
        Term used in the calculation of the second-order term from the macroscopic compressibility
        
    """

    logger = logging.getLogger(__name__)

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
    Cii = C(l_rii_avg, l_aii_avg)

    B = x0ii**(2.0 * l_aii_avg) * (a1sii_2l_aii_avg + Bii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (a1sii_l_rii_avgl_aii_avg + Bii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (a1sii_2l_rii_avg + Bii_2l_rii_avg)
    dA_B = np.einsum("i,ij->ij", dKHS_drhos, 0.5*epsilonii_avg*Cii**2 * B)

    dB = x0ii**(2.0 * l_aii_avg) * (da1sii_2l_aii_avg + dBii_2l_aii_avg) - 2.0 * x0ii**(l_aii_avg + l_rii_avg) * (da1sii_l_rii_avgl_aii_avg + dBii_l_aii_avgl_rii_avg) + x0ii**(2.0 * l_rii_avg) * (da1sii_2l_rii_avg + dBii_2l_rii_avg)
    A_dB = np.einsum("i,ij->ij", KHS, 0.5*epsilonii_avg*Cii**2 * dB)

    da2ii_1pchi_drhos = A_dB + dA_B

    return da2ii_1pchi_drhos

def calc_Achain(rho, Cmol2seg, xi, T, nui, sigmakl, epsilonkl, dkl, xskl, l_rkl, l_akl, beads, beadlibrary, zetax, zetaxstar, KHS):
    r"""
    Calculation of chain contribution of Helmholtz energy, :math:`A^{chain}`.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
        Defined for eq. 11. Note that indices are flipped from definition in reference.
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    l_rkl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    l_akl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    zetax : numpy.ndarray 
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    zetaxstar : numpy.ndarray
        Matrix of hypothetical packing fraction based on sigma for groups (k,l)
    KHS : numpy.ndarray
        (length of densities) isothermal compressibility of system with packing fraction zetax

    Returns
    -------
    Achain : numpy.ndarray
        Chain contribution of Helmholtz energy, length of array rho

    """

    logger = logging.getLogger(__name__)

    #initialize values
    ngroups = len(beads)
    ncomp = np.size(xi)
    zki = np.zeros((ncomp, ngroups), float)
    zkinorm = np.zeros(ncomp, float)
    sigmaii_avg = np.zeros(ncomp, float)
    dii_eff = np.zeros_like(sigmaii_avg)
    epsilonii_avg = np.zeros_like(sigmaii_avg)
    l_rii_avg = np.zeros_like(sigmaii_avg)
    l_aii_avg = np.zeros_like(sigmaii_avg)
    x0ii = np.zeros_like(sigmaii_avg)
    km = np.zeros((np.size(rho), 4))
    gdHS = np.zeros((np.size(rho), ncomp))

    kT = T * constants.kb
    rhos = rho * Cmol2seg

    stepmult = 100

    #compute zki
    for i in range(ncomp):
        for k in range(ngroups):
            zki = index_update(zki,index[i, k],nui[i, k] * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"])
            zkinorm = index_update(zkinorm,index[i],zkinorm[i] + zki[i, k])

    for i in range(ncomp):
        for k in range(ngroups):
            zki = index_update(zki,index[i,k],zki[i, k] / zkinorm[i])

    # compute average molecular segment size: sigmaii_avg
    # compute effective hard sphere diameter : dii_eff
    # compute average interaction energy epsilonii_avg
    #compute average repulsive and attractive exponenets l_rkl, l_akl
    for i in range(ncomp):
        for k in range(ngroups):
            for l in range(ngroups):
                sigmaii_avg = index_update(sigmaii_avg,index[i],sigmaii_avg[i] + zki[i, k] * zki[i, l] * sigmakl[k, l]**3)
                dii_eff = index_update(dii_eff,index[i],dii_eff[i] + zki[i, k] * zki[i, l] * dkl[k, l]**3)
                epsilonii_avg = index_update(epsilonii_avg,index[i],epsilonii_avg[i] + zki[i, k] * zki[i, l] * epsilonkl[k, l] * constants.kb)
                l_rii_avg = index_update(l_rii_avg,index[i],l_rii_avg[i] + zki[i, k] * zki[i, l] * l_rkl[k, l])
                l_aii_avg = index_update(l_aii_avg,index[i],l_aii_avg[i] + zki[i, k] * zki[i, l] * l_akl[k, l])
        dii_eff = index_update(dii_eff,index[i],dii_eff[i]**(1/3.0))
        sigmaii_avg = index_update(sigmaii_avg,index[i],sigmaii_avg[i]**(1/3.0))

    #compute x0ii
    x0ii = sigmaii_avg/dii_eff

    tmp = -np.log(1.0 - zetax) + (42.0 * zetax - 39.0 * zetax**2 + 9.0 * zetax**3 - 2.0 * zetax**4) / (6.0 * (1.0 - zetax)**3)
    km = index_update(km,index[:,0],tmp)
    km = index_update(km,index[:,1],(zetax**4 + 6.0 * zetax**2 - 12.0 * zetax) / (2.0 * (1.0 - zetax)**3))
    km = index_update(km,index[:,2],-3.0 * zetax**2 / (8.0 * (1.0 - zetax)**2))
    km = index_update(km,index[:,3],(-zetax**4 + 3.0 * zetax**2 + 3.0 * zetax) / (6.0 * (1.0 - zetax)**3))

    for i in range(ncomp):
        gdHS = index_update(gdHS,index[:,i],np.exp(km[:, 0] + km[:, 1] * x0ii[i] + km[:, 2] * x0ii[i]**2 + km[:, 3] * x0ii[i]**3))

    da1iidrhos = calc_da1iidrhos(rho, Cmol2seg, dii_eff, l_aii_avg, l_rii_avg, x0ii, epsilonii_avg, zetax)

    a1sii_l_aii_avg = calc_a1s(rho, Cmol2seg, l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avg = calc_a1s(rho, Cmol2seg, l_rii_avg, zetax, epsilonii_avg, dii_eff)

    Bii_l_aii_avg = calc_Bkl(rho, l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_rii_avg = calc_Bkl(rho, l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)

    Cii = C(l_rii_avg, l_aii_avg)

    g1 = (1.0 / (2.0 * np.pi * epsilonii_avg * dii_eff**3)) * (3.0 * da1iidrhos - Cii * l_aii_avg * (x0ii**l_aii_avg) * np.einsum("ij,i->ij", (a1sii_l_aii_avg + Bii_l_aii_avg), 1.0 / rhos) + (Cii * l_rii_avg *  (x0ii**l_rii_avg)) * np.einsum("ij,i->ij", (a1sii_l_rii_avg + Bii_l_rii_avg), 1.0 / rhos))

    #compute g2
    phi7 = np.array([10.0, 10.0, 0.57, -6.7, -8.0])
    alphaii = Cii * ((1.0 / (l_aii_avg - 3.0)) - (1.0 / (l_rii_avg - 3.0)))
    theta = np.exp(epsilonii_avg / kT) - 1.0

    gammacii = np.zeros_like(gdHS)
    for i in range(ncomp):
        tmp = phi7[0] * (-np.tanh(phi7[1] * (phi7[2] - alphaii[i])) + 1.0) * zetaxstar * theta[i] * np.exp(phi7[3] * zetaxstar + phi7[4] * (zetaxstar**2))
        gammacii = index_update(gammacii,index[:, i],tmp)

    a2iidchi = calc_a2ii_1pchi(rho, Cmol2seg, epsilonii_avg, dii_eff, x0ii, l_rii_avg, l_aii_avg, zetax)

    da2iidrhos = calc_da2ii_1pchi_drhos(rho, Cmol2seg, epsilonii_avg, dii_eff, x0ii, l_rii_avg, l_aii_avg, zetax)

    a1sii_2l_aii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_aii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_2l_rii_avg = calc_a1s(rho, Cmol2seg, 2.0 * l_rii_avg, zetax, epsilonii_avg, dii_eff)
    a1sii_l_rii_avgl_aii_avg = calc_a1s(rho, Cmol2seg, l_aii_avg + l_rii_avg, zetax, epsilonii_avg, dii_eff)

    Bii_2l_aii_avg = calc_Bkl(rho, 2.0 * l_aii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_2l_rii_avg = calc_Bkl(rho, 2.0 * l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)
    Bii_l_aii_avgl_rii_avg = calc_Bkl(rho, l_aii_avg + l_rii_avg, Cmol2seg, dii_eff, epsilonii_avg, x0ii, zetax)

    eKC2 = np.einsum("i,j->ij", KHS / rhos, epsilonii_avg * (Cii**2))

    g2MCA = (1.0 / (2.0 * np.pi * (epsilonii_avg**2) * dii_eff**3)) * (( \
        3.0 * da2iidrhos) - \
        (eKC2 * l_rii_avg * (x0ii**(2.0 * l_rii_avg))) * (a1sii_2l_rii_avg + Bii_2l_rii_avg) + \
        eKC2 * (l_rii_avg + l_aii_avg) * (x0ii**(l_rii_avg + l_aii_avg)) * (a1sii_l_rii_avgl_aii_avg + Bii_l_aii_avgl_rii_avg) - \
        eKC2 * l_aii_avg * (x0ii**(2.0 * l_aii_avg)) * (a1sii_2l_aii_avg + Bii_2l_aii_avg))

    g2 = (1.0 + gammacii) * g2MCA
    #g2=np.einsum("i,ij->ij",1.0+gammacii,g2MCA)

    gii = gdHS * np.exp((epsilonii_avg * g1 / (kT * gdHS)) + (((epsilonii_avg / kT)**2) * g2 / gdHS))
    tmp = [(epsilonii_avg * g1 / (kT * gdHS)), (((epsilonii_avg / kT)**2) * g2 / gdHS)]
    Achain = 0.0
    tmp_A = [0, 0]
    for i in range(ncomp):
        beadsum = -1.0

        for k in range(ngroups):
            beadsum = beadsum + (nui[i, k] * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"])

        Achain = Achain - xi[i] * beadsum * np.log(gii[:, i])
        tmp_A[0] = tmp_A[0] - tmp[0][:, i]
        tmp_A[1] = tmp_A[1] - tmp[1][:, i]

    #return Achain,sigmaii_avg,epsilonii_avg,np.array(tmp_A)
    return Achain, sigmaii_avg, epsilonii_avg

############################################################
#                                                          #
#            A Association Site Contribution               #
#                                                          #
############################################################

def calc_assoc_matrices(beads, beadlibrary, sitenames=["H", "e1", "e2"], crosslibrary={}):
    r"""

    Generate matrices used for association site calculations.  Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk (number of sites )

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    sitenames : list[str], Optional, default: []
        List of unique association sites used among components
    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        - l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
        - epsilon*: Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Bonding volume between each association site. Asterisk represents two strings from sitenames.

    Returns
    -------
    epsilonHB : numpy.ndarray
        Interaction energy between each bead and association site.
    Kklab : numpy.ndarray 
        Bonding volume between each association site
    nk : numpy.ndarray
        For each bead the number of each type of site
    """

    logger = logging.getLogger(__name__)

    # initialize variables
    nbeads = len(beads)
    sitemax = np.size(sitenames)
    epsilonHB = np.zeros((nbeads, nbeads, sitemax, sitemax))
    Kklab = np.zeros((nbeads, nbeads, sitemax, sitemax))
    nk = np.zeros((nbeads, sitemax))

    for i in range(nbeads):
        for j in range(np.size(sitenames)):
            if "Nk"+sitenames[j] in beadlibrary[beads[i]]:
                logger.debug("Bead {} has {} of the association site {}".format(beads[i],beadlibrary[beads[i]]["Nk"+sitenames[j]],"Nk"+sitenames[j]))
                nk = index_update(nk,index[i,j],beadlibrary[beads[i]]["Nk" + sitenames[j]])

    if crosslibrary:
        # find any cross terms in the cross term library
        crosslist = []
        for (i, beadname) in enumerate(beads):
            if beadname in crosslibrary:
                for (j, beadname2) in enumerate(beads):
                    if beadname2 in crosslibrary[beadname]:

                        for a in range(np.size(sitenames)):
                            for b in range(np.size(sitenames)):
                                if beads[i] in crosslibrary:
                                    if beads[j] in crosslibrary[beads[i]]:
                                        
                                        epsilon_tmp = "epsilon"+sitenames[a]+sitenames[b]
                                        K_tmp = "K"+sitenames[a]+sitenames[b]
                                        if epsilon_tmp in crosslibrary[beads[i]][beads[j]]:
                                            if (nk[i][a] == 0 or nk[j][b] == 0):
                                                if 0 not in [nk[i][b],nk[j][a]]:
                                                    logger.warning("Site names were listed in wrong order for parameter definitions in cross interaction library. Changing {}_{} - {}_{} interaction to {}_{} - {}_{}".format(beads[i],sitenames[a],beads[j],sitenames[b],beads[i],sitenames[b],beads[j],sitenames[a]))
                                                a, b = [b, a]
                                            elif nk[i][a] == 0:
                                                logger.warning("Cross interaction library parameters suggest a {}_{} - {}_{} interaction, but {} doesn't have site {}.".format(beads[i],sitenames[a],beads[j],sitenames[b],beads[i],sitenames[a]))
                                            elif nk[j][b] == 0:
                                                logger.warning("Cross interaction library parameters suggest a {}_{} - {}_{} interaction, but {} doesn't have site {}.".format(beads[i],sitenames[a],beads[j],sitenames[b],beads[j],sitenames[b]))
    
                                            epsilonHB = index_update(epsilonHB,index[i, j, a, b],crosslibrary[beads[i]][beads[j]][epsilon_tmp])
                                            epsilonHB = index_update(epsilonHB,index[j, i, b, a],epsilonHB[i, j,a, b])
    
                                    if K_tmp in crosslibrary[beads[i]][beads[j]]:
                                        Kklab = index_update(Kklab,index[i, j, a, b],crosslibrary[beads[i]][beads[j]][K_tmp])
                                        Kklab = index_update(Kklab,index[j, i, b, a],Kklab[i, j, a, b])

    for i in range(nbeads):
        for a in range(np.size(sitenames)):
            for b in range(np.size(sitenames)):
                tmp = ["epsilon"+sitenames[a]+sitenames[b], "K"+sitenames[a]+sitenames[b]]
                if all(x in beadlibrary[beads[i]] for x in tmp):
                    epsilonHB = index_update(epsilonHB,index[i, i, a, b],beadlibrary[beads[i]]["epsilon" + sitenames[a] + sitenames[b]])
                    epsilonHB = index_update(epsilonHB,index[i, i, b, a],epsilonHB[i, i, a, b])
                    Kklab = index_update(Kklab,index[i, i, a, b], beadlibrary[beads[i]]["K" + sitenames[a] + sitenames[b]])
                    Kklab = index_update(Kklab,index[i, i, b, a],Kklab[i, i, a, b])                    

    if Kklab.size:
        if max(Kklab.flatten()) > 1e-27:
            raise ValueError("Check units for association site parameter K. Should be in units of m^3.")

    return epsilonHB, Kklab, nk

#@jit
def calc_Xika(indices, rho, xi, nui, nk, Fklab, Kklab, Iij, tol=1e-12, damp=.1):
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
    nrho      = rho.shape[0]
    l_ind     = indices.shape[0]

    Xika_final0 = np.ones((nrho,ncomp, nbeads, nsitesmax))
    Xika_elements = np.ones((2,l_ind))*0.5

    # New Method #############
    inputs = _calc_Xika_fori_debug(0, nrho, _calc_Xika_outer, (Xika_final0, Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol))
    #inputs = lax.fori_loop(0, nrho, _calc_Xika_outer, (Xika_final0, Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol))
    Xika_final, _, _, _, _, _, _, _, _, _, _, _ = inputs
    # Old Method #############
    #l_ind = np.array(indices).shape[0]

    #for r in range(nrho):

    #    #inputs = _calc_Xika_debug(_calc_Xika_cond, _calc_Xika_inner, (Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol))
    #    inputs = lax.while_loop(_calc_Xika_cond, _calc_Xika_inner, (Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol))

    #    Xika_elements, _, _, _, _, _, _, _, _, _, _ = inputs

    #    for jjnd in range(l_ind):
    #        i,k,a = indices[jjnd]
    #        Xika_final = index_update(Xika_final,index[r,i,k,a],Xika_elements[0][jjnd])
     #########################

    return Xika_final

def _calc_Xika_fori_debug(start,stop,body_fun,init_val):

    val = init_val
    for i in np.arange(start,stop):
        val = body_fun(i, val)

    return val

#@jit
def _calc_Xika_outer(r, inputs):
    r"""
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion. Inner loop for use with jit

    Parameters
    ----------
    r : int
        Index used in for loop
    inputs : tuple
        Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol

    Returns
    -------
    inputs : tuple
        Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol
    """

    Xika_final0, Xika_elements0, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol = inputs
    Xika_elements0 = index_update(Xika_elements0,index[1,:],1.0)

    Xika_elements, _, _, _, _, _, _, _, _, _, _ = _calc_Xika_while_debug(_calc_Xika_cond, _calc_Xika_inner, (Xika_elements0, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol))
    #Xika_elements, _, _, _, _, _, _, _, _, _, _ = lax.while_loop(_calc_Xika_cond, _calc_Xika_inner, (Xika_elements0, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol))

    #print(r, rho[r], Xika_elements[0])

    # New Way ################
    _, _, _, Xika_final = lax.fori_loop(0,indices.shape[0],_Xika_update_1,(r, indices, Xika_elements[0], Xika_final0))
    # Old Way ################
    #for jjnd in range(np.array(indices).shape[0]):
    #    i,k,a = indices[jjnd]
    #    Xika_final = index_update(Xika_final,index[r,i,k,a],Xika_elements[0][jjnd])
    ##########################

    inputs = (Xika_final, Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol)

    return inputs

#@jit
def _Xika_update_1(ii, inputs):

    r, indices, Xika_elements0, Xika_final = inputs
    
    i,k,a = indices[ii]
    Xika_final = index_update(Xika_final,index[r,i,k,a],Xika_elements0[ii])
    input_out = (r, indices, Xika_elements0, Xika_final)
    return input_out

#@jit
def _calc_Xika_while_debug(cond_fun,body_fun,init_val):

    val = init_val
    while cond_fun(val):
        val = body_fun(val)
        print(val[0][0],val[0][1],cond_fun(val))
    return val

#@jit
def _calc_Xika_inner(inputs0):
    r"""
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion. Inner loop for use with jit

    Parameters
    ----------
    inputs : tuple
        Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol

    Returns
    -------
    inputs : tuple
        Xika_elements, indices, rho[r], xi, nui, nk, Fklab, Kklab, Iij[r], damp, tol
    """

    Xika_elements0, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol = inputs0

    l_ind = indices.shape[0]

    Xika_elements0 = index_update(Xika_elements0,index[1],Xika_elements0[0])
    Xika_elements0 = index_update(Xika_elements0,index[0,:],1.0)

    # New Way #################
    inputs = lax.fori_loop(0,l_ind,_Xika_update_2a,(Xika_elements0, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol))
    Xika_elements, _, _, _, _, _, _, _, _, _, _ = inputs
    # Old Way #################
    #for ind in range(l_ind):
    #    i, k, a = indices[ind]
    #    for jnd in range(l_ind):
    #        j, l, b = indices[jnd]
    #        delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * Iij[i, j]
    #        tmp = Xika_elements[0][ind] + rho * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[1][jnd] * delta
    #        Xika_elements = index_update(Xika_elements,index[0,ind],tmp)
    ###########################

    Xika_elements = index_update(Xika_elements,index[0], np.reciprocal(Xika_elements[0]))

    Xika_diff = np.subtract(Xika_elements[0], Xika_elements[1])
    obj = np.sum(np.abs(Xika_diff))
    pred = np.divide(obj,np.amax(Xika_elements[1]))
    tmp = np.add(Xika_elements[1],np.multiply(damp,Xika_diff))
    Xika_elements = lax.cond(pred > 1e+3,Xika_elements,lambda x: index_update(x,index[0],tmp),Xika_elements,lambda x: x)

    inputs = (Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol)

    return inputs

#@jit
def _Xika_update_2a(ii,inputs0):

    Xika_elements0, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol = inputs0

    inputs = lax.fori_loop(0,indices.shape[0],_Xika_update_2b,(ii,Xika_elements0, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol))
    _, Xika_elements, _, _, _, _, _, _, _, _, _, _ = inputs

    input_out = (Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol)

    return input_out

#@jit
def _Xika_update_2b( jj, inputs0):

    ii, Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol = inputs0

    i, k, a = indices[ii]
    j, l, b = indices[jj]

    delta = np.multiply(Fklab[k, l, a, b], np.multiply(Kklab[k, l, a, b],Iij[i, j]))
    tmp0 = np.multiply(rho, np.multiply( xi[j], np.multiply( nui[j,l], np.multiply(nk[l,b],Xika_elements[1][jj]))))
    tmp = np.add(Xika_elements[0][ii], np.multiply(tmp0, delta))
    Xika_elements = index_update(Xika_elements,index[0,ii],tmp)

    input_out = ii, Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij, damp, tol

    return input_out


#@jit
def _calc_Xika_cond(inputs):
    r"""
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion. Inner loop for use with jit

    Parameters
    ----------
    Xika_elements : [numpy.ndarray]
        Initial guess in Xika values for each index in indicies

    Returns
    -------
    obj : bool
        Whether array met criteria
    """

    Xika_elements, _, _, _, _, _, _, _, _, _, tol = inputs
    obj = np.greater(np.sum(np.abs(np.subtract(Xika_elements[0], Xika_elements[1]))),tol)

    return obj

def calc_A_assoc(rho, xi, T, nui, xskl, sigmakl, sigmaii_avg, epsilonii_avg, epsilonHB, Kklab, nk):
    r"""
    Calculates the association contribution of the Helmholtz energy, :math:`A^{assoc.}`.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    sigmaii_avg : numpy.ndarray
        Average bead (i.e. group or segment) size in component (i.e. molecule) i.
    epsilonii_avg : numpy.ndarray
        Average bead (i.e. group or segment) potential well depth in component (i.e. molecule) i.
    epsilonHB : numpy.ndarray
        Interaction energy between each bead and association site.
    Kklab : numpy.ndarray 
        Bonding volume between each association site
    nk : numpy.ndarray
        For each bead the number of each type of site

    Returns
    -------
    Aassoc : numpy.ndarray
        Association site contribution of Helmholtz energy, length of array rho
    """

    logger = logging.getLogger(__name__)

    kT = T * constants.kb
    nbeads = list(nui.shape)[1]
    ncomp = np.size(xi)
    nsitesmax = np.size(nk, axis=1)
    Fklab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    epsilonij = np.zeros((ncomp, ncomp))
    Iij = np.zeros((np.size(rho), ncomp, ncomp))
    delta = np.zeros((np.size(rho), ncomp, ncomp, nbeads, nbeads, nsitesmax, nsitesmax))
    Xika = np.zeros((np.size(rho), ncomp, nbeads, nsitesmax))
    Aassoc = np.zeros_like(rho)

    # compute F_klab
    Fklab = np.exp(epsilonHB * constants.kb / kT) - 1.0

    # compute epsilonij
    for i in range(ncomp):
        for j in range(i, ncomp):
            tmp = np.sqrt(sigmaii_avg[i] * sigmaii_avg[j])**3.0 * np.sqrt(epsilonii_avg[i] * epsilonii_avg[j]) / (((sigmaii_avg[i] + sigmaii_avg[j]) / 2.0)**3)
            epsilonij = index_update(epsilonij,index[i, j],tmp)
            epsilonij = index_update(epsilonij,index[j, i],epsilonij[i, j])

    # compute sigmax3
    sigmax3 = np.sum(xskl * (sigmakl**3))

    # compute Iijklab 
    # {BottleNeck}
    for p in range(11):
        for q in range(11 - p):
            #Iij += np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3 * rho)**p), ((kT / epsilonij)**q))
            if p == 0: Iij += np.einsum("i,jk->ijk", constants.cij[p, q] * np.ones(len(rho)), ((kT / epsilonij)**q))
            elif p == 1: Iij += np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3 * rho)), ((kT / epsilonij)**q))
            elif p == 2: 
               rho2 = rho**2
               Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho2)), ((kT / epsilonij)**q))
            elif p == 3: 
                rho3 = rho2*rho
                Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho3)), ((kT / epsilonij)**q))
            elif p == 4: 
                rho4 = rho2**2
                Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho4)), ((kT / epsilonij)**q))
            elif p == 5: 
                rho5 = rho2*rho3
                Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho5)), ((kT / epsilonij)**q))
            elif p == 6: Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho*rho5)), ((kT / epsilonij)**q))
            elif p == 7: Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho2*rho5)), ((kT / epsilonij)**q))
            elif p == 8: Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho3*rho5)), ((kT / epsilonij)**q))
            elif p == 9: Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho4*rho5)), ((kT / epsilonij)**q))
            elif p == 10: Iij = Iij + np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3**p * rho5*rho5)), ((kT / epsilonij)**q))

        indices = assoc_site_indices(xi, nui, nk)
        #calc_Xika_jit = jit(calc_Xika,static_argnums=(0,2,3,4,5,6))
        #Xika, err_array = calc_Xika(indices,rho, xi, nui, nk, Fklab, Kklab, Iij)
        Xika = calc_Xika(indices,rho, xi, nui, nk, Fklab, Kklab, Iij)

    # Compute A_assoc
    for i in range(ncomp):
        for k in range(nbeads):
            for a in range(nsitesmax):
                if nk[k, a] != 0.0:
                    Aassoc = Aassoc + xi[i] * nui[i, k] * nk[k, a] * (np.log(Xika[:, i, k, a]) +
                                                              ((1.0 - Xika[:, i, k, a]) / 2.0))
    print(Aassoc)

    return Aassoc

def assoc_site_indices(xi, nui, nk):
    r""" 
    Make a list of sets of indices that allow quick identification of the relevant association sights. This is needed for solving Xika, the fraction of molecules of component i that are not bonded at a site of type a on group k.

    Parameters
    ----------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    nk : numpy.ndarray
        For each bead the number of each type of site

    Returns
    -------
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    """

    logger = logging.getLogger(__name__)

    lx, ly, lz = len(xi),len(nk),len(nk[0])
    indices_tmp = np.zeros(lx*ly*lz)

    # Indices of components will minimal mole fractions
    tol = 1e-32
    zero_frac = np.asarray(xi<tol)
    l_tmp = 0

    for i, comp in enumerate(nui):
        if not zero_frac[i]:
            for j, bead in enumerate(comp):
                if (bead != 0 and np.any(nk[j]!=0.0)):
                    for k in range(len(nk[j])):
                        if nk[j][k] != 0.0:
                            ind = i*(ly*lz) + j*lz + k
                            indices_tmp = index_update(indices_tmp,index[ind],1.)
                            l_tmp += 1

    indices = np.zeros((l_tmp,3),int)
    bool_assoc = np.asarray(indices_tmp==1.)
    j = 0
    for i in range(lx*ly*lz):
        if bool_assoc[i]:
            i_tmp = float(i)
            ii, tmp = i_tmp // float(ly*lz), i_tmp % float(ly*lz)
            jj, kk = tmp // float(lz), tmp % float(lz)
            indices = index_update(indices,index[j],np.array([ii,jj,kk],int))
            j += 1

    return indices

def obj_Xika(Xika_elements, indices, rho, xi, nui, nk, Fklab, Kklab, Iij):
    r""" 
    Calculate the fraction of molecules of component i that are not bonded at a site of type a on group k in an iterative fashion.

    Parameters
    ----------
    Xika_elements : numpy.ndarray
        A guess in the value of the elements
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    rho : float
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    nk : numpy.ndarray
        For each bead the number of each type of site
    delta : numpy.ndarray
        The association strength between a site of type a on a group of type k of component i and a site of type b on a group of type l of component j. eq. 66, for a specific density
    Fklab : numpy.ndarray
        
    Kklab : numpy.ndarray
        
    Iij : numpy.ndarray
        

    Returns
    -------
    obj : numpy.ndarray
        The sum of the absolute difference between each of the target elements.
    """

    logger = logging.getLogger(__name__)

    ## Option 1: A lot of for loops
#    nbeads    = nui.shape[1]
#    ncomp     = np.size(xi)
#    nsitesmax = np.size(nk, axis=1)
#
#    Xika = np.ones((ncomp, nbeads, nsitesmax))
#    Xika0 = assemble_Xika(Xika_elements,indices,Xika)
#    for i in range(ncomp):
#        for k in range(nbeads):
#            for a in range(nsitesmax):
#
#                for j in range(ncomp):
#                    for l in range(nbeads):
#                        for b in range(nsitesmax):
#                            Xika[i,k,a] += (rho * xi[j] * nui[j,l] * nk[l,b] * Xika0[j,l,b] * delta[i,j,k,l,a,b])
#    Xika = 1./Xika
#    Xika_elements_new = [Xika[i][j][k] for i,j,k in indices]

    ## Option 2: Fewer forloops
    Xika_elements_new = np.ones(len(Xika_elements))
    Xika_elements = np.array(Xika_elements)
    ind = 0
    for i,k,a in indices:
        jnd = 0
        for j,l,b in indices:
            delta = Fklab[k, l, a, b] * Kklab[k, l, a, b] * Iij[i, j]
            tmp = Xika_elements_new + rho * xi[j] * nui[j,l] * nk[l,b] * Xika_elements[jnd] * delta
            Xika_elements_new = index_update(Xika_elements_new,index[ind],tmp)
            jnd = jnd + 1
        ind = ind + 1
    Xika_elements_new = 1./Xika_elements_new

    obj = (Xika_elements_new - Xika_elements)/Xika_elements

    #logger.debug("    Xika: {}, Error: {}".format(Xika_elements_new,obj))

    return obj

def assemble_Xika(Xika_elements,indices,Xika):
    r""" 
    Put matrix values back into template matrix according to sets of indices.

    Parameters
    ----------
    Xika_elements : numpy.ndarray
        A guess in the value of the elements
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    Xika : numpy.ndarray
        A template matrix of the fraction of molecules of component i that are not bonded at a site of type a on group k.

    Returns
    -------
    Xika : numpy.ndarray
        The final matrix of the fraction of molecules of component i that are not bonded at a site of type a on group k,
    """

    logger = logging.getLogger(__name__)

    if len(Xika_elements) != len(indices):
        raise ValueError("Number of elements should each have a corresponding set of indices.")
        logger.exception("Number of elements should each have a corresponding set of indices.")

    for j,ind in enumerate(indices):
        i,k,a = ind
        Xika = index_update(Xika,index[i,k,a],Xika_elements[j])

    return Xika

############################################################
#                                                          #
#            Total A, Helmholtz Free Energy                #
#                                                          #
############################################################

def calc_A(rho, xi, T, beads, beadlibrary, massi, nui, Cmol2seg, xsk, xskl, dkk, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl,epsilonHB, Kklab, nk):
    r"""
    Calculates total Helmholtz energy, :math:`\frac{A}{N k_{B} T}`.

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    massi : numpy.ndarray
        Mass for each component [kg/mol]
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xsk : numpy.ndarray
        Mole fraction of each bead (i.e. segment or group), sum(xsk) should equal 1.0
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    dkk : numpy.ndarray
        Array of hard sphere diameters for each group
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
    l_akl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    l_rkl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    Ckl : numpy.ndarray
        Matrix of mie potential prefactors for k,l groups
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, sigmakl is the mie radius for groups (k,l)
    epsilonHB : numpy.ndarray
        Interaction energy between each bead and association site.
    Kklab : numpy.ndarray 
        Bonding volume between each association site
    nk : numpy.ndarray
        For each bead the number of each type of site

    Returns
    -------
    A : numpy.ndarray
        Total Helmholtz energy, length of array rho
    """

    logger = logging.getLogger(__name__)

    if any(np.array(xi) < 0.):
        raise ValueError("Mole fractions cannot be less than zero.")

    Aideal = calc_Aideal(xi, rho, massi, T)
    AHS, A1, A2, A3, zetax, zetaxstar, KHS = calc_Amono(rho, xi, nui, Cmol2seg, xsk, xskl, dkk, T,epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl)
    Achain, sigmaii_avg, epsilonii_avg = calc_Achain(rho, Cmol2seg, xi, T, nui, sigmakl, epsilonkl, dkl, xskl, l_rkl, l_akl, beads, beadlibrary, zetax, zetaxstar, KHS)

    indices = assoc_site_indices(xi, nui, nk)
    if indices.size != 0:
        tmp = 0
        for i,k,a in indices:
            for j,l,b in indices:
                tmp = tmp + epsilonHB[k, l, a, b]
    else:
        tmp = 0.

    if tmp != 0.:
        Aassoc = calc_A_assoc(rho, xi, T, nui, xskl, sigmakl, sigmaii_avg, epsilonii_avg, epsilonHB, Kklab, nk)
        A = Aideal + AHS + A1 + A2 + A3 + Achain + Aassoc
        
    else:
        A = Aideal + AHS + A1 + A2 + A3 + Achain

    return A


def calc_Ares(rho, xi, T, beads, beadlibrary, massi, nui, Cmol2seg, xsk, xskl, dkk, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl, epsilonHB, Kklab, nk):
    r"""
    Calculates residual Helmholtz energy, :math:`\frac{A^{res.}}{N k_{B} T}` that deviates from ideal

    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [molecules/m^3]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    massi : numpy.ndarray
        Mass for each component [kg/mol]
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component. 
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xsk : numpy.ndarray
        Mole fraction of each bead (i.e. segment or group), sum(xsk) should equal 1.0
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    dkk : numpy.ndarray
        Array of hard sphere diameters for each group
    epsilonkl : numpy.ndarray
        Matrix of well depths for groups (k,l)
    sigmakl : numpy.ndarray
        Matrix of mie diameter for groups (k,l)
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
    l_akl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    l_rkl : numpy.ndarray
        Matrix of mie potential attractive exponents for k,l groups
    Ckl : numpy.ndarray
        Matrix of mie potential prefactors for k,l groups
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, sigmakl is the mie radius for groups (k,l)
    epsilonHB : numpy.ndarray
        Interaction energy between each bead and association site.
    Kklab : numpy.ndarray 
        Bonding volume between each association site
    nk : numpy.ndarray
        For each bead the number of each type of site

    Returns
    -------
    Ares : numpy.ndarray
        Residual Helmholtz energy that deviates from Aideal, length of array rho
    """

    logger = logging.getLogger(__name__)

    if any(np.array(xi) < 0.):
        raise ValueError("Mole fractions cannot be less than zero.")

    AHS, A1, A2, A3, zetax, zetaxstar, KHS = calc_Amono(rho, xi, nui, Cmol2seg, xsk, xskl, dkk, T, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl)
    Achain, sigmaii_avg, epsilonii_avg = calc_Achain(rho, Cmol2seg, xi, T, nui, sigmakl, epsilonkl, dkl, xskl, l_rkl, l_akl, beads, beadlibrary, zetax, zetaxstar, KHS)

    indices = assoc_site_indices(xi, nui, nk)
    if indices.size != 0:
        tmp = 0
        for i,k,a in indices:
            for j,l,b in indices:
                tmp = tmp + epsilonHB[k, l, a, b]
    else:
        tmp = 0.

    if tmp != 0.:
        Aassoc = calc_A_assoc(rho, xi, T, nui, xskl, sigmakl, sigmaii_avg, epsilonii_avg, epsilonHB, Kklab, nk)
        Ares = AHS + A1 + A2 + A3 + Achain + Aassoc

    else:
        Ares = AHS + A1 + A2 + A3 + Achain

    return Ares

