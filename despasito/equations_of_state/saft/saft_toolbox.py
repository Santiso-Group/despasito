
import numpy as np
import logging

from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)

def calc_hard_sphere_matricies(T, sigmakl, beadlibrary, beads, Cprefactor_funcion):
    r"""
    Computes matrix of hard sphere interaction parameters dkk, dkl, and x0kl.
    
    This does not include function specific or association terms.
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    sigmakl : numpy.ndarray
        Matrix of Mie diameter for groups (k,l)
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
        
        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - lambdar: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - lambdaa: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
    
    beads : list[str]
        List of unique bead names used among components
    Cprefactor_funcion : function
        Function used to calculate prefactor for potential
        
    Returns
    -------
    dkl : numpy.ndarray
        Matrix of hard sphere diameters for groups (k,l)
    x0kl : numpy.ndarray
        Matrix of sigmakl/dkl, sigmakl is the Mie radius for groups (k,l)
    """

    nbeads = np.size(beads)
    dkk = np.zeros(nbeads)
    for i in np.arange(nbeads):
        prefactor = Cprefactor_funcion(beadlibrary[beads[i]]["lambdar"], beadlibrary[beads[i]]["lambdaa"])
        dkk[i] = calc_dkk(beadlibrary[beads[i]]["epsilon"], beadlibrary[beads[i]]["sigma"], T, prefactor, beadlibrary[beads[i]]["lambdar"], beadlibrary[beads[i]]["lambdaa"])
    dkl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            dkl[k, l] = (dkk[k] + dkk[l]) / 2.0

    x0kl = sigmakl / dkl

    return dkl, x0kl

def _dkk_int(r, Ce_kT, sigma, lambdar, lambdaa):
    r"""
    Return integrand used to calculate the hard sphere diameter.
    
    :math:`d_{k,k}` of a group k. See eq. 10.
    
    Parameters
    ----------
    r : numpy.ndarray
        Bead distance between zero and :math:`sigma_{k,k}` [Å]
    Ce_kT : float
        :math:`C \epsilon_{k,k}/(k_B T)`, Mie prefactor scaled by kT
    sigma : float
        :math:`\sigma_{k,k}`, Size parameter [Å] (or same units as r)
    lambdar : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    lambdaa : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    
    Returns
    -------
    dkk_int_tmp : numpy.ndarray
        Integrand used to calculate the hard sphere diameter
    """
        
    dkk_int_tmp = 1.0 - np.exp(-Ce_kT * (np.power(sigma / r, lambdar) - np.power(sigma / r, lambdaa)))
        
    return dkk_int_tmp

def calc_dkk(epsilon, sigma, T, Cprefactor, lambdar, lambdaa=6.0):
    r"""
    Calculates hard sphere diameter of a group k, :math:`d_{k,k}`. Defined in eq. 10.
    
    Parameters
    ----------
    epsilon : float
        :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant [K]
    sigma : float
        :math:`\sigma_{k,k}`, Size parameter [Å] (or same units as r)
    T : float
        Temperature of the system [K]
    Cprefactor : float
        Prefactor for chosen potential
    lambdar : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    lambdaa : float, Optional, default: 6.0
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    
    Returns
    -------
    dkk : float
        Hard sphere diameter of a group [Å]
    """
        
    Ce_kT = Cprefactor * epsilon / T
    # calculate integral of dkk_int from 0.0 to sigma

    ## Option 1
    #results = integrate.quad(lambda r: _dkk_int(r, Ce_kT, sigma, lambdar, lambdaa), 0.0, sigma, epsabs=1.0e-16, epsrel=1.0e-16)
    #results = results[0]
    
    ## Option 2: 10pt Gauss Legendre
    # 5pt
    #x = np.array([0.0, 0.5384693101056831, -0.5384693101056831, 0.906179845938664, -0.906179845938664])
    #w = np.array([0.568889, 0.47862867049936647, 0.47862867049936647, 0.23692688505618908, 0.23692688505618908])
    # 10pt
    #w = np.array([0.295524225, 0.295524225, 0.269266719, 0.269266719, 0.219086363, 0.219086363, 0.149451349, 0.149451349, 0.066671344, 0.066671344])
    #x = np.array([-0.148874339, 0.148874339, -0.433395394, 0.433395394, -0.679409568, 0.679409568, -0.865063367, 0.865063367, -0.973906529, 0.973906529])
    # 40pt
    w = np.array([0.077505948, 0.077505948, 0.077039818, 0.077039818, 0.076110362, 0.076110362, 0.074723169, 0.074723169, 0.072886582, 0.072886582, 0.070611647, 0.070611647, 0.067912046, 0.067912046, 0.064804013, 0.064804013, 0.061306242, 0.061306242, 0.057439769, 0.057439769, 0.053227847, 0.053227847, 0.048695808, 0.048695808, 0.043870908, 0.043870908, 0.038782168, 0.038782168, 0.033460195, 0.033460195, 0.027937007, 0.027937007, 0.022245849, 0.022245849, 0.016421058, 0.016421058, 0.010498285, 0.010498285, 0.004521277, 0.004521277])
    x = np.array([-0.038772418, 0.038772418, -0.116084071, 0.116084071, -0.192697581, 0.192697581, -0.268152185, 0.268152185, -0.341994091, 0.341994091, -0.413779204, 0.413779204, -0.483075802, 0.483075802, -0.549467125, 0.549467125, -0.61255389, 0.61255389, -0.671956685, 0.671956685, -0.727318255, 0.727318255, -0.778305651, 0.778305651, -0.824612231, 0.824612231, -0.865959503, 0.865959503, -0.902098807, 0.902098807, -0.932812808, 0.932812808, -0.957916819, 0.957916819, -0.97725995, 0.97725995, -0.990726239, 0.990726239, -0.99823771, 0.99823771])
    
    r = 0.5*sigma*(x+1)
    dkk = 0.5*sigma*np.sum(w*_dkk_int(r, Ce_kT, sigma, lambdar, lambdaa))

    
    # Option 3: Mullers method
    #xgl = np.array([0.97390652852, 0.86506336669, 0.67940956830, 0.43339539413, 0.14887433898])
    #wgl = np.array([0.06667134431, 0.14945134915, 0.21908636252, 0.26926671931, 0.29552422471])
    #x1i = 0.5 * (1 + xgl)
    #x2i = 0.5 * (1 - xgl)
    #
    #xsum = 0.
    #for i in range(len(x1i)):
    #    tmp1 = np.power(x1i[i], -lambdar) - np.power(x1i[i], -lambdaa)
    #    tmp2 = np.power(x2i[i], -lambdar) - np.power(x2i[i], -lambdaa) 
    #    xsum = xsum + wgl[i] * ((x1i[i]**2 * np.exp(-tmp1 * Ce_kT)) + (x2i[i]**2 * np.exp(-tmp2 * Ce_kT)))
    #dcube_s = 1 - (3 / 2 * xsum)
    #dkk = sigma*pow(dcube_s, 1/3)
    
    return dkk

def calc_composition_dependent_variables(xi, nui, beadlibrary, beads):
    r"""
    
    Parameters
    ----------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component.
        Defined for eq. 11. Note that indices are flipped from definition in reference.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
    
    beads : list[str]
        List of unique bead names used among components
    
    Returns
    -------
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by that of bead l
    """
    
    # compute Conversion factor
    Cmol2seg = 0.0
    for i in range(np.size(xi)):
        for j in range(np.size(beads)):
            Cmol2seg += xi[i] * nui[i, j] * beadlibrary[beads[j]]["Vks"] * beadlibrary[beads[j]]["Sk"]

    # initialize variables and arrays
    nbeads = len(beads)
    xsk = np.zeros(nbeads, float)
    # compute xsk
    for k in range(nbeads):
        xsk[k] = np.sum(xi * nui[:, k]) * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"]
    xsk /= Cmol2seg

    # calculate  xskl matrix
    xskl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            xskl[k, l] = xsk[k] * xsk[l]

    return Cmol2seg, xskl

def calc_zetaxstar(rho, Cmol2seg, xskl, sigmakl):
    r"""
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    sigmakl : numpy.ndarray
        Matrix of Mie diameter for groups (k,l)
    
    Returns
    -------
    zetaxstar : numpy.ndarray
        Matrix of hypothetical packing fraction based on sigma for groups (k,l)
    """
    
    # compute zetaxstar eq. 35
    zetaxstar = rho * Cmol2seg * ((np.pi / 6.0) * np.sum(xskl * (sigmakl**3 * constants.molecule_per_nm3)))

    return zetaxstar

def calc_zetax(rho, Cmol2seg, xskl, dkl):
    r"""
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    Cmol2seg : float
        Conversion factor from from molecular number density, :math:`\rho`, to segment (i.e. group) number density, :math:`\rho_S`. Shown in eq. 13
    xskl : numpy.ndarray
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
    dkl : numpy.ndarray
        Matrix of hardsphere diameters for groups (k,l)
    
    Returns
    -------
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    """
    zetax = rho * Cmol2seg * ((np.pi / 6.0) * np.sum(xskl * (dkl**3 * constants.molecule_per_nm3)))

    return zetax

def calc_KHS(zetax):
    r"""
    
    Parameters
    ----------
    zetax : numpy.ndarray
        Matrix of hypothetical packing fraction based on hard sphere diameter for groups (k,l)
    
    Returns
    -------
    KHS : numpy.ndarray
        (length of densities) isothermal compressibility of system with packing fraction zetax
    """
    KHS = ((1.0 - zetax)**4) / (1.0 + (4.0 * zetax) + (4.0 * (zetax**2)) - (4.0 * (zetax**3)) + (zetax**4))

    return KHS


