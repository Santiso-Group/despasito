
import sys
import numpy as np
import scipy.optimize as spo
import logging

from despasito.equations_of_state import constants
from despasito.equations_of_state.saft.compiled_modules.nojit_exts import prefactor

logger = logging.getLogger(__name__)

def remove_insignificant_components(xi_old,massi_old):
    """
    This function will remove any components with mole fractions less than or equal to machine precision.

    Parameters
    ----------
    xi_old : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi_old : numpy.ndarray
        Mass for each component [kg/mol]

   Returns
    -------
    xi_new : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi_new : numpy.ndarray
        Mass for each component [kg/mol]

    """
    ind = np.where(np.array(xi_old)<np.finfo(float).eps)[0]
    xi_new = []
    massi_new = []
    for i in range(len(xi_old)):
        if i not in ind:
            xi_new.append(xi_old[i])
            massi_new.append(massi_old[i])
    xi_new = np.array(xi_new)
    massi_new = np.array(massi_new)

    return xi_new, massi_new

def partial_density_central_difference(xi, rho, T, func, step_size=1E-2, log_method=False):
    """
    Take the derivative of a dependent variable calculated with a given function using the central difference method.
    
    Parameters
    ----------
    xi : list[float]
        Mole fraction of each component
    rho : float
        Molar density of system [mol/m^3]
    T : float
        Temperature of the system [K]
    func : function
        Function used in job to calculate dependent factor. This function should have a single output. Inputs arguements should be (rho, T, xi)
    step_size : float, Optional, default: 1E-4
        Step size used in central difference method
    log_method : bool, Optional, default: False
        Choose to use a log transform in central difference method. This allows easier calulations for very small numbers.
        
    Returns
    -------
    dydxi : numpy.ndarray
        Array of derivative of y with respect to xi
    """
    
    dAdrho = np.zeros(len(xi))

    if log_method: # Central Difference Method with log(y) transform

        y = np.log(rho*np.array(xi,float))

        dy = step_size
        for i in range(np.size(dAdrho)):
            if xi[i] != 0.0:
                Ares = np.zeros(2)
                for j, delta in enumerate((dy, -dy)):
                    y_temp = np.copy(y)
                    y_temp[i] += delta
                    Ares[j] = _partial_density_wrapper(np.exp(y_temp), T, func)
                dAdrho[i] = (Ares[0] - Ares[1]) / (2.0 * dy) / np.exp(y[i])
            else:
                dAdrho[i] = np.finfo(float).eps

    else: # Traditional Central Difference Method
        
        dy = step_size
        y = rho*np.array(xi,float)
        for i in range(np.size(dAdrho)):
            if xi[i] != 0.0:
                Ares = np.zeros(2)
                for j, delta in enumerate((dy, -dy)):
                    y_temp = np.copy(y)
                    if y_temp[i] != 0.:
                        y_temp[i] += delta
                    Ares[j] = _partial_density_wrapper(y_temp, T, func)
                dAdrho[i] = (Ares[0] - Ares[1]) / (2.0 * dy)
            else:
                dAdrho[i] = np.finfo(float).eps

    return dAdrho

def _partial_density_wrapper(rhoi, T, func):
    """
    Compute derivative of Helmholtz energy wrt to density.
    
    Parameters
    ----------
    rhoi : float
        Molar density of each component, add up to the total density [mol/m^3]
    T : float
        Temperature of the system [K]
    func : function
        Function used in job to calculate dependent factor. This function should have a single output.
    
    Returns
    -------
    Ares : float
        Helmholtz energy give number of moles, length of array rho
    """
    
    # Calculate new xi values
    rho = np.array([np.sum(rhoi)])
    xi = rhoi/rho
    
    Ares = func(rho, T, xi)
    
    return Ares

def calc_massi(nui, beadlibrary, beads):
    r"""
    This function extracted the mass of each component
    
    Parameters
    ----------
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
        
        - mass: Bead mass [kg/mol]
    
    beads : list[str]
        List of unique bead names used among components
    
    Returns
    -------
    massi : numpy.ndarray
        Bead mass corresponding to array 'beads' [kg/mol]
    """
    massi = np.zeros(len(nui))
    for i in range(len(nui)):
        for k, bead in enumerate(beads):
            if "mass" in beadlibrary[bead]:
                massi[i] += nui[i, k] * beadlibrary[bead]["mass"]
            else:
                raise ValueError("The mass for bead, {}, was not provided.".format(bead))

    return massi

def extract_property(prop, beadlibrary, beads):
    r"""
    
    
    Parameters
    ----------
    property : str
        Name of property in beadlibrary
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:
    beads : list[str]
        List of unique bead names used among components
    
    Returns
    -------
    prop_array : numpy.ndarray
        array of desired property
    """
    prop_array = np.zeros(len(beads))
    for i , bead in enumerate(beads):
        if prop in beadlibrary[bead]:
                prop_array[i] += beadlibrary[bead][prop]
        else:
            raise ValueError("The property {} for bead, {}, was not provided.".format(prop,bead))

    return prop_array

def calc_hard_sphere_matricies(T, sigmakl, beadlibrary, beads):
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
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k
    
    beads : list[str]
        List of unique bead names used among components
        
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
        dkk[i] = calc_dkk(beadlibrary[beads[i]]["epsilon"], beadlibrary[beads[i]]["sigma"], T,
                                   beadlibrary[beads[i]]["l_r"], beadlibrary[beads[i]]["l_a"])
    dkl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            dkl[k, l] = (dkk[k] + dkk[l]) / 2.0

    x0kl = sigmakl / dkl

    return dkl, x0kl

def _dkk_int(r, Ce_kT, sigma, l_r, l_a):
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
    l_r : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    l_a : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    
    Returns
    -------
    dkk_int_tmp : numpy.ndarray
        Integrand used to calculate the hard sphere diameter
    """
        
    dkk_int_tmp = 1.0 - np.exp(-Ce_kT * (np.power(sigma / r, l_r) - np.power(sigma / r, l_a)))
        
    return dkk_int_tmp

def calc_dkk(epsilon, sigma, T, l_r, l_a=6.0):
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
    l_r : float
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    l_a : float, Optional, default: 6.0
        :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
    
    Returns
    -------
    dkk : float
        Hard sphere diameter of a group [Å]
    """
        
    Ce_kT = prefactor(l_r, l_a) * epsilon / T
    # calculate integral of dkk_int from 0.0 to sigma

    ## Option 1
    #results = integrate.quad(lambda r: _dkk_int(r, Ce_kT, sigma, l_r, l_a), 0.0, sigma, epsabs=1.0e-16, epsrel=1.0e-16)
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
    dkk = 0.5*sigma*np.sum(w*_dkk_int(r, Ce_kT, sigma, l_r, l_a))

    # Option 3: Mullers method
    #xgl = np.array([0.97390652852, 0.86506336669, 0.67940956830, 0.43339539413, 0.14887433898])
    #wgl = np.array([0.06667134431, 0.14945134915, 0.21908636252, 0.26926671931, 0.29552422471])
    #x1i = 0.5 * (1 + xgl)
    #x2i = 0.5 * (1 - xgl)

    #xsum = 0.
    #for i in range(len(x1i)):
    #    tmp1 = np.power(x1i[i], -l_r) - np.power(x1i[i], -l_a)
    #    tmp2 = np.power(x2i[i], -l_r) - np.power(x2i[i], -l_a) 
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
        Matrix of mole fractions of bead (i.e. segment or group) k multiplied by bead l
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
        xsk[k] = np.sum(xi * nui[:, k]) * beadlibrary[beads[k]]["Vks"] * \
            beadlibrary[beads[k]]["Sk"]
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
    # calc zetax eq. 22
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
    # compute KHS(rho), eq. 31
    KHS = ((1.0 - zetax)**4) / (1.0 + (4.0 * zetax) + (4.0 * (zetax**2)) - (4.0 * (zetax**3)) + (zetax**4))

    return KHS

def calc_interaction_matrices(beads, beadlibrary, crosslibrary={}):
    r"""
    Computes matrices of cross interaction parameters epsilonkl, sigmakl, l_akl, or l_rkl, depending on what is given in beadlibrary
    
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
    
    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the     desired parameters may be defined for whichever group combinations are desired. If this matrix isn't provided, the SAFT mixing rules are used.
        
        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        - l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
        
    Returns
    -------
    output : dict
        Dictionary of outputs, the following possibilities aer calculated if all relevant beads have those properties.

        - epsilonkl : numpy.ndarray, Matrix of well depths for groups (k,l)
        - sigmakl : numpy.ndarray, Matrix of Mie diameter for groups (k,l)
        - l_akl : numpy.ndarray, Matrix of Mie potential attractive exponents for k,l groups
        - l_rkl : numpy.ndarray, Matrix of Mie potential attractive exponents for k,l groups

    """
    
    nbeads = len(beads)
    
    output = {}
    if all([True if "epsilon" in beadlibrary[bead] else False for bead in beads]):
        output["epsilonkl"] = np.zeros((nbeads, nbeads))
    if all([True if "sigma" in beadlibrary[bead] else False for bead in beads]):
        output["sigmakl"] = np.zeros((nbeads, nbeads))
    if all([True if "l_r" in beadlibrary[bead] else False for bead in beads]):
        output["l_rkl"] = np.zeros((nbeads, nbeads))
    if all([True if "l_a" in beadlibrary[bead] else False for bead in beads]):
        output["l_akl"] = np.zeros((nbeads, nbeads))

    # compute default interaction parameters for beads
    for k in range(nbeads):
        for l in range(nbeads):
            if "sigmakl" in output:
                output["sigmakl"][k, l] = (beadlibrary[beads[k]]["sigma"] + beadlibrary[beads[l]]["sigma"]) / 2.0
            else:
                logger.warning("Not all of the beads, {}, have values for sigma".format(beads))

            if "l_rkl" in output:
                output["l_rkl"][k, l] = 3 + np.sqrt((beadlibrary[beads[k]]["l_r"] - 3.0) * (beadlibrary[beads[l]]["l_r"] - 3.0))
            else:
                logger.warning("Not all of the beads, {}, have values for l_r".format(beads))

            if "l_akl" in output:
                output["l_akl"][k, l] = 3 + np.sqrt((beadlibrary[beads[k]]["l_a"] - 3.0) * (beadlibrary[beads[l]]["l_a"] - 3.0))
            else:
                logger.warning("Not all of the beads, {}, have values for l_a".format(beads))

            if "epsilonkl" in output:
                if "sigmakl" in output:
                    output["epsilonkl"][k, l] = np.sqrt(beadlibrary[beads[k]]["epsilon"] * beadlibrary[beads[l]]["epsilon"]) * np.sqrt((beadlibrary[beads[k]]["sigma"] ** 3) * (beadlibrary[beads[l]]["sigma"] ** 3)) / (output["sigmakl"][k, l] ** 3)
                else:
                    logger.warning("Size parameters are not available to weight the energy parameters".format(beads))
                    epsilonkl[k, l] = np.sqrt(beadlibrary[beads[k]]["epsilon"] * beadlibrary[beads[l]]["epsilon"])
            else:
                logger.warning("Not all of the beads, {}, have values for epsilon".format(beads))

    # testing if crosslibrary is empty ie not specified
    if crosslibrary:
        # find any cross terms in the cross term library
        crosslist = []
    
        for (i, beadname) in enumerate(beads):
            if beadname in crosslibrary:
                for (j, beadname2) in enumerate(beads):
                    if beadname2 in crosslibrary[beadname]:
                        crosslist.append([i, j])
    
        for i in range(np.size(crosslist, axis=0)):
            a = crosslist[i][0]
            b = crosslist[i][1]
            if beads[a] in crosslibrary:
                if beads[b] in crosslibrary[beads[a]]:
                    if "epsilonkl" in output and "epsilon" in crosslibrary[beads[a]][beads[b]]:
                        output["epsilonkl"][a, b] = crosslibrary[beads[a]][beads[b]]["epsilon"]
                        output["epsilonkl"][b, a] = output["epsilonkl"][a, b]
                    if "l_rkl" in output and "l_r" in crosslibrary[beads[a]][beads[b]]:
                        output["l_rkl"][a, b] = crosslibrary[beads[a]][beads[b]]["l_r"]
                        output["l_rkl"][b, a] = output["l_rkl"][a, b]
                    if "l_akl" in output and "l_a" in crosslibrary[beads[a]][beads[b]]:
                        output["l_akl"][a, b] = crosslibrary[beads[a]][beads[b]]["l_a"]
                        output["l_akl"][b, a] = output["l_akl"][a, b]
                    if "sigmakl" in output and "sigma" in crosslibrary[beads[a]][beads[b]]:
                        output["sigmakl"][a, b] = crosslibrary[beads[a]][beads[b]]["sigma"]
                        output["sigmakl"][b, a] = output["sigmakl"][a, b]

    return output

def calc_component_averaged_properties(nui, Vks, Sk, epsilonkl=None, sigmakl=None, l_akl=None, l_rkl=None):
    r"""
    
    
    Parameters
    ----------
    nui : numpy.array
        :math:`\nu_{i,k}/k_B`, Array of number of components by number of bead types. Defines the number of each type of group in each component.
        Defined for eq. 11. Note that indices are flipped from definition in reference.
    Vks : numpy.ndarray
        :math:`V_{k,s}`, Number of groups, k, in component
    Sk : numpy.ndarray
        :math:`S_{k}`, Shape parameter of group k
    epsilonkl : numpy.ndarray, Optional, default: None
        Matrix of well depths for groups (k,l)
    sigmakl : numpy.ndarray, Optional, default: None
        Matrix of Mie diameter for groups (k,l)
    l_akl : numpy.ndarray, Optional, default: None
        Matrix of Mie potential attractive exponents for k,l groups
    l_rkl : numpy.ndarray, Optional, default: None
        Matrix of Mie potential attractive exponents for k,l groups
        
    Returns
    -------
    output : dict
        Dictionary of outputs, the following possibilities aer calculated if all relevant beads have those properties.

        - epsilonii_avg : numpy.ndarray, Matrix of well depths for groups (k,l)
        - sigmaii_avg : numpy.ndarray, Matrix of Mie diameter for groups (k,l)
        - l_aii_avg : numpy.ndarray, Matrix of Mie potential attractive exponents for k,l groups
        - l_rii_avg : numpy.ndarray, Matrix of Mie potential attractive exponents for k,l groups

    """
    
    ncomp, nbeads = np.shape(nui)
    zki = np.zeros((ncomp, nbeads), float)
    zkinorm = np.zeros(ncomp, float)
    
    output = {}
    if epsilonkl is not None:
        output['epsilonii_avg'] = np.zeros(ncomp, float)

    if sigmakl is not None:
        output['sigmaii_avg'] = np.zeros(ncomp, float)

    if l_rkl is not None:
        output['l_rii_avg'] = np.zeros(ncomp, float)

    if l_akl is not None:
        output['l_aii_avg'] = np.zeros(ncomp, float)

    #compute zki
    for i in range(ncomp):
        for k in range(nbeads):
            zki[i, k] = nui[i, k] * Vks[k] * Sk[k]
            zkinorm[i] += zki[i, k]

    for i in range(ncomp):
        for k in range(nbeads):
            zki[i, k] = zki[i, k] / zkinorm[i]

    for i in range(ncomp):
        for k in range(nbeads):
            for l in range(nbeads):
                if 'sigmaii_avg' in output:
                    output['sigmaii_avg'][i] += zki[i, k] * zki[i, l] * sigmakl[k, l]**3
                if 'epsilonii_avg' in output:
                    output['epsilonii_avg'][i] += zki[i, k] * zki[i, l] * epsilonkl[k, l] * constants.kb
                if 'l_rii_avg' in output:
                    output['l_rii_avg'][i] += zki[i, k] * zki[i, l] * l_rkl[k, l]
                if 'l_aii_avg' in output:
                    output['l_aii_avg'][i] += zki[i, k] * zki[i, l] * l_akl[k, l]
        if 'sigmaii_avg' in output:
            output['sigmaii_avg'][i] = output['sigmaii_avg'][i]**(1/3.0)

    return output

