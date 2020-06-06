
import numpy as np
import logging

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

        dy = step_size
        y = np.log(rho*np.array(xi,float))
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


