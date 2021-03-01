# -- coding: utf8 --

r"""
    
EOS object for SAFT ideal gas contributions to the Helmholtz energy
    
"""

import numpy as np
import logging
import sys

from despasito.equations_of_state import constants
import despasito.equations_of_state.eos_toolbox as tb

logger = logging.getLogger(__name__)


def Aideal_contribution(rho, T, xi, massi, method="Abroglie"):

    r"""
    Return a vector of ideal contribution of Helmholtz energy.
    
    :math:`\frac{A^{ideal}}{N k_{B} T}`
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi : numpy.ndarray
        Vector of component masses that correspond to the mole fractions in xi [kg/mol]
    method : str, Optional, default=Abroglie
        The function name of the method to calculate the ideal contribution of the Helmholtz energy. To add a new one, add a function to: despasito.equations_of_state.saft.Aideal.py
    
    Returns
    -------
    Aideal : numpy.ndarray
        Helmholtz energy of ideal gas for each density given.
    """

    functions = {"Abroglie": Abroglie}

    if method in functions:
        function = functions[method]
    else:
        raise ValueError(
            "Method, {}, was not found to calculate Aideal.".format(method)
        )

    return function(rho, T, xi, massi)


def Abroglie(rho, T, xi, massi):

    r"""
    Return a vector of ideal contribution of Helmholtz energy derived from Broglie wavelength
    
    :math:`\frac{A^{ideal}}{N k_{B} T}`
    
    Parameters
    ----------
    rho : numpy.ndarray
        Number density of system [mol/m^3]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    massi : numpy.ndarray
        Vector of component masses that correspond to the mole fractions in xi [kg/mol]
    
    Returns
    -------
    Aideal : numpy.ndarray
        Helmholtz energy of ideal gas for each density given.
    """

    rho2 = rho * constants.molecule_per_nm3

    xi_tmp, massi_tmp = tb.remove_insignificant_components(xi, massi)

    # rhoi: (number of components,number of densities) number density of each component for each density

    rhoi = np.outer(rho2, xi_tmp)
    Lambda = np.sqrt(
        (constants.h * constants.Nav * constants.m2nm)
        * (constants.h / constants.kb * constants.m2nm)
        / (2.0 * np.pi * massi_tmp * T)
    )
    log_broglie3_rho = np.log(Lambda ** 3 * rhoi)

    if np.isnan(np.sum(np.sum(xi_tmp * log_broglie3_rho, axis=1))):
        raise ValueError(
            "Aideal has values of zero when taking the log. All mole fraction values should be nonzero. Mole fraction: {}".format(
                xi_tmp
            )
        )
    else:
        Aideal = np.sum(xi_tmp * log_broglie3_rho, axis=1) - 1.0

    return Aideal
