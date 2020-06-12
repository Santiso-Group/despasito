# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import scipy.optimize as spo
import os
import sys

import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)

# Check for Numba
if 'NUMBA_DISABLE_JIT' in os.environ:
    disable_jit = os.environ['NUMBA_DISABLE_JIT']
else:
    from .. import jit_stat
    disable_jit = jit_stat.disable_jit
# Check for cython
from .. import cython_stat
disable_cython = cython_stat.disable_cython

flag_fortran = False
if disable_jit and disable_cython:
    try:
        from .compiled_modules import ext_Aassoc_fortran
        flag_fortran = True
    except:
        logger.info("Fortran module failed to import, using pure python. Consider using 'jit' flag")
        from .compiled_modules.ext_Aassoc_python import calc_Xika
elif not disable_cython:
    #from .compiled_modules.ext_Aassoc_cython_2 import calc_Xika
    from .compiled_modules.ext_Aassoc_cython import calc_Xika
else:
    from .compiled_modules.ext_Aassoc_numba import calc_Xika

def calc_Xika_wrap(*args, maxiter=500, tol=1e-12, damp=0.1):
    r""" This function wrapper allows difference types of compiled functions to be referenced.
    """
    if flag_fortran:
        indices, rho, xi, nui, nk, Fklab, Kklab, gr_assoc = args
        Xika_init = 0.5*np.ones(len(indices))
        Xika = ext_Aassoc_fortran.calc_xika(indices,constants.molecule_per_nm3*rho,Xika_init,xi,nui,nk,Fklab,Kklab,gr_assoc,maxiter,tol)
    else:
        Xika, _ = calc_Xika(*args)

    print("hey!",np.shape(Xika),Xika)

    return Xika

def assoc_site_indices(nk, nui, xi=None):
    r"""
    Make a list of sets of indices that allow quick identification of the relevant association sights.
    
    This is needed for solving Xika, the fraction of molecules of component i that are not bonded at a site of type a on group k.
    
    Parameters
    ----------
    nk : numpy.ndarray
        A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    xi : numpy.ndarray, Optional, default: None
        Mole fraction of each component, sum(xi) should equal 1.0
    
    Returns
    -------
    indices : list[list]
        A list of sets of (component, bead, site) to identify the values of the Xika matrix that are being fit
    """

    indices = []

    # List of site indices for each bead type
    bead_sites = []
    for bead in nk:
        bead_sites.append([i for i, site in enumerate(bead) if site != 0])

    # Indices of components will minimal mole fractions
    if xi is not None:
        zero_frac = np.where(np.array(xi)<np.finfo("float").eps)[0]
    else:
        zero_frac = np.array([])

    for i, comp in enumerate(nui):
        if i not in zero_frac:
            for j, bead in enumerate(comp):
                if (bead != 0 and bead_sites[j]):
                    for k in bead_sites[j]:
                        indices.append([i,j,k])

    indices = np.array([np.array(x) for x in indices], dtype=np.int)

    return indices

def initiate_assoc_matrices(beads, beadlibrary, nui):
    r"""
    
    Generate matrices used for association site calculations.
    
    Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk (number of sites )
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.

    Returns
    ----------
    sitenames : list
        This list shows the names of the various association types found
    nk : numpy.ndarray
        A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
    flag_assoc : bool
        If True, this flag indicates that association sites play a role in this system.
    """

    sitenames = []
    nk = [[] for i in range(len(beads))]

    for i, bead in enumerate(beads):
        nk[i] = [0 for x in sitenames]
        for key, value in beadlibrary[bead].items():
            if "Nk" in key:
                tmp = key.split("-")
                if len(tmp) < 2:
                    raise ValueError("Association site names should be defined with hyphens (e.g. Nk-H)")
                else:
                    _, site = tmp 

                if site not in sitenames:
                    for j in range(i):
                        nk[j].append(0)
                    nk[i].append(value)
                    sitenames.append(site)
                else:
                    ind = sitenames.index(site)
                    nk[i][ind] = value
                logger.debug("Bead {} has {} of the association site {}".format(bead,value,site))

    indices = assoc_site_indices(nk, nui)
    if indices.size == 0:
        flag_assoc = False
    else:
        flag_assoc = True

    if flag_assoc:
        logger.info("The following association sites have been identified: {}".format(sitenames))
    else:
        logger.info("No association sites are used in this system.")

    return sitenames, np.array(nk), flag_assoc

def calc_assoc_matrices(beads, beadlibrary, nui, crosslibrary={}, nk=None, sitenames=None):
    r"""
    
    Generate matrices used for association site calculations.
    
    Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk (number of sites )

    Note: Some papers use r instead of Kklab, provide function to calculate Kklab in that case (see Papaioannou 2014)

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    crosslibrary : dict
        A dictionary where bead names are the keys to access a dictionary of a second tier of bead names. This structure contains the EOS cross interaction parameters:

        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    nk : numpy.ndarray
        A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
    sitenames : list
        This list shows the names of the various association types found
    
    Returns
    -------
    epsilonHB : numpy.ndarray
        Interaction energy between each bead and association site.
    Kklab : numpy.ndarray
        Bonding volume between each association site
    """

    nbeads = len(beads)
    if sitenames is None or nk is None:
        sitenames, nk, _ = initiate_assoc_matrices(beadlibrary, beads, nui)
    else:
        nsitesmax = len(sitenames)
    epsilonHB = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    Kklab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))

    # self-interaction
    for i, nk1 in enumerate(nk):
        bead1 = beads[i]
        for a, site1 in enumerate(sitenames):

            if nk1[a] == 0.0:
                continue
       
            for b, site2 in enumerate(sitenames):

                if nk1[b] != 0:
                    epsilon_tmp = "-".join(["epsilonHB",site1,site2])
                    K_tmp = "-".join(["K",site1,site2])
                    
                    if epsilon_tmp in beadlibrary[bead1]:
                        epsilonHB[i, i, a, b] = beadlibrary[bead1][epsilon_tmp]
                        epsilonHB[i, i, b, a] = epsilonHB[i, i, a, b]

                    if K_tmp in beadlibrary[bead1]:
                        Kklab[i, i, a, b] = beadlibrary[bead1][K_tmp]
                        Kklab[i, i, b, a] = Kklab[i, i, a, b]

    # cross-interaction
    for i, nk1 in enumerate(nk):
        bead1 = beads[i]
        for a, site1 in enumerate(sitenames):
            if nk1[a] == 0.0 or bead1 not in crosslibrary:
                continue

            for b, site2 in enumerate(sitenames):
                epsilon_tmp = "-".join(["epsilonHB",site1,site2])
                K_tmp = "-".join(["K",site1,site2])
                for j, bead2 in enumerate(beads):
                    if i == j:
                        continue

                    flag_update = False
                    if bead2 in crosslibrary[bead1]:
                        # Update matrix if found in crosslibrary
                        if epsilon_tmp in crosslibrary[bead1][bead2]:
                            if (nk[i][a] == 0 or nk[j][b] == 0):
                                if 0 not in [nk[i][b],nk[j][a]]:
                                    logger.warning("Site names were listed in the wrong order for parameter definitions in cross interaction library. Changing {}_{} - {}_{} interaction to {}_{} - {}_{}".format( beads[i], sitenames[a], beads[j], sitenames[b], beads[i], sitenames[b], beads[j], sitenames[a]))
                                    a, b = [b, a]
                                elif nk[i][a] == 0:
                                    raise ValueError("Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format( beads[i], sitenames[a], beads[j], sitenames[b],beads[i], sitenames[a]))
                                elif nk[j][b] == 0:
                                    raise ValueError("Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format(beads[i], sitenames[a], beads[j], sitenames[b], beads[j], sitenames[b]))

                            flag_update = True
                            epsilonHB[i,j,a,b] = crosslibrary[bead1][bead2][epsilon_tmp]
                            epsilonHB[j,i,b,a] = epsilonHB[i,j,a,b]
    
                            Kklab[i,j,a,b] = crosslibrary[bead1][bead2][K_tmp]
                            Kklab[j,i,b,a] = Kklab[i,j,a,b]

                    if not flag_update and nk[j][b] != 0 and epsilonHB[j,i,b,a] == 0.0:
                        epsilonHB[i,j,a,b] = np.sqrt(epsilonHB[i,i,a,a]*epsilonHB[j,j,b,b])
                        epsilonHB[j,i,b,a] = epsilonHB[i,j,a,b]

                        Kklab[i,j,a,b] = (((Kklab[i,i,a,a])**(1.0/3.0)+(Kklab[j,j,b,b])**(1.0/3.0))/2.0)**3
                        Kklab[j,i,b,a] = Kklab[i,j,a,b]
                    
    return epsilonHB, Kklab

