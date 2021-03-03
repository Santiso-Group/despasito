# -- coding: utf8 --

r"""
    
EOS object for SAFT association sites contributions to the Helmholtz energy
    
"""

import numpy as np
import logging
import scipy.optimize as spo
import os
import sys

import despasito.equations_of_state.eos_toolbox as tb
from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)

try:
    from .compiled_modules import ext_Aassoc_fortran
    flag_fortran = True
except Exception:
    flag_fortran = False
    logger.warning("Fortran unavailable, using Numba")

from .compiled_modules.ext_Aassoc_cython import calc_Xika as calc_Xika_cython
from .compiled_modules.ext_Aassoc_numba import calc_Xika as calc_Xika_numba
from .compiled_modules.ext_Aassoc_python import calc_Xika as calc_Xika_python


def _calc_Xika_wrap(*args, method_stat, maxiter=500, tol=1e-12, damp=0.1):
    r""" This function wrapper allows difference types of compiled functions to be referenced.
    """

    indices, rho, xi, molecular_composition, nk, Fklab, Kklab, gr_assoc = args

    if len(np.shape(Kklab)) == 4:
        if method_stat.fortran and flag_fortran:
            Xika_init = 0.5 * np.ones(len(indices))
            Xika = ext_Aassoc_fortran.calc_xika_4(
                indices,
                constants.molecule_per_nm3 * rho,
                Xika_init,
                xi,
                molecular_composition,
                nk,
                Fklab,
                Kklab,
                gr_assoc,
                maxiter,
                tol,
            )
        else:
            if method_stat.numba or not flag_fortran:
                Xika, _ = calc_Xika_numba(*args)
            elif method_stat.cython:
                Xika, _ = calc_Xika_cython(*args)
            elif method_stat.python:
                Xika, _ = calc_Xika_python(*args)
                logger.warning("Using pure python. Consider using 'numba' flag")

    elif len(np.shape(Kklab)) == 6:
        if method_stat.fortran or flag_fortran:
            Xika_init = 0.5 * np.ones(len(indices))
            Xika = ext_Aassoc_fortran.calc_xika_6(
                indices,
                constants.molecule_per_nm3 * rho,
                Xika_init,
                xi,
                molecular_composition,
                nk,
                Fklab,
                Kklab,
                gr_assoc,
                maxiter,
                tol,
            )
        else:
            if method_stat.numba or not flag_fortran:
                Xika, _ = calc_Xika_numba(*args)
            elif method_stat.cython:
                Xika, _ = calc_Xika_cython(*args)
            elif method_stat.python:
                Xika, _ = calc_Xika_python(*args)
                logger.warning("Using pure python. Consider using 'numba' flag")

    return Xika


def assoc_site_indices(nk, molecular_composition, xi=None):
    r"""
    Make a list of sets of indices that allow quick identification of the relevant association sights.
    
    This is needed for solving Xika, the fraction of molecules of component i that are not bonded at a site of type a on group k.
    
    Parameters
    ----------
    nk : numpy.ndarray
        A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    xi : numpy.ndarray, Optional, default=None
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

    ## Indices of components will minimal mole fractions
    # if xi is not None:
    #    zero_frac = np.where(np.array(xi)<np.finfo("float").eps)[0]
    # else:
    #    zero_frac = np.array([])

    for i, comp in enumerate(molecular_composition):
        # if i not in zero_frac:
        for j, bead in enumerate(comp):
            if bead != 0 and bead_sites[j]:
                for k in bead_sites[j]:
                    indices.append([i, j, k])

    indices = np.array([np.array(x) for x in indices], dtype=np.int)

    return indices


def initiate_assoc_matrices(beads, bead_library, molecular_composition):
    r"""
    
    Generate matrices used for association site calculations.
    
    Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk (number of sites )
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - Nk\*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    molecular_composition : numpy.ndarray
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
        for key, value in bead_library[bead].items():
            if key.startswith("Nk"):
                tmp = key.split("-")
                if len(tmp) < 2:
                    raise ValueError(
                        "Association site names should be defined with hyphens (e.g. Nk-H)"
                    )
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
                logger.debug(
                    "Bead {} has {} of the association site {}".format(
                        bead, value, site
                    )
                )

    indices = assoc_site_indices(nk, molecular_composition)
    if indices.size == 0:
        flag_assoc = False
    else:
        flag_assoc = True

    if flag_assoc:
        logger.info(
            "The following association sites have been identified: {}".format(sitenames)
        )
    else:
        logger.info("No association sites are used in this system.")

    return sitenames, np.array(nk), flag_assoc


def calc_assoc_matrices(
    beads,
    bead_library,
    molecular_composition,
    cross_library={},
    nk=None,
    sitenames=None,
):
    r"""
    
    Generate matrices used for association site calculations.
    
    Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume, nk (number of sites )

    Note: Some papers use r instead of Kklab, provide function to calculate Kklab in that case (see Papaioannou 2014)

    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    bead_library : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilonHB-\*-\*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K-\*-\*: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - rc-\*-\*: Optional, Cutoff distance for association sites. Asterisk represents two strings from sitenames.
        - rd-\*-\*: Optional, Site position. Asterisk represents two strings from sitenames.
        - Nk-\*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    molecular_composition : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    cross_library : dict, Optional, default={}
        A dictionary where bead names are the keys to access a dictionary of a second tier of bead names. This structure contains the EOS cross interaction parameters:

        - epsilonHB-\*-\*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K-\*-\*: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - rc-\*-\*: Optional, Cutoff distance for association sites. Asterisk represents two strings from sitenames.
        - rd-\*-\*: Optional, Site position. Asterisk represents two strings from sitenames.

    nk : numpy.ndarray, Optional, default=None
        A matrix of (Nbeads x Nsites) Contains for each bead the number of each type of site
    }
    sitenames : list, Optional, default=None
        This list shows the names of the various association types found
    
    Returns
    -------
    output_dict : dict
        This dictionary contains parameters relevant to calculating association site contributions. The following matrices may be inside, each of the size (ngroups, ngroups, nsites, nsites).

        - epsilonHB: Interaction energy between each bead and association site.
        - Kklab, Optional: Bonding volume between each association site
        - rc_klab, Optional: Cutoff distance for association sites
        - rd_klab, Optional: Association site position

    """

    nbeads = len(beads)
    if np.any(sitenames == None) or np.any(nk == None):
        sitenames, nk, _ = initiate_assoc_matrices(
            bead_library, beads, molecular_composition
        )
    else:
        nsitesmax = len(sitenames)
    epsilonHB = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    Kklab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    rc_klab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    rd_klab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))

    flag_Kklab = False
    flag_rc_klab = False
    flag_rd_klab = False
    # self-interaction
    for i, nk1 in enumerate(nk):
        bead1 = beads[i]
        for a, site1 in enumerate(sitenames):

            if nk1[a] == 0.0:
                continue

            for b, site2 in enumerate(sitenames):

                if nk1[b] != 0:
                    epsilon_tmp = "-".join(["epsilonHB", site1, site2])
                    K_tmp = "-".join(["K", site1, site2])
                    rc_tmp = "-".join(["rc", site1, site2])
                    rd_tmp = "-".join(["rd", site1, site2])

                    if epsilon_tmp in bead_library[bead1] and (
                        K_tmp not in bead_library[bead1]
                        and rc_tmp not in bead_library[bead1]
                    ):
                        raise ValueError(
                            "An association site energy parameter for {} was given for bead {}, but not the bonding information. Either K-sitename-sitename or rc-sitename-sitename must be given.".format(
                                "{}-{}".format(site1, site2), bead1
                            )
                        )
                    elif K_tmp in bead_library[bead1] and rc_tmp in bead_library[bead1]:
                        raise ValueError(
                            "Both association site bonding volumes and cutoff distances were provided for bead {}. This is redundant.".format(
                                bead1
                            )
                        )
                    elif epsilon_tmp not in bead_library[bead1] and (
                        K_tmp in bead_library[bead1] or rc_tmp in bead_library[bead1]
                    ):
                        raise ValueError(
                            "An association site bonding information for {} was given for bead {}, but not the energy parameter. epsilonHB must be given.".format(
                                "{}-{}".format(site1, site2), bead1
                            )
                        )

                    if epsilon_tmp in bead_library[bead1]:
                        epsilonHB[i, i, a, b] = bead_library[bead1][epsilon_tmp]
                        epsilonHB[i, i, b, a] = epsilonHB[i, i, a, b]
                    else:
                        continue

                    if K_tmp in bead_library[bead1]:
                        flag_Kklab = True
                        Kklab[i, i, a, b] = bead_library[bead1][K_tmp]
                        Kklab[i, i, b, a] = Kklab[i, i, a, b]

                    if rc_tmp in bead_library[bead1]:
                        flag_rc_klab = True
                        rc_klab[i, i, a, b] = bead_library[bead1][rc_tmp]
                        rc_klab[i, i, b, a] = rc_klab[i, i, a, b]

                    if rd_tmp in bead_library[bead1]:
                        flag_rd_klab = True
                        rd_klab[i, i, a, b] = bead_library[bead1][rd_tmp]
                        rd_klab[i, i, b, a] = rd_klab[i, i, a, b]

    # cross-interaction
    for i, nk1 in enumerate(nk):
        bead1 = beads[i]
        for a, site1 in enumerate(sitenames):
            if nk1[a] == 0.0 or bead1 not in cross_library:
                continue

            for b, site2 in enumerate(sitenames):
                epsilon_tmp = "-".join(["epsilonHB", site1, site2])
                K_tmp = "-".join(["K", site1, site2])
                for j, bead2 in enumerate(beads):
                    if i == j:
                        continue

                    flag_update = False
                    if bead2 in cross_library[bead1]:
                        # Update matrix if found in cross_library
                        if epsilon_tmp in cross_library[bead1][bead2]:
                            if nk[i][a] == 0 or nk[j][b] == 0:
                                if 0 not in [nk[i][b], nk[j][a]]:
                                    logger.warning(
                                        "Site names were listed in the wrong order for parameter definitions in cross interaction library. Changing {}_{} - {}_{} interaction to {}_{} - {}_{}".format(
                                            beads[i],
                                            sitenames[a],
                                            beads[j],
                                            sitenames[b],
                                            beads[i],
                                            sitenames[b],
                                            beads[j],
                                            sitenames[a],
                                        )
                                    )
                                    a, b = [b, a]
                                elif nk[i][a] == 0:
                                    raise ValueError(
                                        "Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format(
                                            beads[i],
                                            sitenames[a],
                                            beads[j],
                                            sitenames[b],
                                            beads[i],
                                            sitenames[a],
                                        )
                                    )
                                elif nk[j][b] == 0:
                                    raise ValueError(
                                        "Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format(
                                            beads[i],
                                            sitenames[a],
                                            beads[j],
                                            sitenames[b],
                                            beads[j],
                                            sitenames[b],
                                        )
                                    )

                            flag_update = True
                            epsilonHB[i, j, a, b] = cross_library[bead1][bead2][
                                epsilon_tmp
                            ]
                            epsilonHB[j, i, b, a] = epsilonHB[i, j, a, b]

                            if flag_Kklab:
                                Kklab[i, j, a, b] = cross_library[bead1][bead2][K_tmp]
                                Kklab[j, i, b, a] = Kklab[i, j, a, b]

                    if (
                        not flag_update
                        and nk[j][b] != 0
                        and epsilonHB[j, i, b, a] == 0.0
                    ):
                        epsilonHB[i, j, a, b] = np.sqrt(
                            epsilonHB[i, i, a, a] * epsilonHB[j, j, b, b]
                        )
                        epsilonHB[j, i, b, a] = epsilonHB[i, j, a, b]
                        if flag_Kklab:
                            Kklab[i, j, a, b] = (
                                (
                                    (Kklab[i, i, a, a]) ** (1.0 / 3.0)
                                    + (Kklab[j, j, b, b]) ** (1.0 / 3.0)
                                )
                                / 2.0
                            ) ** 3
                            Kklab[j, i, b, a] = Kklab[i, j, a, b]

    output = {"epsilonHB": epsilonHB}
    if flag_Kklab:
        output["Kklab"] = Kklab
    elif flag_rc_klab:
        output["rc_klab"] = rc_klab
    elif flag_rd_klab:
        output["rd_klab"] = rd_klab

    if flag_Kklab and flag_rc_klab:
        raise ValueError(
            "Both association site bonding volumes and cutoff distances were provided. This is redundant."
        )
    elif flag_rd_klab and not flag_rc_klab:
        raise ValueError(
            "Association site position were provided, but not cutoff distances."
        )

    return output


def calc_bonding_volume(rc_klab, dij_bar, rd_klab=None, reduction_ratio=0.25):
    """
    Calculate the association site bonding volume matrix 

    Dimensions of (ncomp, ncomp, nbeads, nbeads, nsite, nsite)

    Parameters
    ----------
    rc_klab : numpy.ndarray
        This matrix of cutoff distances for association sites for each site type in each group type
    dij_bar : numpy.ndarray
        Component averaged hard sphere diameter
    rd_klab : numpy.ndarray, Optional, default=None
        Position of association site in each group (nbead, nbead, nsite, nsite)
    reduction_ratio : float, Optional, default=0.25
        Reduced distance of the sites from the center of the sphere of interaction. This value is used when site position, rd_klab is None

    Returns
    -------
    Kijklab : numpy.ndarray
        Matrix of binding volumes
    """

    ncomp = len(dij_bar)
    nbead, _, nsite, _ = np.shape(rc_klab)
    Kijklab = np.zeros((ncomp, ncomp, nbead, nbead, nsite, nsite))

    for i in range(ncomp):
        for j in range(ncomp):
            for k in range(nbead):
                for l in range(nbead):
                    for a in range(nsite):
                        for b in range(nsite):
                            if rc_klab[k, l, a, b] != 0:
                                if rd_klab == None:
                                    rd = reduction_ratio * dij_bar[i, j]
                                else:
                                    rd = rd_klab[k, l, a, b]

                                tmp0 = np.pi * dij_bar[i, j] ** 2 / (18 * rd ** 2)
                                tmp11 = np.log(
                                    (rc_klab[k, l, a, b] + 2 * rd) / dij_bar[i, j]
                                )
                                tmp12 = (
                                    6 * rc_klab[k, l, a, b] ** 3
                                    + 18 * rc_klab[k, l, a, b] ** 2 * rd
                                    - 24 * rd ** 3
                                )
                                tmp21 = rc_klab[k, l, a, b] + 2 * rd - dij_bar[i, j]
                                tmp22 = (
                                    22 * rd ** 2
                                    - 5 * rd * rc_klab[k, l, a, b]
                                    - 7 * rd * dij_bar[i, j]
                                    - 8 * rc_klab[k, l, a, b] ** 2
                                    + rc_klab[k, l, a, b] * dij_bar[i, j]
                                    + dij_bar[i, j] ** 2
                                )

                                Kijklab[i, j, k, l, a, b] = tmp0 * (
                                    tmp11 * tmp12 + tmp21 * tmp22
                                )

    return Kijklab
