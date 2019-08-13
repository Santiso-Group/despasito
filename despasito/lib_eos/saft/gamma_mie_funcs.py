"""
    despasito
    DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output
    
    Routines for calculating the Helmoltz energy for the SAFT-gamma equation of state.
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
from scipy import integrate
import scipy.optimize as spo

from . import constants
from . import Achain as Ac
from . import solv_assoc


def calc_Aideal(xi, rho, massi, T):
    """ Return a vector of Aideal/(N*kb*T) (number of densities) as defined in eq. 4 
        Input:
            xi: (number of components) numpy array of mol fractions sum(xi) should equal 1.0
            rho: (number of densities) number density of system N/V in m^-3
            massi: (number of components) numpy array of mass for each component in kg/mol
            T: scalar Temperature in Kelvin
        Uses:
            numpy
            kb, Nav, and h from constants.py
    """

    # rhoi: (number of components,number of densities) number density of each component for each density
    rhoi = np.outer(rho, xi)
    Lambda3 = (constants.h / np.sqrt(2.0 * np.pi * (massi / constants.Nav) * constants.kb * T))**3
    Aideal_tmp = rhoi*Lambda3
    if not any(np.sum(xi * np.log(Aideal_tmp), axis=1)):
        print(np.array(Aideal_tmp).T)
        print("lambda",Lambda3)
        print(xi, massi)
        Aideal = []
        for a in Aideal_tmp:
            if not any(np.sum(xi * np.log(a), axis=1)): 
                Aideal.append(np.sum(xi * np.log(a), axis=1) - 1.0)
            else:
                print("Aideal",a)
                Aideal.append(0.0)
        Aideal = np.array(Aideal)
    else:
        Aideal = np.sum(xi * np.log(Aideal_tmp), axis=1) - 1.0

    return Aideal


def dkk_int(r, Ce_kT, sigma, l_r, l_a):
    """ Return integrand of eq. 10 for a value r
        Input:
            r: bead distance
            Ce_kT: C*epsilon/(kT), Mie prefactor normalized by kT
            sigma: bead diameter in Angstroms (or same units as r)
            l_r: repulsive exponent Mie potential
            l_a: attractive exponent Mie potential
    """
    return 1.0 - np.exp(-Ce_kT * (np.power(sigma / r, l_r) - np.power(sigma / r, l_a)))


def calc_dkk(epsilon, sigma, T, l_r, l_a=6.0):
    """ Returns dkk defined in eq. 10
        Input:
            epsilon: Mie potential energy well depth, epsilon/kb in units Kelvin  
            sigma: sigma: bead diameter in Angstroms (or same units as r)
            T: Temperature in Kelvin
            l_r: repulsive exponent Mie potential
            l_a: attractive exponent Mie potential
        Uses:
            scipy.integrate
            C(l_r,l_a) from  py (this file)
    """

    Ce_kT = C(l_r, l_a) * epsilon / T
    # calculate integral of dkk_int from 0.0 to sigma
    results = integrate.quad(lambda r: dkk_int(r, Ce_kT, sigma, l_r, l_a), 0.0, sigma, epsabs=1.0e-16, epsrel=1.0e-16)

    return results[0]


def calc_Amonopre(xi, nui, beads, beadlibrary):
    """ Return Amonopre, where rhos=rho*Amonopre in eq. 13
        Input:
            xi: (number of components) numpy array of mol fractions sum(xi) should equal 1.0
            nui: (number of components,number of bead types) numpy array, list of bead quantites in each component 
                defined for eq. 11. Note that indicies are flipped from definition in reference.
            beads: list of strings of unique bead types used in any of the components
            beadlibrary: dictionary of bead parameters, where items in beads are the keys for the dictionary
    """
    Amonopre = 0.0
    for i in range(np.size(xi)):
        for j in range(np.size(beads)):
            Amonopre += xi[i] * nui[i, j] * beadlibrary[beads[j]]["Vks"] * beadlibrary[beads[j]]["Sk"]

    return Amonopre


def calc_Bkl(rho, l_kl, Amonopre, dkl, epsilonkl, x0kl, etax):  # calc_Bkl(dkl,x0kl,epsilonkl,l_kl,etax,rho,Amonopre):
    """ Return Bkl(rho*Amonopre,l_kl) in K as defined in eq. 20
        Input:
            rho: (number of densities) number density of system N/V in m^-3
            l_kl: (nbead,nbead) numpy array of mie potential exponents for k,l groups
            Amonopre: constant computed by calc_Amonopre
            dkl: (nbead,nbead) numpy array of hardsphere diameters for groups (k,l)
            epsilonkl: (nbead,nbead) numpy array of well depths for groups (k,l)
            x0kl: (nbead,nbead) numpy array of sigmakl/dkl, sigmakl is the mie radius for groups (k,l)
            etax: (nrho) numpy array of hypothetical packing fraction
    """
    # initialize Ikl and Jkl
    # Ikl = np.zeros_like(l_kl)
    # Jkl = np.zeros_like(l_kl)

    # compute Ikl(l_kl), eq. 23
    Ikl = (1.0 - (x0kl**(3.0 - l_kl))) / (l_kl - 3.0)
    # compute Jkl(l_kl), eq. 24
    Jkl = (1.0 - ((x0kl**(4.0 - l_kl)) * (l_kl - 3.0)) + ((x0kl**(3.0 - l_kl)) * (l_kl - 4.0))) / ((l_kl - 3.0) *
                                                                                                   (l_kl - 4.0))

    if np.size(np.shape(l_kl)) == 2:
        # Bkl=np.zeros((np.size(rho),np.size(l_kl,axis=0),np.size(l_kl,axis=0)))

        Bkl = np.einsum("i,jk", rho * (Amonopre * 2.0 * np.pi),
                        (dkl**3) * epsilonkl) * (np.einsum("i,jk", (1.0 - (etax / 2.0)) / (
                            (1.0 - etax)**3), Ikl) - np.einsum("i,jk", ((9.0 * etax * (1.0 + etax)) /
                                                                        (2.0 * ((1 - etax)**3))), Jkl))
    elif np.size(np.shape(l_kl)) == 1:
        Bkl = np.einsum("i,j", rho * (Amonopre * 2.0 * np.pi),
                        (dkl**3) * epsilonkl) * (np.einsum("i,j", (1.0 - (etax / 2.0)) / (
                            (1.0 - etax)**3), Ikl) - np.einsum("i,j", ((9.0 * etax * (1.0 + etax)) /
                                                                       (2.0 * ((1 - etax)**3))), Jkl))
    else:
        print('Error unexpeced l_kl shape in Bkl')

    # Bkl = rho * Amonopre * 2.0 * np.pi * (dkl**3) * epsilonkl * \
    #      ((Ikl*(1.0-(etax/2.0))/((1.0-etax)**3)) - ((Jkl*9.0*etax*(1.0+etax))/(2.0*((1-etax)**3))))

    return Bkl


def calc_a1s(rho, Amonopre, l_kl, etax, epsilonkl, dkl):
    """ Return a1s,kl(rho*Amonopre,l_kl) in K as defined in eq. 25
        Input:
            rho: (number of densities) number density of system N/V in m^-3
            Amonopre: constant computed by calc_Amonopre
            l_kl: (nbead,nbead) numpy array of mie potential exponents for k,l groups
            etax: hypothetical packing fraction
            epsilonkl: (nbead,nbead) numpy array of well depths for groups (k,l)
            dkl: (nbead,nbead) numpy array of hardsphere diameters for groups (k,l)
    """

    nbeads = np.size(dkl, axis=0)
    etax_pow = np.zeros((np.size(rho), 4))
    etax_pow[:, 0] = etax
    for i in range(1, 4):
        etax_pow[:, i] = etax_pow[:, i - 1] * etax_pow[:, 0]

    # check if you have more than 1 bead types
    if np.size(np.shape(l_kl)) == 2:
        etakl = np.zeros((np.size(rho), nbeads, nbeads))
        for k in range(nbeads):
            for l in range(nbeads):
                cikl = np.inner(constants.ckl_coef, np.array([1.0, l_kl[k, l]**-1, l_kl[k, l]**-2, l_kl[k, l]**-3]).T)
                etakl[:, k, l] = np.einsum("ij,j", etax_pow, cikl)

        a1s = np.einsum("ijk,jk->ijk", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Amonopre * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        a1s = np.einsum("i,ijk->ijk", rho, a1s)

    elif np.size(np.shape(l_kl)) == 1:
        etakl = np.zeros((np.size(rho), nbeads))
        for k in range(nbeads):
            cikl = np.inner(constants.ckl_coef, np.array([1.0, l_kl[k]**-1, l_kl[k]**-2, l_kl[k]**-3]).T)
            etakl[:, k] = np.einsum("ij,j", etax_pow, cikl)
        a1s = np.einsum("ij,j->ij", (1.0 - (etakl / 2.0)) / ((1.0 - etakl)**3),
                        -2.0 * np.pi * Amonopre * ((epsilonkl * (dkl**3)) / (l_kl - 3.0)))
        a1s = np.einsum("i,ij->ij", rho, a1s)
    else:
        print('Error in calc_a1s, unexpected array size')

    return a1s


def calc_fm(alphakl, mlist):
    """ Returns numpy array of fm(alphakl) where a list of m values are specified in mlist eq. 39
        Inputs:
            alphakl: (nbead,nbead) "a dimensionless form of the integrated vdW energy of the Mie potential" eq. 33
            mlist: (number of m values) a numpy array of integers
    """
    nbeads = np.size(alphakl, axis=0)
    if np.size(np.shape(alphakl)) == 2:
        fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0), np.size(alphakl, axis=0)))
    elif np.size(np.shape(alphakl)) == 1:
        fmlist = np.zeros((np.size(mlist), np.size(alphakl, axis=0)))
    else:
        print('Error: unexpected shape in calcfm')
    mlist = mlist - 1

    for i, m in enumerate(mlist):
        for n in range(4):
            fmlist[i] += constants.phimn[m, n] * (alphakl**n)
        dum = np.ones_like(fmlist[i])
        for n in range(4, 7):
            dum += constants.phimn[m, n] * (alphakl**(n - 3.0))
        fmlist[i] = fmlist[i] / dum

    return fmlist


def calc_interaction_matrices(beads, beadlibrary, crosslibrary={}):
    """
    Computes matrix of interaction parameters epsilonkl,sigmakl,l_akl,l_rkl (attractive and repulsive exponents), Ckl,
    using the beadlibrary (read from a file).
    This does not include association terms
    If non default mixing parameters are need (epsilon and l_r only) specify the crosslibrary dictionary
    """
    nbeads = len(beads)
    sigmakl = np.zeros((nbeads, nbeads))
    l_rkl = np.zeros((nbeads, nbeads))
    l_akl = np.zeros((nbeads, nbeads))
    epsilonkl = np.zeros((nbeads, nbeads))

    # compute default interaction parameters for beads
    for k in range(nbeads):
        for l in range(nbeads):
            sigmakl[k, l] = (beadlibrary[beads[k]]["sigma"] + beadlibrary[beads[l]]["sigma"]) / 2.0
            l_rkl[k, l] = 3 + np.sqrt((beadlibrary[beads[k]]["l_r"] - 3.0) * (beadlibrary[beads[l]]["l_r"] - 3.0))
            l_akl[k, l] = 3 + np.sqrt((beadlibrary[beads[k]]["l_a"] - 3.0) * (beadlibrary[beads[l]]["l_a"] - 3.0))
            epsilonkl[k, l] = np.sqrt(beadlibrary[beads[k]]["epsilon"] * beadlibrary[beads[l]]["epsilon"]) * \
                              np.sqrt((beadlibrary[beads[k]]["sigma"] ** 3) * (beadlibrary[beads[l]]["sigma"] ** 3)) / (
                                          sigmakl[k, l] ** 3)
    # testing if crosslibrary is empty ie not specified
    if crosslibrary:
        # find any cross terms in the cross term library
        crosslist = []

        for (i, beadname) in enumerate(beads):
            if beadname in list(crosslibrary.keys()):
                for (j, beadname2) in enumerate(beads):
                    if beadname2 in list(crosslibrary[beadname].keys()):
                        crosslist.append([i, j])

        for i in range(np.size(crosslist, axis=0)):
            try:
                a = crosslist[i][0]
                b = crosslist[i][1]
                epsilonkl[a, b] = crosslibrary[beads[a]][beads[b]]["epsilon"]
                epsilonkl[b, a] = epsilonkl[a, b]
            except KeyError:
                pass
            try:
                l_rkl[a, b] = crosslibrary[beads[a]][beads[b]]["l_r"]
                l_rkl[b, a] = l_rkl[a, b]
            except KeyError:
                pass

    Ckl = C(l_rkl, l_akl)

    return epsilonkl, sigmakl, l_akl, l_rkl, Ckl


def calc_hard_sphere_matricies(beads, beadlibrary, sigmakl, T):
    """
    Computes matrix of interaction parameters dkk, dkl, and x0kl
    This does not include function specific or assocciation terms

    """
    nbeads = len(beads)
    dkk = np.zeros(np.size(beads))
    for i in range(np.size(beads)):
        dkk[i] = calc_dkk(beadlibrary[beads[i]]["epsilon"], beadlibrary[beads[i]]["sigma"], T,
                          beadlibrary[beads[i]]["l_r"], beadlibrary[beads[i]]["l_a"])
    dkl = np.zeros((nbeads, nbeads))
    for k in range(nbeads):
        for l in range(nbeads):
            dkl[k, l] = (dkk[k] + dkk[l]) / 2.0

    x0kl = sigmakl / dkl

    return dkk, dkl, x0kl


def calc_assoc_matrices(beads, beadlibrary, sitenames=["H", "e1", "e2"], crosslibrary={}):
    """
    Return:
    epsilonHB: interaction energy between each bead and associtation site
    Kklab: bonding volume between each association site
    nk: for each bead the number of each type of site
    
    Inputs
    beads: list of beads
    beadlibrary: dictionary of bead parameters
    sitenames: list of sitenames (default specified)
    crosslibrary: dictionary of cross parameters
    Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk
     (number of sites ))
    """
    # initialize variables
    nbeads = len(beads)
    sitemax = np.size(sitenames)
    epsilonHB = np.zeros((nbeads, nbeads, sitemax, sitemax))
    Kklab = np.zeros_like(epsilonHB)
    nk = np.zeros((nbeads, sitemax))

    for i in range(nbeads):
        for j in range(np.size(sitenames)):
            try:
                nk[i, j] = beadlibrary[beads[i]]["Nk" + sitenames[j]]
            except KeyError:
                pass

    if crosslibrary:
        # find any cross terms in the cross term library
        crosslist = []
        for (i, beadname) in enumerate(beads):
            if beadname in list(crosslibrary.keys()):
                for (j, beadname2) in enumerate(beads):
                    if beadname2 in list(crosslibrary[beadname].keys()):
                        crosslist.append([i, j])

        for i in range(np.size(crosslist, axis=0)):
            for a in range(np.size(sitenames)):
                for b in range(np.size(sitenames)):
                    try:
                        epsilonHB[crosslist[i][0], crosslist[i][1], a, b] = \
                        crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]][
                            "epsilon" + sitenames[a] + sitenames[b]]
                        # epsilonHB[crosslist[i][0],crosslist[i][1],b,a]=epsilonHB[crosslist[i][0],crosslist[i][1],a,b]
                        # epsilonHB[crosslist[i][1],crosslist[i][0],a,b]=epsilonHB[crosslist[i][0],crosslist[i][1],a,b]
                        epsilonHB[crosslist[i][1], crosslist[i][0], b, a] = epsilonHB[crosslist[i][0], crosslist[i][1],
                                                                                      a, b]
                    except KeyError:
                        pass

                    try:
                        Kklab[crosslist[i][0], crosslist[i][1], a, b] = \
                        crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]]["K" + sitenames[a] + sitenames[b]]
                        # Kklab[crosslist[i][0],crosslist[i][1],b,a]=Kklab[crosslist[i][0],crosslist[i][1],a,b]
                        # Kklab[crosslist[i][1],crosslist[i][0],a,b]=Kklab[crosslist[i][0],crosslist[i][1],a,b]
                        Kklab[crosslist[i][1], crosslist[i][0], b, a] = Kklab[crosslist[i][0], crosslist[i][1], a, b]
                    except KeyError:
                        pass

    for i in range(nbeads):
        for a in range(np.size(sitenames)):
            for b in range(np.size(sitenames)):
                try:
                    epsilonHB[i, i, a, b] = beadlibrary[beads[i]]["epsilon" + sitenames[a] + sitenames[b]]
                    epsilonHB[i, i, b, a] = epsilonHB[i, i, a, b]
                    Kklab[i, i, a, b] = beadlibrary[beads[i]]["K" + sitenames[a] + sitenames[b]]
                    Kklab[i, i, b, a] = Kklab[i, i, a, b]

                except KeyError:
                    pass
    return epsilonHB, Kklab, nk


def C(l_r, l_a):
    """ Returns C, the Mie potential prefactor, defined in eq. 2
        Inputs:
            l_r: repulsive exponent of the Mie potential
            l_a: attractive exponent of the Mie potential
    """

    return (l_r / (l_r - l_a)) * (l_r / l_a)**(l_a / (l_r - l_a))


def calc_da1iidrho(rho, Amonopre, dii, l_aii, l_rii, x0ii, epsilonii, etax):
    step = np.sqrt(np.finfo(float).eps) * rho
    if step < np.finfo(float).eps:
        print('Warning density step is very small')

    fdcoef = np.array([1.0, -8.0, 8.0, -1.0])
    rholist = np.array([rho - (2.0 * step), rho - step, rho + step, rho + (2.0 * step)])
    # print 'rholist',rholist

    a1ii = np.zeros((np.size(rholist), np.size(l_aii)))

    for i, rhoi in enumerate(rholist):
        a1ii[i] = calc_a1ii(rhoi, Amonopre, dii, l_aii, l_rii, x0ii, epsilonii, etax)

        # print a1ii.T
    return np.sum(a1ii.T * fdcoef, axis=1) / (12.0 * step)


def calc_a1ii(rho, Amonopre, dii, l_aii, l_rii, x0ii, epsilonii, etax):
    """ Returns a1ii
    """

    Cii = C(l_rii, l_aii)

    Bii_r = calc_Bkl(rho, l_rii, Amonopre, dii, epsilonii, x0ii, etax)
    Bii_a = calc_Bkl(rho, l_aii, Amonopre, dii, epsilonii, x0ii, etax)
    a1s_r = calc_a1s(rho, Amonopre, l_rii, etax, epsilonii, dii)
    a1s_a = calc_a1s(rho, Amonopre, l_aii, etax, epsilonii, dii)

    return (Cii * (((x0ii**l_aii) * (a1s_a + Bii_a)) - ((x0ii**l_rii) * (a1s_r + Bii_r))))[0]


def calc_da2iidrho(rho, Amonopre, KHS, dii, chiii, l_aii, l_rii, x0ii, epsilonii, etax):
    Cii = C(l_rii, l_aii)
    step = np.sqrt(np.finfo(float).eps) * rho
    if step < np.finfo(float).eps:
        print('Warning density step is very small')
    fdcoef = np.array([1.0, -8.0, 8.0, -1.0])
    rholist = np.array([rho - (2.0 * step), rho - step, rho + step, rho + (2 * step)])
    a2ii = np.zeros((np.size(rholist), np.size(l_aii)))

    tau = np.zeros(np.size(l_aii))
    tau = (KHS / 2.0) * epsilonii * (Cii**2)

    for i, rhoi in enumerate(rholist):
        Bii_2r = calc_Bkl(rhoi, 2.0 * l_rii, Amonopre, dii, epsilonii, x0ii, etax)
        Bii_2a = calc_Bkl(rhoi, 2.0 * l_aii, Amonopre, dii, epsilonii, x0ii, etax)
        Bii_ar = calc_Bkl(rhoi, l_aii + l_rii, Amonopre, dii, epsilonii, x0ii, etax)
        a1s_2r = calc_a1s(rhoi, Amonopre, 2.0 * l_rii, etax, epsilonii, dii)
        a1s_2a = calc_a1s(rhoi, Amonopre, 2.0 * l_rii, etax, epsilonii, dii)
        a1s_ar = calc_a1s(rhoi, Amonopre, l_aii + l_rii, etax, epsilonii, dii)
        # really a2ii/(1+chiii)
        a2ii[i] = tau * (((x0ii**(2.0 * l_aii)) * (a1s_2a + Bii_2a)) - ((2.0 * (x0ii**(l_rii + l_aii))) *
                                                                        (a1s_ar + Bii_ar)) + ((x0ii**(2.0 * l_rii)) *
                                                                                              (a1s_2r + Bii_2r)))
    # print 'a2ii***********',a2ii

    return np.sum(a2ii.T * fdcoef, axis=1) / (12.0 * step)


def calc_Amono(rho, xi, nui, beads, beadlibrary, dkk, Amonopre, T, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl):
    """ Returns AHS, A1, A2, and A3: (number of densities) Amono components Note these quantites are normalized by NkbT 
                eta is realy zeta
                etax: (number of densities) packing fraciton based on hard sphere eq. 22
                etaxstar: (number of densities) packing fraction based on sigma
                KHS: (nubmer of densities) isothermal compressability of system with packing fraction etax
                xskl: (nbead,nbead) matrix of xs,k (eq. 15)
        Inputs:
            rho: (number of densities) number density of system N/V in m^-3
            xi: (number of components) numpy array of mol fractions sum(xi) should equal 1.0
            nui: (number of components,number of bead types) numpy array, list of bead quantites in each component 
                 defined for eq. 11. Note that indicies are flipped from definition in reference.
            beads: list of strings of unique bead types used in any of the components
            beadlibrary: dictionary of bead parameters, where items in beads are the keys for the dictionary
            dkk: (nbead) numpy array of hard sphere reference diameters for group k.
            Amonopre: segment number density prefactor, rhos=rho*Amonopre in eq. 13, computed in calc_Amonopre
            T: Temperature
            epsilonkl: (nbead,nbead) depth of potential energy well for each bead-bead interaction in K
            sigmakl: (nbead,nbead) Mie potential segment diameter in m
            dkl: (nbead,nbead) numpy array of hard sphere reference diameters for k,l group pairs.
            l_akl: (nbead,nbead) Mie potential atractive exponent for k,l pairs
            l_rkl: (nbead,nbead) Mie potential repulsive expoenent for k,l pairs
            Ckl: (nbead,nbead) numpy array of mie potential prefactors for k,l group pairs.
            x0kl: (nbead,nbead) sigmakl/dkl
        Uses numpy
         
            
    """

    # initialize variables
    nbeads = len(beads)  # nbeads is the number of unique groups used by any compnent
    rhos = rho * Amonopre

    ##### compute AHS (eq. 16) #####

    # initialize variables for AHS
    eta = np.zeros((np.size(rho), 4))
    xsk = np.zeros(nbeads, float)

    # compute  xsk, eq. 15
    for k in range(nbeads):
        xsk[k] = np.sum(xi * nui[:, k]) * beadlibrary[beads[k]]["Vks"] * beadlibrary[beads[k]]["Sk"]
    xsk /= Amonopre

    # compute eta, eq. 14
    for m in range(4):
        eta[:, m] = rhos * (np.sum(xsk * (dkk**m)) * (np.pi / 6.0))

    if rho.any() == 0.0:
        print(rho)
    # compute AHS, eq. 16
    AHS = (6.0 / (np.pi * rho)) * (np.log(1.0 - eta[:, 3]) * (((eta[:, 2]**3) / (eta[:, 3]**2)) - eta[:, 0]) +
                                   (3.0 * eta[:, 1] * eta[:, 2] /
                                    (1 - eta[:, 3])) + ((eta[:, 2]**3) / (eta[:, 3] * ((1.0 - eta[:, 3])**2))))
    # print xi
    # print eta
    # print rho
    # print rhos
    # print dkk
    # print xsk
    # print xsk*(dkk**3)
    # print np.sum(xsk*(dkk**3))*(np.pi/6.0)*rhos
    # print 'end section ahs'

    ##### compute a1kl, eq. 19 #####

    xskl = np.zeros((nbeads, nbeads))

    # calculate  xskl matrix
    for k in range(nbeads):
        for l in range(nbeads):
            xskl[k, l] = xsk[k] * xsk[l]

    # calc etax eq. 22
    etax = rhos * ((np.pi / 6.0) * np.sum(xskl * (dkl**3)))

    # compute components of eq. 19

    # compute Bkl(rhos,lambdakl_a)
    Bakl = calc_Bkl(rho, l_akl, Amonopre, dkl, epsilonkl, x0kl, etax)
    # compute Bkl(rhos,lambdakl_r)
    Brkl = calc_Bkl(rho, l_rkl, Amonopre, dkl, epsilonkl, x0kl, etax)

    # compute a1,kl_s(rhos,lambdakl_a)
    a1s_la = calc_a1s(rho, Amonopre, l_akl, etax, epsilonkl, dkl)
    # compute a1,kl_s(rhos,lambdakl_r)
    a1s_lr = calc_a1s(rho, Amonopre, l_rkl, etax, epsilonkl, dkl)

    # compute a1kl, eq. 19
    a1kl = Ckl * (((x0kl**l_akl) * (a1s_la + Bakl)) - ((x0kl**l_rkl) * (a1s_lr + Brkl)))

    ##### compute a2kl, eq. 30 #####

    # initialize variables for a2kl
    # a2kl = np.zeros((nbeads,nbeads))
    # alphakl = np.zeros((nbeads,nbeads))

    # compute KHS(rho), eq. 31
    KHS = ((1.0 - etax)**4) / (1.0 + (4.0 * etax) + (4.0 * (etax**2)) - (4.0 * (etax**3)) + (etax**4))

    # compute alphakl eq. 33
    alphakl = Ckl * ((1.0 / (l_akl - 3.0)) - (1.0 / (l_rkl - 3.0)))

    # compute etaxstar eq. 35
    etaxstar = rhos * ((np.pi / 6.0) * np.sum(xskl * (sigmakl**3)))

    # compute f1, f2, and f3 for eq. 32
    fmlist123 = calc_fm(alphakl, np.array([1, 2, 3]))

    chikl = np.einsum("i,jk", etaxstar, fmlist123[0]) + np.einsum("i,jk", etaxstar**5, fmlist123[1]) + np.einsum(
        "i,jk", etaxstar**8, fmlist123[2])

    a1s_2la = calc_a1s(rho, Amonopre, 2.0 * l_akl, etax, epsilonkl, dkl)
    a1s_2lr = calc_a1s(rho, Amonopre, 2.0 * l_rkl, etax, epsilonkl, dkl)
    a1s_lalr = calc_a1s(rho, Amonopre, l_akl + l_rkl, etax, epsilonkl, dkl)
    B_2la = calc_Bkl(rho, 2.0 * l_akl, Amonopre, dkl, epsilonkl, x0kl, etax)
    B_2lr = calc_Bkl(rho, 2.0 * l_rkl, Amonopre, dkl, epsilonkl, x0kl, etax)
    B_lalr = calc_Bkl(rho, l_akl + l_rkl, Amonopre, dkl, epsilonkl, x0kl, etax)

    a2kl = (x0kl**(2.0 * l_akl)) * (a1s_2la + B_2la) - ((2.0 * x0kl**(l_akl + l_rkl)) *
                                                        (a1s_lalr + B_lalr)) + ((x0kl**(2.0 * l_rkl)) *
                                                                                (a1s_2lr + B_2lr))
    a2kl *= (1.0 + chikl) * epsilonkl * (Ckl**2)  # *(KHS/2.0)
    a2kl = np.einsum("i,ijk->ijk", KHS / 2.0, a2kl)

    ##### compute a3kl #####
    a3kl = np.zeros((nbeads, nbeads))
    fmlist456 = calc_fm(alphakl, np.array([4, 5, 6]))

    a3kl = np.einsum("i,jk", etaxstar, -(epsilonkl**3) * fmlist456[0]) * np.exp(
        np.einsum("i,jk", etaxstar, fmlist456[1]) + np.einsum("i,jk", etaxstar**2, fmlist456[2]))
    # a3kl=-(epsilonkl**3)*fmlist456[0]*etaxstar*np.exp((fmlist456[1]*etaxstar)+(fmlist456[2]*(etaxstar**2)))

    # compute a1, a2, a3 from 18, 29, and 37 respectively
    a1 = np.einsum("ijk,jk->i", a1kl, xskl)
    a2 = np.einsum("ijk,jk->i", a2kl, xskl)
    a3 = np.einsum("ijk,jk->i", a3kl, xskl)

    # compute A1, A2, and A3
    # note that a1, a2, and a3 have units of K, K^2, and K^3 respectively
    A1 = (Amonopre / T) * a1
    A2 = (Amonopre / (T**2)) * a2
    A3 = (Amonopre / (T**3)) * a3

    # print 'ACOMP',AHS, A1, A2, A3

    return AHS, A1, A2, A3, etax, etaxstar, KHS, xskl


def calc_A_assoc(rho, xi, T, nui, beads, beadlibrary, xskl, sigmakl, sigmaii3, epsilonii, epsilonHB, Kklab, nk):
    """
    xskl
    sigma
    Compute Association 
    epsilonHB
    rho
    

    """
    kT = T * constants.kb
    nbeads = len(beads)
    ncomp = np.size(xi)
    nsitesmax = np.size(nk, axis=1)
    # print nsitesmax
    Fklab = np.zeros((nbeads, nbeads, nsitesmax, nsitesmax))
    epsilonij = np.zeros((ncomp, ncomp))
    Iij = np.zeros((np.size(rho), ncomp, ncomp))
    delta = np.zeros((np.size(rho), ncomp, ncomp, nbeads, nbeads, nsitesmax, nsitesmax))
    Xika = np.zeros((np.size(rho), ncomp, nbeads, nsitesmax))
    Aassoc = np.zeros_like(rho)

    # compute F_klab
    Fklab = np.exp(epsilonHB * constants.kb / kT) - 1.0

    # compute epsilonij
    sigmaii = sigmaii3**(1.0 / 3.0)
    for i in range(ncomp):
        for j in range(i, ncomp):
            epsilonij[i, j] = np.sqrt(sigmaii3[i] * sigmaii3[j]) * np.sqrt(epsilonii[i] * epsilonii[j]) / ((
                (sigmaii[i] + sigmaii[j]) / 2.0)**3)
            epsilonij[j, i] = epsilonij[i, j]

    # print 'epsilonij',epsilonij
    # compute sigmax3
    Amonopre = calc_Amonopre(xi, nui, beads, beadlibrary)
    sigmax3 = Amonopre * np.sum(xskl * (sigmakl**3))

    # compute Iijklab
    for p in range(11):
        for q in range(11 - i):
            # temp=np.einsum("i,jk->ijk",constants.cij[p,q]*((sigmax3*rho)**p),((kT/epsilonij)**q))
            # temp=constants.cij[p,q]*((sigmax3*rho[0])**p)*((kT/epsilonij)**q)
            # print p,q
            Iij += np.einsum("i,jk->ijk", constants.cij[p, q] * ((sigmax3 * rho)**p), ((kT / epsilonij)**q))
    # print Iij
    # compute deltaijklab
    for i in range(ncomp):
        for j in range(ncomp):
            for k in range(nbeads):
                for l in range(nbeads):
                    for a in range(nsitesmax):
                        for b in range(nsitesmax):
                            # print Fklab[k,l,a,b],Kklab[k,l,a,b],Iij[i,j]
                            if nui[i, k] and nui[j, l] > 0:
                                delta[:, i, j, k, l, a, b] = Fklab[k, l, a, b] * Kklab[k, l, a, b] * Iij[:, i, j]

    Xika0 = np.zeros((ncomp, nbeads, nsitesmax))
    Xika0[:, :, :] = 1.0

    Xika = solv_assoc.min_xika(rho, Xika0, xi, nui, nk, delta, 500, 1.0E-12)
    if np.any(Xika < 0.0):
        Xika0[:, :, :] = 0.5
        sol = spo.root(calc_Xika_wrap, Xika0, args=(xi, rho[0], nui, nk, delta[0]), method='broyden1')
        Xika0 = sol.x
        Xika = solv_assoc.min_xika(rho, Xika0, xi, nui, nk, delta, 500, 1.0E-12)
        # Xika=solv_assoc.min_xika(rho,Xika0,xi,nui,nk,delta,500,1.0E-12)
        print('Xika out of bounds')

    for i in range(ncomp):
        for k in range(nbeads):
            for a in range(nsitesmax):
                if nk[k, a] != 0.0:
                    Aassoc += xi[i] * nui[i, k] * nk[k, a] * (np.log(Xika[:, i, k, a]) +
                                                              ((1.0 - Xika[:, i, k, a]) / 2.0))

    return Aassoc


def calc_Xika_wrap(Xika0, xi, rho, nui, nk, delta):
    # val=solv_assoc.calc_xika(Xika0,xi,rho,nui,nk,delta)
    # print np.size(rho)
    # print np.size(Xika0)
    obj_func, Xika = solv_assoc.calc_xika(Xika0, xi, rho, nui, nk, delta)
    return obj_func


def calc_A(rho, xi, T, massi, nui, beads, beadlibrary, dkk, Amonopre, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl,
           epsilonHB, Kklab, nk):
    # t1=time.time()
    Aideal = calc_Aideal(xi, rho, massi, T)
    AHS, A1, A2, A3, etax, etaxstar, KHS, xskl = calc_Amono(rho, xi, nui, beads, beadlibrary, dkk, Amonopre, T,
                                                            epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl)
    Achain, sigmaii3, epsilonii = Ac.calc_Achain(rho, Amonopre, xi, nui, etax, sigmakl, epsilonkl, dkl, xskl, l_rkl,
                                                 l_akl, beads, beadlibrary, T, etaxstar, KHS)
    # t2=time.time()
    # print t2-t1
    if np.sum(nk) > 0.0:
        Aassoc = calc_A_assoc(rho, xi, T, nui, beads, beadlibrary, xskl, sigmakl, sigmaii3, epsilonii, epsilonHB,
                              Kklab, nk)
        A = Aideal + AHS + A1 + A2 + A3 + Achain + Aassoc
    else:
        A = Aideal + AHS + A1 + A2 + A3 + Achain

    # NoteHere
    # nrho=np.size(rho)/2
    # tmp = []
    # for a in [Aideal,AHS,A1,A2,A3,Achain]:
    #    tmp.append(a[:nrho]-a[nrho:])
    # tmp = np.array(tmp).T.tolist()
    # with open("test_Helmholtz.csv","w") as f:
    #    f.write("Aideal,AHS,A1,A2,A3,Achain\n")
    #    for line in tmp:
    #        line = [str(x) for x in line]
    #        f.write(", ".join(line)+"\n")

    # with open("Output_Achain.csv","w") as f:
    #    f.write("tmp_g1, tmp_g2\n")
    #    for line in tmp_A.T:
    #        f.write("%s,%s\n" % (str(line[0]),str(line[1])))

    return A


def calc_Ares(rho, xi, T, massi, nui, beads, beadlibrary, dkk, Amonopre, epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl,
              x0kl, epsilonHB, Kklab, nk):
    # t1=time.time()
    AHS, A1, A2, A3, etax, etaxstar, KHS, xskl = calc_Amono(rho, xi, nui, beads, beadlibrary, dkk, Amonopre, T,
                                                            epsilonkl, sigmakl, dkl, l_akl, l_rkl, Ckl, x0kl)
    Achain, sigmaii3, epsilonii = Ac.calc_Achain(rho, Amonopre, xi, nui, etax, sigmakl, epsilonkl, dkl, xskl, l_rkl,
                                                 l_akl, beads, beadlibrary, T, etaxstar, KHS)
    # t2=time.time()
    # print t2-t1
    if np.sum(nk) > 0.0:
        Aassoc = calc_A_assoc(rho, xi, T, nui, beads, beadlibrary, xskl, sigmakl, sigmaii3, epsilonii, epsilonHB,
                              Kklab, nk)
        Ares = AHS + A1 + A2 + A3 + Achain + Aassoc
    else:
        Ares = AHS + A1 + A2 + A3 + Achain
    # print Aideal,AHS,A1,A2,A3,Achain
    # print Ares
    return Ares


def Calc_dadT(rho,
              T,
              xi,
              massi,
              nui,
              beads,
              beadlibrary,
              dkk,
              epsilonkl,
              sigmakl,
              dkl,
              l_akl,
              l_rkl,
              Ckl,
              x0kl,
              epsilonHB=[],
              Kklab=[],
              nk=[],
              sitenames=["H", "e1", "e2"],
              crosslibrary={}):
    """
        Given rho N/m3 and T compute denstiy given SAFT parameters
        """
    step = np.sqrt(np.finfo(float).eps) * T * 1000.0
    nrho = np.size(rho)

    #computer rho+step and rho-step for better a bit better performance
    Amonopre = calc_Amonopre(xi, nui, beads, beadlibrary)
    Ap = calc_A(np.array([rho]), xi, T + step, massi, nui, beads, beadlibrary, dkk, Amonopre, epsilonkl, sigmakl, dkl,
                l_akl, l_rkl, Ckl, x0kl, epsilonHB, Kklab, nk)
    Am = calc_A(np.array([rho]), xi, T - step, massi, nui, beads, beadlibrary, dkk, Amonopre, epsilonkl, sigmakl, dkl,
                l_akl, l_rkl, Ckl, x0kl, epsilonHB, Kklab, nk)

    return (Ap - Am) / (2.0 * step)
