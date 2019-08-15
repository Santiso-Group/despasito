"""
    despasito
    DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output
    
    EOS object for SAFT-Gamma-Mie
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import sys
import numpy as np

from . import constants
from . import gamma_mie_funcs as funcs
# Later this line will be in an abstract class file in this directroy, and all versions of SAFT will reference it
from despasito.equations_of_state.interface import EOStemplate

# ________________ Saft Family ______________
# NoteHere: Insert SAFT family abstract class in this directory to clean up


class saft_gamma_mie(EOStemplate):
    def __init__(self, kwargs):

        # Self interaction parameters
        xi = kwargs['xi']
        self.nui = kwargs['nui']
        self.beads = kwargs['beads']
        self.beadlibrary = kwargs['beadlibrary']

        massi = np.zeros_like(xi)
        for i in range(np.size(xi)):
            for k in range(np.size(self.beads)):
                massi[i] += self.nui[i, k] * self.beadlibrary[self.beads[k]]["mass"]
        self.massi = massi

        # Cross interaction parameters
        if 'crosslibrary' in kwargs:
            crosslibrary = kwargs['crosslibrary']
        else:
            crosslibrary = {}

        epsilonkl, sigmakl, l_akl, l_rkl, Ckl = funcs.calc_interaction_matrices(self.beads,
                                                                                self.beadlibrary,
                                                                                crosslibrary=crosslibrary)

        self.epsilonkl = epsilonkl
        self.sigmakl = sigmakl
        self.l_akl = l_akl
        self.l_rkl = l_rkl
        self.Ckl = Ckl

        # Association sites
        if 'sitenames' in kwargs:
            self.sitenames = kwargs['sitenames']
        else:
            self.sitenames = []

        epsilonHB, Kklab, nk = funcs.calc_assoc_matrices(self.beads,
                                                         self.beadlibrary,
                                                         sitenames=self.sitenames,
                                                         crosslibrary=crosslibrary)
        self.epsilonHB = epsilonHB
        self.Kklab = Kklab
        self.nk = nk
        self.T = np.nan

    def temp_dependent_variables(self, T):
        dkk, dkl, x0kl = funcs.calc_hard_sphere_matricies(self.beads, self.beadlibrary, self.sigmakl, T)
        self.T = T
        self.dkk = dkk
        self.dkl = dkl
        self.x0kl = x0kl

    def P(self, rho, T, xi):
        """
       Given rho N/m3 and T compute denstiy given SAFT parameters
       """

        if not self.dkl.any():
            raise Exception("Temperature dependent variables haven't been specified.")

        step = np.sqrt(np.finfo(float).eps) * rho * 10000.0
        # Decreasing step size by 2 orders of magnitude didn't reduce noise in P values
        nrho = np.size(rho)

        # computer rho+step and rho-step for better a bit better performance
        Amonopre = funcs.calc_Amonopre(xi, self.nui, self.beads, self.beadlibrary)
        A = funcs.calc_A(np.append(rho + step, rho - step), xi, T, self.massi, self.nui, self.beads, self.beadlibrary, self.dkk, Amonopre, self.epsilonkl, self.sigmakl, self.dkl, self.l_akl, self.l_rkl, self.Ckl,
                         self.x0kl, self.epsilonHB, self.Kklab, self.nk)
        return (A[:nrho]-A[nrho:])*((constants.kb*T)/(2.0*step))*(rho**2)

    def chemicalpotential(self, P, rho, xi, T):

        if not self.dkl.any():
            raise Exception("Temperature dependent variables haven't been specified.")

        daresdxi = np.zeros_like(xi)
        mui = np.zeros_like(xi)
        Amonopre = funcs.calc_Amonopre(xi, self.nui, self.beads, self.beadlibrary)
        nmol = 1.0
        dnmol = 1.0E-4

        # compute mui
        for i in range(np.size(mui)):
            dAres = np.zeros(2)
            ares = funcs.calc_Ares(rho * constants.Nav, xi, T, self.massi, self.nui, self.beads, self.beadlibrary,
                                   self.dkk, Amonopre, self.epsilonkl, self.sigmakl, self.dkl, self.l_akl, self.l_rkl,
                                   self.Ckl, self.x0kl, self.epsilonHB, self.Kklab, self.nk)
            for j, delta in enumerate((dnmol, -dnmol)):
                xi_temp = np.copy(xi)
                xi_temp[i] += delta
                Amonopre = funcs.calc_Amonopre(xi_temp, self.nui, self.beads, self.beadlibrary)
                # xi_temp/=(nmol+delta)
                dAres[j] = funcs.calc_Ares(rho * constants.Nav, xi_temp, T, self.massi, self.nui, self.beads,
                                           self.beadlibrary, self.dkk, Amonopre, self.epsilonkl, self.sigmakl,
                                           self.dkl, self.l_akl, self.l_rkl, self.Ckl, self.x0kl, self.epsilonHB,
                                           self.Kklab, self.nk)
            daresdxi[i] = (dAres[0] - dAres[1]) / (2.0 * dnmol)

        # compute Z
        Z = P / (rho * T * constants.Nav * constants.kb)
        xjdaresdxj = np.sum(xi * daresdxi)
        for i in range(np.size(mui)):
            mui[i] = ares + Z - 1.0 + daresdxi[i] - xjdaresdxj - np.log(Z)
        return mui

    def density_max(self, xi, maxpack=0.65):

        if not self.dkl.any():
            raise Exception("Temperature dependent variables haven't been specified.")

        Amonopre = funcs.calc_Amonopre(xi, self.nui, self.beads, self.beadlibrary)
        # initialize variables and arrays
        nbeads = len(self.beads)
        xsk = np.zeros(nbeads, float)
        xskl = np.zeros((nbeads, nbeads))

        # compute xsk
        for k in range(nbeads):
            xsk[k] = np.sum(xi * self.nui[:, k]) * self.beadlibrary[self.beads[k]]["Vks"] * \
                     self.beadlibrary[self.beads[k]]["Sk"]
        xsk /= Amonopre
        # calculate  xskl matrix
        for k in range(nbeads):
            for l in range(nbeads):
                xskl[k, l] = xsk[k] * xsk[l]

        # estimate the maximum density based on the hard spher packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        maxrho = maxpack * 6.0 / (Amonopre * np.pi * np.sum(xskl * (self.dkl**3))) / constants.Nav
        return maxrho

    def __str__(self):
        string = "Beads:" + str(self.beads) + "\n"
        if np.isnan(self.T):
            string += "Temperature dependent variables haven't been specified."
        else:
            string += "T:" + str(self.T)
        return string

