# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging
import scipy.optimize as spo
import os

import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state import constants

logger = logging.getLogger(__name__)

# Check for Numba
if 'NUMBA_DISABLE_JIT' in os.environ:
    disable_jit = os.environ['NUMBA_DISABLE_JIT']
else:
    from ... import jit_stat
    disable_jit = jit_stat.disable_jit

if disable_jit:
    try:
        from ..compiled_modules import solv_assoc
    except ImportError:
        try:
            from ..compiled_modules.jit_exts import calc_Xika
            logger.info("Fortan module is not available, using Numba")
        except ImportError:
            from ..compiled_modules.c_exts import calc_Xika
            logger.info("Fortan module and Numba are not available, using Cython")
    except:
        raise ImportError("Fortran, Numba, and Cython modules are not available.")
else:
    from ..compiled_modules.jit_exts import calc_Xika

# Check for cython
from ... import cython_stat
disable_cython = cython_stat.disable_cython
if not disable_cython:
    if not disable_jit:
        logger.warning("Flag for Numba and Cython were given. Using Numba")
    else:
        from ..compiled_modules.c_exts import calc_Xika

class Aassoc():

    r"""
    Calculates the association contribution of the Helmholtz energy, :math:`A^{assoc.}`.
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):
        
        needed_attributes = ['nui','beads','beadlibrary']
        
        if not hasattr(self, 'eos_dict'):
            self.eos_dict = {}

        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                self.eos_dict[key] = kwargs[key]
                    
        if 'crosslibrary' not in kwargs:
            self.eos_dict['crosslibrary'] = {}
        else:
            self.eos_dict['crosslibrary'] = kwargs['crosslibrary']

        if not hasattr(self,'sitenames'):
            if 'sitenames' not in kwargs:
                self.eos_dict['sitenames']=["H", "e1", "e2"]
            else:
                self.eos_dict['sitenames']=kwargs['sitenames']
        self.nsitesmax = len(self.eos_dict['sitenames'])

        if not hasattr(self, 'Vks'):
            self.eos_dict['Vks'] = tb.extract_property("Vks",self.eos_dict['beadlibrary'],self.eos_dict['beads'])
        if not hasattr(self, 'Sk'):
            self.eos_dict['Sk'] = tb.extract_property("Sk",self.eos_dict['beadlibrary'],self.eos_dict['beads'])

        # Initialize temperature attribute
        if not hasattr(self, 'xi'):
            self.xi = np.nan
                
        if not hasattr(self, 'nbeads'):
            self.ncomp, self.nbeads = np.shape(self.eos_dict['nui'])

        # Initiate cross interaction terms
        if not all([hasattr(self, key) for key in ['epsilonkl','sigmakl']]):
            self.eos_dict.update(tb.calc_interaction_matrices(self.eos_dict['beads'],self.eos_dict['beadlibrary'],crosslibrary=self.eos_dict['crosslibrary']))

        # Initiate average interaction terms
        if not all([hasattr(self, key) for key in ['sigmaii_avg', 'epsilonii_avg']]):
            self.eos_dict.update(tb.calc_component_averaged_properties(self.eos_dict['nui'], self.eos_dict['Vks'],self.eos_dict['Sk'],epsilonkl=self.eos_dict['epsilonkl'], sigmakl=self.eos_dict['sigmakl'], l_akl=self.eos_dict['l_akl'], l_rkl=self.eos_dict['l_rkl']))
                
        self.calc_assoc_matrices()
        self.check_assoc()
    
    def Aassoc(self, rho, T, xi):
        r"""
        Outputs :math:`A^{association}`.
    
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
    
        Returns
        -------
        Aassoc : numpy.ndarray
            Helmholtz energy of monomers for each density given.
        """
        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)

        kT = T * constants.kb
        
        # compute F_klab
        Fklab = np.exp(self.eos_dict['epsilonHB'] / T) - 1.0
        Iij = self.calc_Iij(rho, T, xi)

        if disable_jit:
            # Compute Xika: with Fortran   {BottleNeck}
            # compute deltaijklab
            delta = np.zeros((np.size(rho), self.ncomp, self.ncomp, self.nbeads, self.nbeads, self.nsitesmax, self.nsitesmax))
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    for k in range(self.nbeads):
                        for l in range(self.nbeads):
                            for a in range(self.nsitesmax):
                                for b in range(self.nsitesmax):
                                    # print(Fklab[k,l,a,b],Kklab[k,l,a,b],Iij[i,j])
                                    if self.eos_dict['nui'][i, k] and self.eos_dict['nui'][j, l] > 0:
                                        delta[:, i, j, k, l, a, b] = Fklab[k, l, a, b] * self.eos_dict['Kklab'][k, l, a, b] * Iij[:, i, j] * 1e-27
            
            Xika0 = np.zeros((self.ncomp, self.nbeads, self.nsitesmax))
            Xika0[:, :, :] = 1.0
            Xika = solv_assoc.min_xika(rho*constants.Nav, Xika0, xi, self.eos_dict['nui'], self.eos_dict['nk'], delta, 500, 1.0E-12) # {BottleNeck}
            if np.any(Xika < 0.0):
                Xika0[:, :, :] = 0.5
                sol = spo.root(calc_Xika_wrap, Xika0, args=(xi, rho[0], delta[0]), method='broyden1')
                Xika0 = sol.x
                Xika = solv_assoc.min_xika(rho*constants.Nav, Xika0, xi, self.eos_dict['nui'], self.eos_dict['nk'], delta, 500, 1.0E-12) # {BottleNeck}
                logger.warning('Xika out of bounds')
        
        else:
            # Compute Xika: with python with numba  {BottleNeck}
            indices = self.assoc_site_indices(xi=xi)
            Xika, err_array = calc_Xika(indices,rho, xi, self.eos_dict['nui'], self.eos_dict['nk'], Fklab, self.eos_dict['Kklab'], Iij)
        
        # Compute A_assoc
        Aassoc = np.zeros(np.size(rho))
        for i in range(self.ncomp):
            for k in range(self.nbeads):
                for a in range(self.nsitesmax):
                    if self.eos_dict['nk'][k, a] != 0.0:
                        tmp = (np.log(Xika[:, i, k, a]) + ((1.0 - Xika[:, i, k, a]) / 2.0))
                        Aassoc += xi[i] * self.eos_dict['nui'][i, k] * self.eos_dict['nk'][k, a] * tmp

        return Aassoc
            
    def calc_Xika_wrap(self, Xika0, xi, rho, delta):
        r"""
        Uses Fortran modules to calculate the fraction of molecules of component i that are not bonded at a site of type a on group k.
        
        Parameters
        ----------
        Xika0 :
            Guess in Xika matrix
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        rho : numpy.ndarray
            Number density of system [molecules/m^3]
        delta : numpy.ndarray
            The association strength between a site of type a on a group of type k of component i and a site of type b on a group of type l of component j. eq. 66
        
        Returns
        -------
        obj_func :
            Used in calculation of association term of Helmholtz energy
        """
        
        obj_func, Xika = solv_assoc.calc_xika(Xika0, xi, rho*constants.Nav, self.eos_dict['nui'], self.eos_dict['nk'], delta)
        return obj_func

    def calc_Iij(self,rho, T, xi):
        r"""
        Outputs :math:`A^{chain}`.
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        
        Returns
        -------
        Iij :
        
        """

        self._check_density(rho)
        self._check_composition_dependent_parameters(xi)
        kT = T * constants.kb

        if not hasattr(self, 'cij'):
            self._initiate_cij()
        
        # compute epsilonij
        epsilonij = np.zeros((self.ncomp, self.ncomp))
        for i in range(self.ncomp):
            for j in range(i, self.ncomp):
                epsilonij[i, j] = np.sqrt(self.eos_dict['sigmaii_avg'][i] * self.eos_dict['sigmaii_avg'][j])**3.0 * np.sqrt(self.eos_dict['epsilonii_avg'][i] * self.eos_dict['epsilonii_avg'][j]) / (((self.eos_dict['sigmaii_avg'][i] + self.eos_dict['sigmaii_avg'][j]) / 2.0)**3)
                epsilonij[j, i] = epsilonij[i, j]
        
        sigmax3 = np.sum(self.eos_dict['xskl'] * (self.eos_dict['sigmakl']**3 * constants.molecule_per_nm3))

        Iij = np.zeros((np.size(rho), self.ncomp, self.ncomp))
        for p in range(11):
            for q in range(11 - p):
            #Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3 * rho)**p), ((kT / epsilonij)**q))
                if p == 0:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * np.ones(len(rho)), ((kT / epsilonij)**q))
                elif p == 1:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3 * rho)), ((kT / epsilonij)**q))
                elif p == 2:
                    rho2 = rho**2
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho2)), ((kT / epsilonij)**q))
                elif p == 3:
                    rho3 = rho2*rho
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho3)), ((kT / epsilonij)**q))
                elif p == 4:
                    rho4 = rho2**2
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho4)), ((kT / epsilonij)**q))
                elif p == 5:
                    rho5 = rho2*rho3
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho5)), ((kT / epsilonij)**q))
                elif p == 6:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho*rho5)), ((kT / epsilonij)**q))
                elif p == 7:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho2*rho5)), ((kT / epsilonij)**q))
                elif p == 8:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho3*rho5)), ((kT / epsilonij)**q))
                elif p == 9:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho4*rho5)), ((kT / epsilonij)**q))
                elif p == 10:
                    Iij += np.einsum("i,jk->ijk", self.cij[p, q] * ((sigmax3**p * rho5*rho5)), ((kT / epsilonij)**q))
        return Iij

    def _initiate_cij(self):
        """
        
        
        Attributes
        ----------
        cij : numpy.ndarray
        
        """
    
        self.cij=np.array([[7.56425183020431e-02, -1.28667137050961e-01, 1.28350632316055e-01, -7.25321780970292e-02, 2.57782547511452e-02, -6.01170055221687e-03, 9.33363147191978e-04, -9.55607377143667e-05, 6.19576039900837e-06, -2.30466608213628e-07, 3.74605718435540e-09],\
                  [1.34228218276565e-01, -1.82682168504886e-01, 7.71662412959262e-02, -7.17458641164565e-04, -8.72427344283170e-03, 2.97971836051287e-03, -4.84863997651451e-04, 4.35262491516424e-05, -2.07789181640066e-06, 4.13749349344802e-08, 0.00000000000000e+00],\
                  [-5.65116428942893e-01, 1.00930692226792e+00, -6.60166945915607e-01, 2.14492212294301e-01, -3.88462990166792e-02, 4.06016982985030e-03, -2.39515566373142e-04, 7.25488368831468e-06, -8.58904640281928e-08, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [-3.87336382687019e-01, -2.11614570109503e-01, 4.50442894490509e-01, -1.76931752538907e-01, 3.17171522104923e-02, -2.91368915845693e-03, 1.30193710011706e-04, -2.14505500786531e-06, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [2.13713180911797e+00, -2.02798460133021e+00, 3.36709255682693e-01, 1.18106507393722e-03, -6.00058423301506e-03, 6.26343952584415e-04, -2.03636395699819e-05, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [-3.00527494795524e-01, 2.89920714512243e+00, -5.67134839686498e-01, 5.18085125423494e-02, -2.39326776760414e-03, 4.15107362643844e-05, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [-6.21028065719194e+00, -1.92883360342573e+00, 2.84109761066570e-01, -1.57606767372364e-02, 3.68599073256615e-04, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [1.16083532818029e+01, 7.42215544511197e-01, -8.23976531246117e-02, 1.86167650098254e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [-1.02632535542427e+01, -1.25035689035085e-01, 1.14299144831867e-02, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [4.65297446837297e+00, -1.92518067137033e-03, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],\
                  [-8.67296219639940e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00]])

    def assoc_site_indices(self, xi=None):
        r"""
        Make a list of sets of indices that allow quick identification of the relevant association sights.
        
        This is needed for solving Xika, the fraction of molecules of component i that are not bonded at a site of type a on group k.
        
        Parameters
        ----------
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
        for bead in self.eos_dict['nk']:
            bead_sites.append([i for i, site in enumerate(bead) if site != 0])

        # Indices of components will minimal mole fractions
        if xi is not None:
            zero_frac = np.where(np.array(xi)<1e-32)[0]
        else:
            zero_frac = np.array([])

        for i, comp in enumerate(self.eos_dict['nui']):
            if i not in zero_frac:
                for j, bead in enumerate(comp):
                    if (bead != 0 and bead_sites[j]):
                        for k in bead_sites[j]:
                            indices.append([i,j,k])

        indices = np.array([np.array(x) for x in indices])

        return indices

    def calc_assoc_matrices(self):
        r"""
        
        Generate matrices used for association site calculations.
        
        Compute epsilonHB (interaction energy for association term),Kklab (association interaction bonding volume,nk (number of sites )
        
        Attributes
        ----------
        epsilonHB : numpy.ndarray
            Interaction energy between each bead and association site.
        Kklab : numpy.ndarray
            Bonding volume between each association site
        nk : numpy.ndarray
            For each bead the number of each type of site
        """
    
        # initialize variables
        epsilonHB = np.zeros((self.nbeads, self.nbeads, self.nsitesmax, self.nsitesmax))
        Kklab = np.zeros((self.nbeads, self.nbeads, self.nsitesmax, self.nsitesmax))
        nk = np.zeros((self.nbeads, self.nsitesmax))
        
        for i, bead in enumerate(self.eos_dict['beads']):
            for j, sitename in enumerate(self.eos_dict['sitenames']):
                Nk_tmp = "Nk"+ sitename
                if Nk_tmp in self.eos_dict['beadlibrary'][bead]:
                    logger.debug("Bead {} has {} of the association site     {}".format(bead,self.eos_dict['beadlibrary'][bead][Nk_tmp],Nk_tmp))
                    nk[i, j] = self.eos_dict['beadlibrary'][bead][Nk_tmp]

        if 'crosslibrary' in self.eos_dict and len(self.eos_dict['crosslibrary']) >0:
            crosslibrary = self.eos_dict['crosslibrary']
            beads = self.eos_dict['beads']
        
            # find any cross terms in the cross term library
            crosslist = []
            for (i, beadname) in enumerate(beads):
                if beadname in crosslibrary:
                    for (j, beadname2) in enumerate(beads):
                        if beadname2 in crosslibrary[beadname]:
                            crosslist.append([i, j])
        
            for i in range(np.size(crosslist, axis=0)):
                for a in range(self.nsitesmax):
                    for b in range(self.nsitesmax):
                        if beads[crosslist[i][0]] in crosslibrary:
                            if beads[crosslist[i][1]] in crosslibrary[beads[crosslist[i][0]]]:
        
                                epsilon_tmp = "epsilon"+self.eos_dict['sitenames'][a]+self.eos_dict['sitenames'][b]
                                K_tmp = "K"+self.eos_dict['sitenames'][a]+self.eos_dict['sitenames'][b]
                                if epsilon_tmp in crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]]:
                                    if (nk[crosslist[i][0]][a] == 0 or nk[crosslist[i][1]][b] == 0):
                                        if 0 not in [nk[crosslist[i][0]][b],nk[crosslist[i][1]][a]]:
                                            logger.warning("Site names were listed in wrong order for parameter definitions in cross interaction library. Changing {}_{} - {}_{} interaction to {}_{} - {}_{}".format(beads[crosslist[i][0]],self.eos_dict['sitenames'][a],beads[crosslist[i]    [1]],self.eos_dict['sitenames'][b],beads[crosslist[i][0]],self.eos_dict['sitenames'][b],beads[crosslist[i]    [1]],self.eos_dict['sitenames'][a]))
                                            a, b = [b, a]
                                        elif nk[crosslist[i][0]][a] == 0:
                                            logger.warning("Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format(beads[crosslist[i][0]],self.eos_dict['sitenames'][a],beads[crosslist[i]    [1]],self.eos_dict['sitenames'][b],beads[crosslist[i][0]],self.eos_dict['sitenames'][a]))
                                        elif nk[crosslist[i][1]][b] == 0:
                                            logger.warning("Cross interaction library parameters suggest a {}_{} - {}_{}   interaction, but {} doesn't have site {}.".format(beads[crosslist[i][0]],self.eos_dict['sitenames'][a],beads[crosslist[i]    [1]],self.eos_dict['sitenames'][b],beads[crosslist[i][1]],self.eos_dict['sitenames'][b]))
        
                                    epsilonHB[crosslist[i][0], crosslist[i][1], a, b] = \
                                crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]][epsilon_tmp]
                                    epsilonHB[crosslist[i][1], crosslist[i][0], b, a] = epsilonHB[crosslist[i][0],     crosslist[i][1],a, b]
        
                                if K_tmp in crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]]:
                                    Kklab[crosslist[i][0], crosslist[i][1], a, b] = \
                                    crosslibrary[beads[crosslist[i][0]]][beads[crosslist[i][1]]][K_tmp]
                                    Kklab[crosslist[i][1], crosslist[i][0], b, a] = Kklab[crosslist[i][0], crosslist[i][1], a, b]

        for i,bead in enumerate(self.eos_dict['beads']):
            for a in range(self.nsitesmax):
                for b in range(self.nsitesmax):
                    tmp = ["epsilon"+self.eos_dict['sitenames'][a]+self.eos_dict['sitenames'][b], "K"+self.eos_dict['sitenames'][a]+self.eos_dict['sitenames'][b]]
                    if all(x in self.eos_dict['beadlibrary'][bead] for x in tmp):
                        epsilonHB[i, i, a, b] = self.eos_dict['beadlibrary'][bead]["epsilon" + self.eos_dict['sitenames'][a] + self.eos_dict['sitenames'][b]]
                        epsilonHB[i, i, b, a] = epsilonHB[i, i, a, b]
                        Kklab[i, i, a, b] = self.eos_dict['beadlibrary'][bead]["K" + self.eos_dict['sitenames'][a] + self.eos_dict['sitenames'][b]]
                        Kklab[i, i, b, a] = Kklab[i, i, a, b]

        self.eos_dict['epsilonHB'] = epsilonHB
        self.eos_dict['Kklab'] = Kklab
        self.eos_dict['nk'] = nk
            
    def check_assoc(self):
        r""" Check if any association sites are used in this system.
        """
        indices = self.assoc_site_indices()
        if indices.size != 0:
            tmp = 0
            for i,k,a in indices:
                for j,l,b in indices:
                    tmp += self.eos_dict['epsilonHB'][k, l, a, b]
        else:
            tmp = 0.
       
        if tmp == 0.0:
            self.flag_assoc = False
        else:
            self.flag_assoc = True

    def _check_density(self, rho):
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        """
        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
                raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

    def _check_composition_dependent_parameters(self, xi):
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        rho : numpy.ndarray
        Number density of system [mol/m^3]
        T : float
        Temperature of the system [K]
        xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
        
        Atributes
        ---------
        eos_dict : dict
        The following entries are updated: Cmol2seg, xskl
        """
        xi = np.array(xi)
        if not np.all(self.xi == xi):
            self.eos_dict['Cmol2seg'], self.eos_dict['xskl'] = tb.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beadlibrary'], self.eos_dict['beads'])

            self.xi = xi

    def __str__(self):

        string = "Beads: {}, Sitenames".format(self.eos_dict['beads'],eos_dict['sitenames'])
        return string


