"""
This module contains our thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS. The thermo module contains a series of wrapper to handle the inputs and outputs of these functions.

.. todo:: 
    Add types of scipy solving methods and the types available
    
    Update if statement to generalize as a factory
    
"""

import numpy as np
import sys
from scipy import interpolate
import scipy.optimize as spo
from scipy.misc import derivative
from scipy.ndimage.filters import gaussian_filter1d
import copy
import time
import matplotlib.pyplot as plt
import logging

from . import fund_constants as const

######################################################################
#                                                                    #
#                              Calc CC Params                        #
#                                                                    #
######################################################################
def calc_CC_Pguess(xilist, Tlist, CriticalProp):
    r"""
    Computes the Mie parameters of a mixture from the mixed critical properties of the pure components. 
    From: Mejia, A., C. Herdes, E. Muller. Ind. Eng. Chem. Res. 2014, 53, 4131-4141
    
    Parameters
    ----------
    xilist : list[list[xi]]
        List of different Mole fractions of each component, sum(xi) should equal 1.0 for each set. Each set of components corresponds to a temperature in Tlist.
    Tlist : list[float]
        Temperature of the system corresponding to composition in xilist [K]
    CriticalProp : list[list]
        List of critical properties :math:`T_C`, :math:`P_C`, :math:`\omega`, :math:`\rho_{0.7}`, :math:`Z_C`, :math:`V_C`, and molecular weight, where each of these properties is a list of values for each bead.
    
    Returns
    -------
    Psatm : list[float]
        A list of guesses in pressure based on critical properties, of the same length as xilist and Tlist [Pa]
    """

    logger = logging.getLogger(__name__)

    Tc, Pc, omega, rho_7, Zc, Vc, M = CriticalProp

    ############## Calculate Mixed System Mie Parameters
    flag = 0
    if all(-0.847 > x > 0.2387 for x in omega):
        flag = 1
        logger.warning("Omega is outside of the range that these correlations are valid")

    a = [14.8359, 22.2019, 7220.9599, 23193.4750, -6207.4663, 1732.9600]
    b = [0.0, -6.9630, 468.7358, -983.6038, 914.3608, -1383.4441]
    c = [0.1284, 1.6772, 0.0, 0.0, 0.0, 0.0]
    d = [0.0, 0.4049, -0.1592, 0.0, 0.0, 0.0]
    j = [1.8966, -6.9808, 10.6330, -9.2041, 4.2503, 0.0]
    k = [0.0, -1.6205, -0.8019, 1.7086, -0.5333, 1.0536]

    Tcm, Pcm, sigma, epsilon, Psatm = [[] for x in range(5)]

    i = 0
    jj = 1

    if flag == 1:
        Psatm = np.nan
    elif flag == 0: 
        for kk, xi in enumerate(xilist):
            # Mixture alpha
            omegaij = xi[i] * omega[i] + xi[jj] * omega[jj]
            tmp1 = np.sum([a[ii] * omegaij**ii for ii in range(6)])
            tmp2 = np.sum([b[ii] * omegaij**ii for ii in range(6)])
            l_r = tmp1 / (1. + tmp2)
            C = (l_r / (l_r - 6.)) * (l_r / 6.)**(6. / (l_r - 6.))
            al_tmp = C * (1. / 3. - 1. / (l_r - 3.))
            # Mixture Critical Properties Stewart-Burkhardt-Voo
            K = xi[i] * Tc[i] / Pc[i]**.5 + xi[jj] * Tc[jj] / Pc[jj]**.5
            tmp1 = xi[i] * Tc[i] / Pc[i] + xi[jj] * Tc[jj] / Pc[jj]
            tmp2 = xi[i] * (Tc[i] / Pc[i])**.5 + xi[jj] * (Tc[jj] / Pc[jj])**.5
            J = tmp1 / 3. + 2. / 3. * tmp2**2.
            Tc_tmp = K**2. / J
            Pc_tmp = (K / J)**2.
            # Mixture Pressure Prausnitz-Gunn
            if (Tlist[kk] / Tc[i] > 1. or Tlist[kk] / Tc[jj] > 1.):
                R = 8.3144598  # [kg*m^2/(s^2*mol*K)] Gas constant
                tmp1 = Zc[i] + Zc[jj]
                tmp2 = xi[i] * M[i] * Vc[i] + xi[jj] * M[jj] * Vc[jj]
                Pc_tmp = R * Tc_tmp * tmp1 / tmp2
            # Mixture Molar Density, Plocker Knapp
            Mij = M[i] * xi[i] + M[jj] * xi[jj]
            rho_tmp = 8. / Mij / ((rho_7[i] * M[i])**(-1. / 3.) + (rho_7[jj] * M[jj])**(-1. / 3.))**3.
            Nav = 6.0221415e+23  # avogadros number
    
            tmp1 = np.sum([c[ii] * al_tmp**ii for ii in range(6)])
            tmp2 = np.sum([d[ii] * al_tmp**ii for ii in range(6)])
            Tc_star = tmp1 / (1. + tmp2)
            eps_tmp = Tc_tmp / Tc_star  # [K], multiply by kB to change to energy units
    
            tmp3 = np.sum([j[ii] * al_tmp**ii for ii in range(6)])
            tmp4 = np.sum([k[ii] * al_tmp**ii for ii in range(6)])
            rho_star = tmp3 / (1. + tmp4)
            sig_tmp = (rho_star / rho_tmp / Nav)**(1. / 3.)
    
            # Calculate Psat
            eos_dict['massi'] = np.array([Mij])
            eos_dict['nui'] = np.array([[1]])
            eos_dict['beads'] = ['bead']
            eos_dict['beadlibrary'] = {
                'bead': {
                    'l_r': l_r,
                    'epsilon': eps_tmp,
                    'Vks': 1.0,
                    'Sk': 1.0,
                    'l_a': 6,
                    'mass': Mij,
                    'sigma': sig_tmp
                }
            }
            eos = eos("saft.gamma_mie",**eos_dict)
    
            if (Tlist[kk] < Tc_tmp):
                Psat_tmp, rholsat_tmp, rhogsat_tmp = calc_Psat(Tlist[kk], np.array([1.0]), eos)
            else:
                Psat_tmp = np.nan
    
            if np.isnan(Psat_tmp):
                Psatm = np.nan
                break
    
            # Save values
            # Nothing is done with rholsat_tmp and rhogsat_tmp
            Tcm.append(Tc_tmp)
            Pcm.append(Pc_tmp)
            sigma.append(sig_tmp)
            epsilon.append(eps_tmp)
            Psatm.append(Psat_tmp)

    return Psatm

######################################################################
#                                                                    #
#                      Pressure-Density Curve                        #
#                                                                    #
######################################################################
def PvsRho(T, xi, eos, minrhofrac=(1.0 / 500000.0), rhoinc=5.0, vspacemax=1.0E-4, maxpack=0.65):

    r"""
    Computes the Mie parameters of a mixture from the mixed critical properties of the pure components. 
    From: Mejia, A., C. Herdes, E. Muller. Ind. Eng. Chem. Res. 2014, 53, 4131-4141
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    minrhofrac : float, Optional, default: (1.0/500000.0)
        Fraction of the maximum density used to calculate, and is equal to, the minimum density of the density array. The minimum density is the reciprocal of the maximum specific volume used to calculate the roots. Passed from inputs to through the dictionary rhodict.
    rhoinc : float, Optional, default: 5.0
        The increment between density values in the density array. Passed from inputs to through the dictionary rhodict.
    vspacemax : float, Optional, default: 1.0E-4
        Maximum increment between specific volume array values. After conversion from density to specific volume, the increment values are compared to this value. Passed from inputs to through the dictionary rhodict.
    maxpack : float, Optional, default: 0.65
        Maximum packing fraction. Passed from inputs to through the dictionary rhodict.

    Returns
    -------
    vlist : numpy.ndarray
        Specific volume array. Length depends on values in rhodict [:math:`m^3`/mol]
    Plist : numpy.ndarray
        Pressure associated with specific volume of system with given temperature and composition [Pa]
    """

    logger = logging.getLogger(__name__)
    if type(xi) == list:
        xi = np.array(xi)

    #estimate the maximum density based on the hard sphere packing fraction, part of EOS
    maxrho = eos.density_max(xi, T, maxpack=maxpack)
    #min rho is a fraction of max rho, such that minrho << rhogassat
    minrho = maxrho * minrhofrac
    #list of densities for P,rho and P,v
    rholist = np.arange(minrho, maxrho, rhoinc)
    #check rholist to see when the spacing
    vspace = (1.0 / rholist[:-1]) - (1.0 / rholist[1:])
    if np.amax(vspace) > vspacemax:
        vspaceswitch = np.where(vspace > vspacemax)[0][-1]
        rholist_2 = 1.0 / np.arange(1.0 / rholist[vspaceswitch + 1], 1.0 / minrho, vspacemax)[::-1]
        rholist = np.append(rholist_2, rholist[vspaceswitch + 2:])

    #compute Pressures (Plist) for rholsit
    Plist = eos.P(rholist, T, xi)

    #Flip Plist and rholist arrays
    Plist = Plist[:][::-1]
    rholist = rholist[:][::-1]
    vlist = 1.0 / rholist

    return vlist, Plist


######################################################################
#                                                                    #
#                      Pressure-Volume Spline                        #
#                                                                    #
######################################################################
def PvsV_spline(vlist, Plist):
    r"""
    Fit arrays of specific volume and pressure values to a cubic Univariate Spline.
    
    Parameters
    ----------
    vlist : numpy.ndarray
        Specific volume array. Length depends on values in rhodict [:math:`m^3`/mol]
    Plist : numpy.ndarray
        Pressure associated with specific volume of system with given temperature and composition [Pa]
    
    Returns
    -------
    Pvspline : obj
        Function object of pressure vs. specific volume
    roots : list
        List of specific volume roots. Subtract a system pressure from the output of Pvsrho to find density of vapor and/or liquid densities.
    extrema : list
        List of specific volume values corresponding to local minima and maxima.
    """

    logger = logging.getLogger(__name__)

    Psmoothed = gaussian_filter1d(Plist, sigma=.5)

    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed)
    roots = Pvspline.roots().tolist()
    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed, k=4)
    extrema = Pvspline.derivative().roots().tolist()
    if extrema: 
        if len(extrema) > 2: extrema = extrema[0:2]

    if len(roots) ==2:
        slope, root2 = np.polyfit(vlist[-4:], Plist[-4:], 1)
        roots = np.append(roots,[root2])

    #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

    return Pvspline, roots, extrema

######################################################################
#                                                                    #
#                      Pressure-Volume Spline                        #
#                                                                    #
######################################################################
def PvsV_plot(vlist, Plist, Pvspline, markers=[]):
    r"""
    Plot pressure vs. specific volume.
    
    Parameters
    ----------
    vlist : numpy.ndarray
        Specific volume array. Length depends on values in rhodict [:math:`m^3`/mol]
    Plist : numpy.ndarray
        Pressure associated with specific volume of system with given temperature and composition [Pa]
    Pvspline : obj
        Function object of pressure vs. specific volume
    markers : list, Optional, default: []
        List of plot markers used in plot
    """

    logger = logging.getLogger(__name__)

    plt.figure(1)
    plt.plot(vlist,Plist,label="Orig.")
    plt.plot(vlist,Pvspline(vlist),label="Smoothed")
    plt.plot([vlist[0], vlist[-1]],[0,0],"k")
    for k in range(len(markers)):
        plt.plot([markers[k], markers[k]],[min(Plist),max(Plist)],"k")
    plt.xlabel("Specific Volume [$m^3$/mol]"), plt.ylabel("Pressure [Pa]")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

######################################################################
#                                                                    #
#                              Calc Psat                             #
#                                                                    #
######################################################################
def calc_Psat(T, xi, eos, rhodict={}):
    r"""
    Computes the saturated pressure, gas and liquid densities for a single component system given Temperature and Mie parameters
    T: Saturated Temperature in Kelvin
    minrhofrac: Fraction of maximum hard sphere packing fraction for gas density
    rhoinc: spacing densities for rholist in mol/m^3. Smaller values will generate a more accurate curve at increasing computational cost
    Returns Saturated Pressure in Pa, liquid denisty, and gas density in mol/m^3
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    Psat : float
        Saturation pressure given system information [Pa]
    rhov : float
        Density of vapor at saturation pressure [mol/:math:`m^3`]
    rhol : float
        Density of liquid at saturation pressure [mol/:math:`m^3`]
    """

    logger = logging.getLogger(__name__)

    if np.count_nonzero(xi) != 1:
        if np.count_nonzero(xi>0.1) != 1:
            raise ValueError("Multiple components have compositions greater than 10%, check code for source")
        else:
            ind = np.where((xi>0.1)==True)[0]
            raise ValueError("Multiple components have compositions greater than 0. Do you mean to obtain the saturation pressure of {} with a mole fraction of {}?".format(eos._beads[ind],xi[ind]))

    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    if (not extrema or len(extrema)<2):
        logger.warning('Error: One of the components is above its critical point, add an exception to setPsat')
        Psat = np.nan
        roots = [1.0, 1.0, 1.0]

    else:
        ind_Pmin1 = np.argwhere(np.diff(Plist) > 0)[0][0]
        ind_Pmax1 = np.argmax(Plist[ind_Pmin1:]) + ind_Pmin1

        Pmaxsearch = Plist[ind_Pmax1]

        Pconverged = 10 # If the pressure is negative (under tension), we search from a value just above vacuum 
        Pminsearch = max(Pconverged, np.amin(Plist[ind_Pmin1:ind_Pmax1]))

        #print(Pminsearch,Pmaxsearch)
        #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

        #search Pressure that gives equal area in maxwell construction
        Psat = spo.minimize_scalar(eq_area,
                               args=(Plist, vlist),
                               bounds=(Pminsearch * 1.0001, Pmaxsearch * .9999),
                               method='bounded')

        #Using computed Psat find the roots in the maxwell construction to give liquid (first root) and vapor (last root) densities
        Psat = Psat.x
        Pvspline, roots, extrema = PvsV_spline(vlist, Plist-Psat)

        if len(roots) ==2:
            slope, root2 = np.polyfit(vlist[-4:], Plist[-4:]-Psat, 1)
            roots = np.append(roots,[root2])

    #Psat,rholsat,rhogsat
    return Psat, 1.0 / roots[0], 1.0 / roots[2]

######################################################################
#                                                                    #
#                              Eq Area                               #
#                                                                    #
######################################################################
def eq_area(shift, Pv, vlist):
    r"""
    Objective function used to calculate the saturation pressure.

    Note: If the curve hasn't decayed to 0 yet, estimate the remaining area as a triangle. This isn't super accurate but we are just using the saturation pressure to get started.
    
    Parameters
    ----------
    shift : float
        Guess in Psat value used to translate the pressure vs. specific volume curve [Pa]
    Pv : numpy.ndarray
        Pressure associated with specific volume of system with given temperature and composition [Pa]
    vlist : numpy.ndarray
        Specific volume array. Length depends on values in rhodict [:math:`m^3`/mol]

    Returns
    -------
    obj_value : float
        Output of objective function, the addition of the positive area between first two roots, and negative area between second and third roots, quantity squared.

    """

    logger = logging.getLogger(__name__)

    Pvspline, roots, extrema = PvsV_spline(vlist, Pv-shift)

    if len(roots) >=3:
        a = Pvspline.integral(roots[0], roots[1])
        b = Pvspline.integral(roots[1], roots[2])
    elif len(roots) == 2:
        a = Pvspline.integral(roots[0], roots[1])
        # If the curve hasn't decayed to 0 yet, estimate the remaining area as a triangle. This isn't super accurate but we are just using the saturation pressure to get started.
        slope, root2 = np.polyfit(vlist[-4:], Pv[-4:]-shift, 1)
        b = Pvspline.integral(roots[1], vlist[-1]) + (Pv[-1]-shift)*(root2-vlist[-1])/2
    else:
        logger.warning("Pressure curve without cubic properties has wrongly been accepted. Try decreasing minrhofrac")
        PvsV_plot(vlist, Pv-shift, Pvspline, markers=extrema)

    return (a + b)**2

######################################################################
#                                                                    #
#                              Calc Rho V Full                       #
#                                                                    #
######################################################################
def calc_rhov(P, T, xi, eos, rhodict={}):
    r"""
    Computes vapor density under system conditions.
    
    Parameters
    ----------
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    rhov : float
        Density of vapor at system pressure [mol/:math:`m^3`]
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    """

    logger = logging.getLogger(__name__)

    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Plist = Plist-P
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    logger.debug("    Find rhov: P {} Pa, roots {} m^3/mol".format(P,roots))

    l_roots = len(roots)
    if l_roots == 0:
        if Pvspline(1/vlist[-1]) < 0:
            if not len(extrema):
                flag = 2
                logger.info("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                flag = 1
                logger.info("    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(T,xi))
            rho_tmp = 1/vlist[0]
        elif min(Plist)+P > 0:
            rho_tmp = np.nan
            if not len(extrema):
                flag = 4
                logger.info("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                flag = 4
                logger.info("    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
        else:
            logger.warning("    Flag 3: The T and yi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(T,xi))
            flag = 3
            PvsV_plot(vlist, Plist, Pvspline, markers=extrema)
            rho_tmp = np.nan
    elif l_roots == 1:
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
        elif (Pvspline(roots[0])+P) > (Pvspline(max(extrema))+P):
            #logger.debug("Extrema: {}".format(extrema))
            #logger.debug("Roots: {}".format(roots))
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(T,xi))
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
    elif l_roots == 2:
        if (Pvspline(roots[0])+P) < 0.:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 1: This T and xi, {} {}, combination produces a liquid under tension at this pressure".format(T,xi))
        else:
            flag = 4
            rho_tmp = np.nan
            logger.debug("    Flag 4: There should be a third root! Assume ideal gas P: {}".format(P))
    else: # 3 roots
        rho_tmp = 1.0 / roots[2]
        flag = 0

    if flag in [0,2]: # vapor or critical fluid
        tmp = [rho_tmp*.99, rho_tmp*1.01]
        if (Pdiff(tmp[0],P, T, xi, eos)*Pdiff(tmp[1],P, T, xi, eos))<0:
            rho_tmp = spo.brentq(Pdiff, tmp[0], tmp[1], args=(P, T, xi, eos), rtol=0.0000001)
        else:
            if Plist[0] < 0:
                logger.warning("Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(tmp))
            else:
                rho_tmp = spo.root(Pdiff, rho_tmp, args=(P, T, xi, eos), method="hybr", tol=0.0000001)
                rho_tmp = rho_tmp.x[0]

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    return rho_tmp, flag


######################################################################
#                                                                    #
#                              Calc Rho L Full                       #
#                                                                    #
######################################################################
def calc_rhol(P, T, xi, eos, rhodict={}):
    r"""
    Computes liquid density under system conditions.
    
    Parameters
    ----------
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    rhol : float
        Density of liquid at system pressure [mol/:math:`m^3`]
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    logger = logging.getLogger(__name__)

    # Get roots and local minima and maxima 
    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Plist = Plist-P
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    logger.debug("    Find rhol: P {} Pa, roots {} m^3/mol".format(P,str(roots)))

    if extrema:
        if len(extrema) == 1:
            logger.warning("One extrema at {}, assume weird minima behavior. Check your parameters.".format(1/extrema[0])) 

    # Assess roots, what is the liquid density
    l_roots = len(roots)
    if l_roots == 0: # zero roots
        if Pvspline(1/vlist[-1]):
            if not len(extrema):
                flag = 2
                logger.info("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                flag = 1
                logger.info("    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(T,xi))
            rho_tmp = 1/vlist[1]
        elif min(Plist)+P > 0:
            rho_tmp = np.nan
            if not len(extrema):
                flag = 4
                logger.info("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                flag = 4
                logger.info("    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
        else:
            flag = 3
            logger.error("    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(str(T),str(xi)))
            rho_tmp = np.nan
            PvsV_plot(vlist, Plist+P, Pvspline, markers=extrema)
    elif l_roots == 2: # 2 roots
        if (Pvspline(roots[0])+P) < 0.:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 1: This T and xi, {} {}, combination produces a liquid under tension at this pressure".format(T,xi))
        else: # There should be three roots, but the values of specific volume don't go far enough to pick up the last one
            flag = 1
            rho_tmp = 1.0 / roots[0]
    elif l_roots == 1: # 1 root
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
        elif (Pvspline(roots[0])+P) > (Pvspline(max(extrema))+P):
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(T,xi))
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.info("    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
    else: # 3 roots
        rho_tmp = 1.0 / roots[0]
        flag = 1

    if flag in [1,2]: # liquid or critical fluid
        tmp = [rho_tmp*.99, rho_tmp*1.01]
        if (Pdiff(tmp[0],P, T, xi, eos)*Pdiff(tmp[1],P, T, xi, eos))<0:
            rho_tmp = spo.brentq(Pdiff, tmp[0], tmp[1], args=(P, T, xi, eos), rtol=0.0000001)
        else:
            if Plist[0] < 0:
                logger.warning("Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(tmp))
            else:
                rho_tmp = spo.root(Pdiff, rho_tmp, args=(P, T, xi, eos), method="hybr", tol=0.0000001)
                rho_tmp = rho_tmp.x[0]

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    return rho_tmp, flag

######################################################################
#                                                                    #
#                              Calc Pdiff                            #
#                                                                    #
######################################################################
def Pdiff(rho, Pset, T, xi, eos):
    """
    Calculate difference between set point pressure and computed pressure for a given density
    
    Parameters
    ----------
    rho : float
        Density of system [mol/:math:`m^3`]
    Pset : float
        Guess in pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    
    Returns
    -------
    Pdiff : float
        Difference in set pressure and predicted pressure given system conditions.
    """

    logger = logging.getLogger(__name__)

    Pguess = eos.P(rho, T, xi)

    return (Pguess - Pset)

######################################################################
#                                                                    #
#                          Calc phi vapor                            #
#                                                                    #
######################################################################
def calc_phiv(P, T, yi, eos, rhodict={}):
    r"""
    Computes vapor fugacity coefficient under system conditions.
    
    Parameters
    ----------
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    yi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    phiv : float
        Fugacity coefficient of vapor at system pressure
    rhov : float
        Density of vapor at system pressure [mol/:math:`m^3`]
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    logger = logging.getLogger(__name__)

    rhov, flagv = calc_rhov(P, T, yi, eos, rhodict)
    if flagv == 4:
        phiv = np.ones_like(yi)
        rhov = 0.
        logger.info("    rhov set to 0.")
    else:

        phiv = eos.fugacity_coefficient(P, np.array([rhov]), yi, T)

        #muiv = eos.chemicalpotential_old(P, np.array([rhov]), yi, T)
        #phiv = np.exp(muiv)

    print("Vapor Fugacity Coeff",phiv)

    return phiv, rhov, flagv

######################################################################
#                                                                    #
#                         Calc phi liquid                            #
#                                                                    #
######################################################################
def calc_phil(P, T, xi, eos, rhodict={}):
    r"""
    Computes liquid fugacity coefficient under system conditions.
    
    Parameters
    ----------
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    phil : float
        Fugacity coefficient of liquid at system pressure
    rhol : float
        Density of liquid at system pressure [mol/:math:`m^3`]
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true.
    """

    logger = logging.getLogger(__name__)

    rhol, flagl = calc_rhol(P, T, xi, eos, rhodict)
    if flagl == 4:
        phil = np.ones(len(xi))
        rhol = 0.
        logger.info("    rhol set to 0.")
    else:

        phil = eos.fugacity_coefficient(P, np.array([rhol]), xi, T)

        #muil = eos.chemicalpotential(P, np.array([rhol]), xi, T)
        #phil = np.exp(muil)

    print("Liquid Fugacity Coeff",phil)

    return phil, rhol, flagl

######################################################################
#                                                                    #
#                              Calc P range                          #
#                                                                    #
######################################################################
def calc_Prange_xi(T, xi, yi, eos, rhodict={}, Pmin=1000, zi_opts={}):
    r"""
    Obtain min and max pressure values, where the liquid mole fraction is set and the objective function at each of those values is of opposite sign.
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    Pmin : float, Optional, default: 1000.0
        Minimum pressure in pressure range that restricts searched space.
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    Prange : list
        List of min and max pressure range
    """

    logger = logging.getLogger(__name__)

    global yi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    if not len(extrema):
        Pmax = 100000 # 1 MPa
    else:
        Pmax = max(Pvspline(extrema))
    Parray = [Pmin, Pmax]

    #################### Find Pressure range and Objective Function values

    # Root of min from liquid curve is absolute minimum
    ObjArray = [0, 0]
    yi_range = yi

    ind = 0
    maxiter = 200
    for z in range(maxiter):
        # Find Obj Function for Min pressure above
        p = Parray[0]
        phil, rhol, flagl = calc_phil(p, T, xi, eos, rhodict=rhodict)
        if any(np.isnan(phil)):
            logger.error("Estimated minimum pressure is too low.")
            Parray[0] += Pmin
            continue
        yi_range, phiv, flagv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, **zi_opts)
        ObjArray[0] = (np.sum(xi * phil / phiv) - 1.0)
        logger.info("Estimated Minimum Pressure: {},  Obj. Func: {}".format(Parray[0],ObjArray[0]))
        if ObjArray[0] > 0:
            break
        else:
            Parray[0] /= 2

    if z == maxiter-1:
        logger.error("Proper minimum pressure for liquid density could not be found")
            
    for z in range(maxiter):
        if z == 0:
            # Find Obj function for Max Pressure above
            p = Parray[1]
            phil, rhol, flagl = calc_phil(p, T, xi, eos, rhodict=rhodict)
            if any(np.isnan(phil)):
                raise ValueError
            yi_range, phiv, flagv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, **zi_opts)
            ObjArray[1] = (np.sum(xi * phil / phiv) - 1.0)
            logger.info("Estimate Maximum Pressure: {},  Obj. Func: {}".format(Parray[1],ObjArray[1]))
        else:
            tmp_sum = np.abs(ObjArray[-2] + ObjArray[-1])
            tmp_dif = np.abs(ObjArray[-2] - ObjArray[-1])
            if tmp_dif > tmp_sum:
                if flagv not in [0,2,4]:
                    logger.info("Estimated pressure {}  doesn't produce a vapor, flag={}, Obj Func: {}".format(Parray[-1],flagv,ObjArray[-1]))
                    p = 0.9*Parray[-1]
                else:
                    logger.info("Got the pressure range!")
                    slope = (ObjArray[-1] - ObjArray[-2]) / (Parray[-1] - Parray[-2])
                    intercept = ObjArray[-1] - slope * Parray[-1]
                    Pguess = -intercept / slope

                #plt.plot(Parray,ObjArray)
                #plt.plot([Pguess,Pguess],[ObjArray[-1],ObjArray[-2]],'k')
                #plt.plot([Parray[0],Parray[-1]],[0,0],'k')
                #plt.ylabel("Obj. Function")
                #plt.xlabel("Pressure / Pa")
                #plt.show()
                break
            elif z == maxiter-1:
                logger.error('A change in sign for the objective function could not be found, inspect progress')
                plt.plot(Parray, ObjArray)
                plt.plot([Parray[0], Parray[-1]], [0, 0], 'k')
                plt.ylabel("Obj. Function")
                plt.xlabel("Pressure / Pa")
                plt.show()
            else:
                if len(ObjArray) < 2:
                    p = 2 * Parray[-1]
                else:
                    slope = (ObjArray[-1] - ObjArray[-2]) / (Parray[-1] - Parray[-2])
                    intercept = ObjArray[-1] - slope * Parray[-1]
                    p = (-intercept / slope)*1.2 # Add additional 20% to ensure negative value
            
            Parray.append(p)
            phil, rhol, flagl = calc_phil(p, T, xi, eos, rhodict=rhodict)
            if any(np.isnan(phil)):
                raise ValueError("Fugacity coefficient should not be NaN")
            yi_range, phiv, flagv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, **zi_opts)
            ObjArray.append(np.sum(xi * phil / phiv) - 1.0)
            logger.info("New Estimate for Maximum Pressure: {},  Obj. Func: {}".format(Parray[-1],ObjArray[-1]))

    Prange = Parray[-2:]
    ObjRange = ObjArray[-2:]
    logger.info("[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange),str(ObjRange)))
    logger.info("Initial guess in pressure: {} Pa".format(Pguess))

    yi_global = yi_range

    return Prange, Pguess

######################################################################
#                                                                    #
#                              Calc P range                          #
#                                                                    #
######################################################################
def calc_Prange_yi(T, xi, yi, eos, rhodict={}, Pmin=1000, zi_opts={}):
    r"""
    Obtain min and max pressure values, where the vapor mole fraction is set and the objective function at each of those values is of opposite sign.
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    Pmin : float, Optional, default: 1000.0
        Minimum pressure in pressure range that restricts searched space.
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    Prange : list
        List of min and max pressure range
    """

    logger = logging.getLogger(__name__)

    global xi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = PvsRho(T, yi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    # Calculation the highest pressure possbile
    Pmax = max(Pvspline(extrema))
    Parray = [Pmin, Pmax]

    ############################# Test
    pressures = np.linspace(Pmin,Pmax,30)
    pressure2 = np.linspace(pressures[-2],Pmax,20)
    pressures = np.concatenate((pressures,pressure2),axis=0)
    obj_list = []
    for p in pressures:
        phiv, rhov, flagv = calc_phiv(p, T, yi, eos, rhodict=rhodict)
        xi, phil, flagl = solve_xi_yiT(xi, yi, phiv, p, T, eos, rhodict=rhodict, **zi_opts)
        obj_list.append(np.sum(yi * phiv / phil) - 1.0)
    plt.plot(pressures,obj_list,".-")
    plt.plot([Pmin,Pmax],[0,0],"k")
    plt.show()

    #################### Find Pressure range and Objective Function values

    ObjArray = [0, 0]
    xi_range = xi

    for j,i in enumerate([0,1,0]):
        p = Parray[i]
        phiv, rhov, flagv = calc_phiv(p, T, yi, eos, rhodict=rhodict)
        xi_range, phil, flagl = solve_xi_yiT(xi_range, yi, phiv, p, T, eos, rhodict=rhodict, **zi_opts)
        ObjArray[i] = (np.sum(yi * phiv / phil) - 1.0)
        if i == 0:
            logger.info("Estimate Minimum pressure: {},  Obj. Func: {}".format(p,ObjArray[i]))
        elif i == 1:
            if ObjArray[i] < 1e-3:
                ObjArray[i] = 0.0
            logger.info("Estimate Maximum pressure: {},  Obj. Func: {}".format(p,ObjArray[i]))
        # Check pressure range
        if j < 2:
            tmp_sum = np.abs(ObjArray[0] + ObjArray[1])
            tmp_dif = np.abs(ObjArray[0] - ObjArray[1])
            if tmp_dif >= tmp_sum:
                logger.info("Got the pressure range!")
                slope = (ObjArray[1] - ObjArray[-2]) / (Parray[1] - Parray[0])
                intercept = ObjArray[1] - slope * Parray[1]
                Pguess = -intercept / slope
                break
            else:
                newPmin = 10
                if Parray[0] != newPmin:
                    Parray[0] = newPmin
                else:
                    raise ValueError("No VLE data may be found given this temperature and vapor composition. If there are no errors in parameter definitions, consider updating the thermo function 'solve_xi_yiT'.")
            
    Prange = Parray[-2:]
    ObjRange = ObjArray[-2:]
    logger.info("[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange),str(ObjRange)))
    logger.info("Initial guess in pressure: {} Pa".format(Pguess))

    xi_global = xi_range

    return Prange, Pguess


######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_yi_xiT(yi, xi, phil, P, T, eos, rhodict={}, maxiter=30, tol=1e-6):
    r"""
    Find vapor mole fraction given pressure, liquid mole fraction, and temperature. Objective function is the sum of the predicted "mole numbers" predicted by the computed fugacity coefficients. Note that by "mole number" we mean that the prediction will only sum to 1 when the correct pressure is chosen in the outer loop. In this inner loop, we seek to find a mole fraction that is converged to reproduce itself in a prediction. If it hasn't, the new "mole numbers" are normalized into mole fractions and used as the next guess.
    In the case that a guess doesn't produce a gas or critical fluid, we use another function to produce a new guess.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Guess in vapor mole fraction of each component, sum(xi) should equal 1.0
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    maxiter : int, Optional, default: 30
        Maximum number of iteration for both the outer pressure and inner vapor mole fraction loops
    tol : float, Optional, default: 1e-6
        Tolerance in sum of predicted yi "mole numbers"

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    phiv : float
        Fugacity coefficient of vapor at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    logger = logging.getLogger(__name__)

    global yi_global

    yi /= np.sum(yi)
    yi_total = [np.sum(yi)]
    flag_check_vapor = True # Make sure we only search for vapor compositions once
    logger.info("T {}, xi {}, phil {}".format(T, xi, phil))
    for z in range(maxiter):

        yi_tmp = yi/np.sum(yi)
        logger.info("    yi guess {}".format(yi_tmp))

        # Try yi
        phiv, rhov, flagv = calc_phiv(P, T, yi_tmp, eos, rhodict=rhodict)

        if ((any(np.isnan(phiv)) or flagv==1) and flag_check_vapor): # If vapor density doesn't exist
            flag_check_vapor = False
            logger.info("    Composition doesn't produce a vapor, let's find one!")
            if all(yi != 0.):
                yinew = find_new_yi(P, T, phil, xi, eos, rhodict=rhodict)
                phiv, rhov, flagv = calc_phiv(P, T, yinew, eos, rhodict=rhodict)
                yinew = xi * phil / phiv
            else:
                yinew = yi

            if any(np.isnan(phiv)):
                phiv = np.nan
                logger.error("Fugacity coefficient of vapor should not be NaN")
        else:
            yinew = xi * phil / phiv

        # Check for bouncing between values
        if len(yi_total) > 3:
            tmp1 =  (np.abs(np.sum(yinew)-yi_total[-2]) + np.abs(yi_total[-1]-yi_total[-3]))
            tmp2 = (np.abs(np.sum(yinew)-yi_total[-2]) + np.abs(yi_total[-1]-yi_total[-3])) < tol/1000
            if (tmp1 < np.abs(np.sum(yinew)-yi_total[-1]) and tmp1 < 1e-5):
                # This occurs when the P vs. v curve doesn't cross the 0 axis, there could be a larger problem causing this, but in my experience, it's because the curve is not long enough to converge to zero. Instead of the possible endless increase in vector length and a substantial increase in computational time, we simply set the fugacity coefficient to ideal and the density to 0. When an iteration on our assumption produces a vapor near ideality, it then may predict an ideal gas. This causes the constant back and forth that really isn't that important to solve, as the fugacity coefficients are unity regardless.
                logger.info("    yi_total is bouncing between {} and {}, choose the lowest value (outer loop obj. function).".format(np.sum(yinew),yi_total[-1]))
                if np.sum(yinew) > yi_total[-1]:
                    yinew = yi
                    phiv, rhov, flagv = calc_phiv(P, T, yi_tmp, eos, rhodict=rhodict)

        logger.info("    yi calc {}, phiv {}".format(yinew,phiv))
        logger.info("    Old yi_total: {}, New yi_total: {}, Change: {}".format(yi_total[-1],np.sum(yinew),np.sum(yinew)-yi_total[-1])) 

        # Check convergence
        if abs(np.sum(yinew)-yi_total[-1]) < tol:
            ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp>0]))[0] 
            yi2 = yinew/np.sum(yinew)
            if np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp] < tol:
                yi_global = yi_tmp
                logger.info("    Found yi")
                break

        if z < maxiter-1:
            yi_total.append(np.sum(yinew))
            yi = yinew

    ## If yi wasn't found in defined number of iterations
    yi_tmp = yi/np.sum(yi)
    yinew /= np.sum(yinew)

    ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp>0]))[0]
    if z == maxiter - 1:
        yi2 = yinew/np.sum(yinew)
        tmp = (np.abs(yinew[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp])
        logger.warning('    More than {} iterations needed. Error in Smallest Fraction: {} %%'.format(maxiter, tmp*100))
        if tmp > .1: # If difference is greater than 10%
            yinew = find_new_yi(P, T, phil, xi, eos, rhodict=rhodict)
        yinew = spo.least_squares(obj_yi, yinew[0], bounds=(0.,1.), args=(P, T, phil, xi, eos, rhodict))
        yi = yinew.x
        yi = np.array([yi,1-yi])
        obj = obj_yi(yi, P, T, phil, xi, eos, rhodict=rhodict)
        logger.warning('    Find yi with root algorithm, yi {}, obj {}'.format(yi,obj))
    else:
        logger.info("    Inner Loop Final yi: {}, Final Error on Smallest Fraction: {}".format(yi_tmp,np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp]*100))

    return yi_tmp, phiv, flagv

######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_xi_yiT(xi, yi, phiv, P, T, eos, rhodict={}, maxiter=20, tol=1e-6):
    r"""
    Find liquid mole fraction given pressure, vapor mole fraction, and temperature. Objective function is the sum of the predicted "mole numbers" predicted by the computed fugacity coefficients. Note that by "mole number" we mean that the prediction will only sum to 1 when the correct pressure is chosen in the outer loop. In this inner loop, we seek to find a mole fraction that is converged to reproduce itself in a prediction. If it hasn't, the new "mole numbers" are normalized into mole fractions and used as the next guess.
    In the case that a guess doesn't produce a liquid or critical fluid, we use another function to produce a new guess.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Guess in liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    phiv : float
        Fugacity coefficient of liquid at system pressure
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    maxiter : int, Optional, default: 20
        Maximum number of iteration for both the outer pressure and inner vapor mole fraction loops
    tol : float, Optional, default: 1e-6
        Tolerance in sum of predicted xi "mole numbers"

    Returns
    -------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    logger = logging.getLogger(__name__)

    global xi_global

    xi /= np.sum(xi)
    xi_total = np.sum(xi)
    for z in range(maxiter):

        xi /= np.sum(xi)
        logger.info("    xi guess {}".format(xi))

        # Try xi
        phil, rhol, flagl = calc_phil(P, T, xi, eos, rhodict=rhodict)

        if (any(np.isnan(phil)) or flagl==0): # If liquid density doesn't exist
            raise ValueError("This composition under these system conditions doesn't produce a liquid or critical fluid. This system must be approaching its critical point and has a suitably small pressure. No contingency function has been established.")

        xinew = yi * phiv / phil
        logger.info("    xi calc {}".format(xinew))
        logger.info("    Old xi_total: {}, New xi_total: {}, Change: {}".format(xi_total,np.sum(xinew),np.sum(xinew)-xi_total))

        # Check convergence
        if abs(np.sum(xinew)-xi_total) < tol:
            ind_tmp = np.where(xi == min(xi[xi>0]))[0]
            xi2 = xinew/np.sum(xinew)
            if np.abs(xi2[ind_tmp] - xi[ind_tmp]) / xi[ind_tmp] < tol:
                xi_global = xi
                logger.info("    Found xi")
                break

        if z < maxiter-1:
            xi = xinew/np.sum(xinew)
            xi_total = np.sum(xinew)

    ## If xi wasn't found in defined number of iterations
    xinew /= np.sum(xinew)

    ind_tmp = np.where(xi == min(xi[xi>0]))[0]
    if z == maxiter - 1:
        xi2 = xinew/np.sum(xinew)
        logger.warning('    More than {} iterations needed. Error in Smallest Fraction: {} %%'.format(maxiter, (np.abs(xi2[ind_tmp] - xi[ind_tmp]) / xi[ind_tmp])*100))
    else:
        logger.info("    Inner Loop Final xi: {}, Final Error on Smallest Fraction: {}".format(xi,np.abs(xi2[ind_tmp] - xi[ind_tmp]) / xi[ind_tmp]*100))

    return xi, phil, flagl

######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def find_new_yi(P, T, phil, xi, eos, rhodict={}):
    r"""
    Search vapor mole fraction combinations for a new estimate that produces a vapor density.
    
    Parameters
    ----------
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    phil : float
        Fugacity coefficient of liquid at system pressure
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """

    logger = logging.getLogger(__name__)

    yi_ext = np.linspace(0.01,.99,30) # Guess for yi
    obj_ext = []
    #flag_ext = []
    yi_total2_ext = []
    rho_ext = []
    phi_ext = []
    flag_ext = [[],[]]

    for yi in yi_ext:
        yi = np.array([yi, 1-yi])
        ####
        phiv, rhov, flagv = calc_phiv(P, T, yi, eos, rhodict=rhodict)
        yinew = xi * phil / phiv
        yinew_total_1 = np.sum(yinew)

        yi2 = yinew/yinew_total_1
        phiv2, rhov2, flagv2 = calc_phiv(P, T, yi2, eos, rhodict=rhodict)
        yinew = xi * phil / phiv2
        yinew_total_2 = np.sum(yinew)

        logger.debug("yi_totals {} {}".format(yinew_total_1,yinew_total_2))

        obj = yinew_total_1 - yinew_total_2

        flag_ext[0].append(flagv)
        flag_ext[1].append(flagv2)
        ######
    #    obj = obj_yi(yi, P, T, phil, xi, eos, rhodict=rhodict)
        obj_ext.append(abs(obj))
        logger.debug("    Obj yi {} total1 - total2 = {}".format(yi,obj))

    #plt.figure(1)
    #plt.plot(yi_ext,obj_ext,".-b")
    #plt.figure(2)
    #plt.plot(yi_ext,flag_ext[0],".-b")
    #plt.plot(yi_ext,flag_ext[1],".-r")
    #plt.show()

    obj_ext = np.array(obj_ext)
    flag_ext = np.array(flag_ext)
    print(type(flag_ext),type(flag_ext[0]))

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    logger.debug("    Number of valid mole fractions: {}".format(tmp))
    if tmp == 0:
        yi_tmp = np.nan
        obj_tmp = np.nan
    else:
        # Remove any NaN
        obj_tmp  =  obj_ext[~np.isnan(obj_ext)]
        yi_tmp   =   yi_ext[~np.isnan(obj_ext)]
        flag_tmp = flag_ext[1][~np.isnan(obj_ext)]
 
        # Assess vapor values
        ind = [i for i in range(len(flag_tmp)) if flag_tmp[i] not in [1,3]]
        if ind:
            obj_tmp = [obj_tmp[i] for i in ind]
            yi_tmp = [yi_tmp[i] for i in ind]

        # Choose values with lowest objective function
        ind = np.where(np.abs(obj_tmp)==min(np.abs(obj_tmp)))[0][0]
        obj_tmp = obj_tmp[ind]
        yi_tmp = yi_tmp[ind]

    logger.info("    Found new guess in yi: {}, Obj: {}".format(yi_tmp,obj_tmp))
    yi = yi_tmp
    if type(yi) not in [list,np.ndarray]:
        yi = np.array([yi, 1-yi])

    return yi

######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def obj_yi(yi, P, T, phil, xi, eos, rhodict={}):
    r"""
    Objective function for solving for stable vapor mole fraction.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    phil : float
        Fugacity coefficient of liquid at system pressure
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    obj : numpy.ndarray
        Objective function for solving for vapor mole fractions
    """

    logger = logging.getLogger(__name__)

    if type(yi) == float or len(yi) == 1:
        if type(yi) in [list, np.ndarray]:
            yi = np.array([yi[0], 1-yi[0]])
        else:
            yi = np.array([yi, 1-yi])

    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, rhodict=rhodict)
    yinew = xi * phil / phiv
    yinew_total_1 = np.sum(yinew)

    yi2 = yinew/yinew_total_1
    phiv2, rhov2, flagv2 = calc_phiv(P, T, yi2, eos, rhodict=rhodict)
    yinew = xi * phil / phiv2
    yinew_total_2 = np.sum(yinew)

    logger.debug("yi_totals {} {}".format(yinew_total_1,yinew_total_2))

    obj = yinew_total_1 - yinew_total_2
    
    return obj


######################################################################
#                                                                    #
#                              Solve Xi in root finding              #
#                                                                    #
######################################################################
def solve_xi_root(xi0, yi, phiv, P, T, eos, rhodict):
    r"""
    Objective function used to search liquid mole fraction and solve inner loop of dew point calculations.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Guess in liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    phiv : float
        Fugacity coefficient of vapor at system pressure
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    obj_value : list
        List of percent change between guess that is input and the updated version from recalculating with fugacity coefficients.
    """

    logger = logging.getLogger(__name__)

    xi0 /= np.sum(xi0)
    xi = xi0

    phil, rhol, flagl = calc_phil(P, T, xi, eos, rhodict={})
    xinew = yi * phiv / phil
    xinew /= np.sum(xinew)

    logger.info('    xi: {}, xinew: {}, Percent Error: {}'.format(xi,xinew,((xinew - xi)/xi*100)))

    ind_tmp = np.where(xi==min(xi))[0]
    return np.abs(xinew[ind_tmp]-xi[ind_tmp])/xi[ind_tmp]

######################################################################
#                                                                    #
#                       Solve Yi in Root Finding                     #
#                                                                    #
######################################################################

def solve_yi_root(yi0, xi, phil, P, T, eos, rhodict={}):
    r"""
    Objective function used to search vapor mole fraction and solve inner loop of bubble point calculations.
    
    Parameters
    ----------
    yi0 : numpy.ndarray
        Guess in vapor mole fraction of each component, sum(yi) should equal 1.0
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    P : float
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    obj_value : list
        List of absolute change between guess that is input and the updated version from recalculating with fugacity coefficients.
    """

    logger = logging.getLogger(__name__)

    yi0 /= np.sum(yi0)
    yi = yi0

    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, rhodict={})
    yinew = xi * phil / phiv
    yinew = yinew / np.sum(yinew)

    logger.info('    yi: {}, yinew: {}, Percent Error: {}'.format(yi,yinew,((yinew - yi)/yi*100)))

    ind_tmp = np.where(yi==min(yi))[0]
    return np.abs(yinew[ind_tmp]-yi[ind_tmp])/yi[ind_tmp]

######################################################################
#                                                                    #
#                              Solve P xT                            #
#                                                                    #
######################################################################
def solve_P_xiT(P, xi, T, eos, rhodict, zi_opts={}):
    r"""
    Objective function used to search pressure values and solve outer loop of P bubble point calculations.
    
    Parameters
    ----------
    P : float
        Guess in pressure of the system [Pa]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    

    Returns
    -------
    obj_value : float
        :math:`\sum\frac{x_{i}\{phi_l}{\phi_v}-1`
    """

    logger = logging.getLogger(__name__)

    global yi_global

    if P < 0:
        return 10.0

    logger.info("P Guess: {} Pa".format(P))

    #find liquid density
    phil, rhol, flagl = calc_phil(P, T, xi, eos, rhodict={})

    yinew, phiv, flagv = solve_yi_xiT(yi_global, xi, phil, P, T, eos, rhodict=rhodict, **zi_opts)
    yi_global = yi_global / np.sum(yi_global)

    #given final yi recompute
    phiv, rhov, flagv = calc_phiv(P, T, yi_global, eos, rhodict={})

    Pv_test = eos.P(rhov, T, yi_global)
    obj_value = float((np.sum(xi * phil / phiv) - 1.0))
    logger.info('Obj Func: {}, Pset: {}, Pcalc: {}'.format(obj_value, P, Pv_test[0]))

    return obj_value

######################################################################
#                                                                    #
#                              Solve P yT                            #
#                                                                    #
######################################################################
def solve_P_yiT(P, yi, T, eos, rhodict, zi_opts={}):
    r"""
    Objective function used to search pressure values and solve outer loop of P dew point calculations.
    
    Parameters
    ----------
    P : float
        Guess in pressure of the system [Pa]
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    obj_value : list
        :math:`\sum\frac{y_{i}\{phi_v}{\phi_l}-1`
    """

    logger = logging.getLogger(__name__)

    global xi_global

    if P < 0:
        return 10.0

    #find liquid density
    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, rhodict={})

    xi_global, phil, flagl = solve_xi_yiT(xi_global, yi, phiv, P, T, eos, rhodict=rhodict, **zi_opts)
    xi_global = xi_global / np.sum(xi_global)

    #given final yi recompute
    phil, rhol, flagl = calc_phil(P, T, xi_global, eos, rhodict={})

    Pv_test = eos.P(rhov, T, xi_global)
    obj_value = (np.sum(xi_global * phil / phiv) - 1.0)
    logger.info('Obj Func: {}, Pset: {}, Pcalc: {}'.format(obj_value, P, Pv_test[0]))

    return obj_value

######################################################################
#                                                                    #
#                   Set Psat for Critical Components                 #
#                                                                    #
######################################################################
def setPsat(ind, eos):
    r"""
    Generate dummy value for component saturation pressure if it is above its critical point.
    
    Parameters
    ----------
    ind : int
        Index of bead that is above critical point
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.

    Returns
    -------
    Psat : float
        Dummy value of saturation pressure [Pa]
    NaNbead : str
        Bead name of the component that is above it's critical point
    """

    logger = logging.getLogger(__name__)

    for j in range(np.size(eos._nui[ind])):
        if eos._nui[ind][j] > 0.0 and eos._beads[j] == "CO2":
            Psat = 10377000.0
        elif eos._nui[ind][j] > 0.0 and eos._beads[j] == "N2":
            Psat = 7377000.0
        elif eos._nui[ind][j] > 0.0 and ("CH4" in eos._beads[j]):
            Psat = 6377000.0
        elif eos._nui[ind][j] > 0.0 and ("CH3CH3" in eos._beads[j]):
            Psat = 7377000.0
        elif eos._nui[ind][j] > 0.0:
            #Psat = np.nan
            Psat = 7377000.0
            NaNbead = eos._beads[j]
            logger.warning("Bead, {}, is above its critical point. Psat is assumed to be {}. To add an exception go to thermodynamics.calc.setPsat".format(NaNbead,Psat))

    if "NaNbead" not in list(locals().keys()):
       NaNbead = "No NaNbead"
       logger.info("No beads above their critical point")

    return Psat, NaNbead 

######################################################################
#                                                                    #
#                              Calc yT phase                         #
#                                                                    #
######################################################################
def calc_yT_phase(yi, T, eos, rhodict={}, zi_opts={}, Pguess=-1, meth="hybr", pressure_opts={}):
    r"""
    Calculate dew point mole fraction and pressure given system vapor mole fraction and temperature.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default: -1
        Guess the system pressure at the dew point. A negative value will force an estimation based on the saturation pressure of each component.
    meth : str, Optional, default: "broyden1"
        Choose the method used to solve the dew point calculation
    pressure_opts : dict, Optional, default: {}
        Options used in the given method, "meth", to solve the outer loop in the solving algorithm

    Returns
    -------
    P : float
        Pressure of the system [Pa]
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    flagl : int
        Flag identifying the fluid type for the liquid mole fractions, expected is liquid, 1. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    flagv : int
        Flag identifying the fluid type for the vapor mole fractions, expected is vapor or 0. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    logger = logging.getLogger(__name__)

    global xi_global

    # Estimate pure component vapor pressures
    Psat = np.zeros_like(yi)
    for i in range(np.size(yi)):
        yi_tmp = np.zeros_like(yi)
        yi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, yi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                raise ValueError("Component, {}, is beyond it's critical point at {} K. Add an exception to setPsat".format(NaNbead,T))

    # Estimate initial pressure
    if Pguess < 0:
        P=1.0/np.sum(yi/Psat)
    else:
        P = Pguess

    # Estimate initial xi
    if ("xi_global" not in globals() or any(np.isnan(xi_global))):
        xi_global = P * (yi / Psat)
        xi_global /= np.sum(xi_global)
        xi_global = copy.deepcopy(xi_global)
    xi = xi_global 

    #Prange, Pguess = calc_Prange_yi(T, xi, yi, eos, rhodict, zi_opts=zi_opts)
    #logger.info("Given Pguess: {}, Suggested: {}".format(P, Pguess))
    #P = Pguess

    #################### Root Finding without Boundaries ###################
    if meth in ['broyden1', 'broyden2']:
        outer_dict = {'fatol': 1e-5, 'maxiter': 25, 'jac_options': {'reduction_method': 'simple'}}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth in ['hybr_broyden1', 'hybr_broyden2']:
        outer_dict = {'fatol': 1e-5, 'maxiter': 25, 'jac_options': {'reduction_method': 'simple'}}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        Pfinal = spo.root(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method="hybr")
        Pfinal = spo.root(solve_P_yiT, Pfinal.x, args=(yi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth == 'anderson':
        outer_dict = {'fatol': 1e-5, 'maxiter': 25}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth in ['hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']:
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)

#################### Minimization Methods with Boundaries ###################
    elif meth in ["TNC", "L-BFGS-B", "SLSQP"]:
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        if len(Prange) == 2:
            Pfinal = spo.minimize(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method=meth, bounds=[tuple(Prange)], options=outer_dict)
        else:
            Pfinal = spo.minimize(solve_P_yiT, P, args=(yi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)

#################### Root Finding with Boundaries ###################
    elif meth == "brent":
        outer_dict = {"rtol":1e-7}
        for key, value in pressure_opts.items():
            if key in ["xtol","rtol","maxiter","full_output","disp"]:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        P = spo.brentq(solve_P_yiT, Prange[0], Prange[1], args=(yi, T, eos, rhodict, zi_opts), **outer_dict)
    elif meth == "least_squares":
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.least_squares(solve_P_xiT, P, bounds=(Prange[0],Prange[1]), args=(xi, T, eos, rhodict, zi_opts), **outer_dict)

    #Given final P estimate
    if meth != "brent":
        P = Pfinal.x
        logger.info("Optimization terminated successfully: {} {}".format(Pfinal.success,Pfinal.message))

    #find vapor density and fugacity
    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, rhodict={})
    if "tol" in zi_opts:
        if zi_opts["tol"] > 1e-10:
            zi_opts["tol"] = 1e-10

    xi, phil, flagl = solve_xi_yiT(xi_global, yi, phiv, P, T, eos, rhodict, **zi_opts)
    xi_global = xi
    obj = solve_P_yiT(P, yi, T, eos, rhodict=rhodict)

    return P, xi, flagl, flagv, obj

######################################################################
#                                                                    #
#                              Calc xT phase                         #
#                                                                    #
######################################################################
def calc_xT_phase(xi, T, eos, rhodict={}, zi_opts={}, Pguess=-1, meth="hybr", pressure_opts={}):
    r"""
    Calculate bubble point mole fraction and pressure given system liquid mole fraction and temperature.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    zi_opts : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default: -1
        Guess the system pressure at the dew point. A negative value will force an estimation based on the saturation pressure of each component.
    meth : str, Optional, default: "broyden1"
        Choose the method used to solve the dew point calculation
    pressure_opts : dict, Optional, default: {}
        Options used in the given method, "meth", to solve the outer loop in the solving algorithm

    Returns
    -------
    P : float
        Pressure of the system [Pa]
    yi : numpy.ndarray
        Mole fraction of each component, sum(yi) should equal 1.0
    flagv : int
        Flag identifying the fluid type for the vapor mole fractions, expected is vapor or 0. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    flagl : int
        Flag identifying the fluid type for the liquid mole fractions, expected is liquid, 1. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    logger = logging.getLogger(__name__)

    global yi_global

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                logger.error("Component, {}, is beyond it's critical point. Add an exception to setPsat".format(NaNbead))

    # Estimate initial pressure
    if Pguess < 0:
        P=1.0/np.sum(xi/Psat)
    else:
        P = Pguess

    if ("yi_global" not in globals() or any(np.isnan(yi_global))):
        yi_global = xi * Psat / P
        yi_global /= np.sum(yi_global)
        yi_global = copy.deepcopy(yi_global)
        logger.info("Guess yi in calc_xT_phase with Psat: {}".format(yi_global))
    yi = yi_global

#    logger.info("Initial: P: {}, yi: {}".format(Pguess,str(yi)))
#    Pguess, yi = bubblepoint_guess(Pguess, yi, xi, T, phil, eos, rhodict)
#    logger.info("Updated: P: {}, yi: {}".format(Pguess,str(yi)))

    Prange, Pguess = calc_Prange_xi(T, xi, yi, eos, rhodict, zi_opts=zi_opts)
    logger.info("Given Pguess: {}, Suggested: {}".format(P, Pguess))
    P = Pguess

    if meth not in ["brent", "least_squares", "TNC", "L-BFGS-B", "SLSQP", 'hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane', 'anderson', 'hybr_broyden1', 'hybr_broyden2', 'broyden1', 'broyden2']:
        logger.error("Optimization method, {}, not supported.".format(meth))

    #################### Root Finding without Boundaries ###################
    if meth in ['broyden1', 'broyden2']:
        outer_dict = {'fatol': 1e-5, 'maxiter': 25, 'jac_options': {'reduction_method': 'simple'}}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth in ['hybr_broyden1', 'hybr_broyden2']:
        outer_dict = {'fatol': 1e-5, 'maxiter': 25, 'jac_options': {'reduction_method': 'simple'}}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        Pfinal = spo.root(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method="hybr")
        Pfinal = spo.root(solve_P_xiT, Pfinal.x, args=(xi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth == 'anderson':
        outer_dict = {'fatol': 1e-5, 'maxiter': 25}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)
    elif meth in ['hybr', 'lm', 'linearmixing', 'diagbroyden', 'excitingmixing', 'krylov', 'df-sane']:
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.root(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)

#################### Minimization Methods with Boundaries ###################
    elif meth in ["TNC", "L-BFGS-B", "SLSQP"]:
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        if len(Prange) == 2:
            Pfinal = spo.minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method=meth, bounds=[tuple(Prange)], options=outer_dict)
        else:
            Pfinal = spo.minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict, zi_opts), method=meth, options=outer_dict)

#################### Root Finding with Boundaries ###################
    elif meth == "brent":
        outer_dict = {"rtol":1e-7}
        for key, value in pressure_opts.items():
            if key in ["xtol","rtol","maxiter","full_output","disp"]:
                outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        P = spo.brentq(solve_P_xiT, Prange[0], Prange[1], args=(xi, T, eos, rhodict, zi_opts), **outer_dict)
    elif meth == "least_squares":
        outer_dict = {}
        for key, value in pressure_opts.items():
            outer_dict[key] = value
        logger.debug("Using the method, {}, with the following options:\n{}".format(meth,outer_dict))
        Pfinal = spo.least_squares(solve_P_xiT, P, bounds=(Prange[0],Prange[1]), args=(xi, T, eos, rhodict, zi_opts), **outer_dict)

    #Given final P estimate
    if meth != "brent":
        P = Pfinal.x

    #find liquid density and fugacity
    phil, rhol, flagl = calc_phil(P, T, xi, eos, rhodict={})
    if "tol" in zi_opts:
        if zi_opts["tol"] > 1e-10:
            zi_opts["tol"] = 1e-10

    yi, phiv, flagv = solve_yi_xiT(yi_global, xi, phil, P, T, eos, rhodict, **zi_opts)
    yi_global = yi
    obj = solve_P_xiT(P, xi, T, eos, rhodict=rhodict)

    return P, yi_global, flagv, flagl, obj

######################################################################
#                                                                    #
#                              Calc PT phase                         #
#                                                                    #
######################################################################
def calc_PT_phase(xi, T, eos, rhodict={}):
    r"""
    **Not Complete**
    Calculate the PT phase diagram given liquid mole fraction and temperature
    
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    P : float
        Pressure of the system [Pa]
    yi : numpy.ndarray
        Mole fraction of each component, sum(yi) should equal 1.0
    """

    logger = logging.getLogger(__name__)

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                logger.error("Component, {}, is beyond it's critical point. Add an exception to setPsat".format(NaNbead))

    zi = np.array([0.5, 0.5])

    #estimate ki
    ki = Psat / P

    #estimate beta (not thermodynamic) vapor frac
    beta = (1.0 - np.sum(ki * zi)) / np.prod(ki - 1.0)


######################################################################
#                                                                    #
#                              Calc dadT                             #
#                                                                    #
######################################################################
def calc_dadT(rho, T, xi, eos, rhodict={}):
    r"""
    Calculate the derivative of the Helmholtz energy with respect to temperature, :math:`\frac{dA}{dT}`, give a list of density values and system conditions.
    
    Parameters
    ----------
    rho : numpy.ndarray
        Density array. Length depends on values in rhodict [mol/:math:`m^3`]
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    dadT : numpy.ndarray
        Array of derivative values of Helmholtz energy with respect to temperature
    """

    logger = logging.getLogger(__name__)

    step = np.sqrt(np.finfo(float).eps) * T * 1000.0
    nrho = np.size(rho)

    #computer rho+step and rho-step for better a bit better performance
    Ap = calchelmholtz.calc_A(np.array([rho]), xi, T + step, eos)
    Am = calchelmholtz.calc_A(np.array([rho]), xi, T - step, eos)

    return (Ap - Am) / (2.0 * step)

