"""
This module contains our thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS. The thermo module contains a series of wrapper to handle the inputs and outputs of these functions.
    
"""

import sys
import numpy as np
from scipy import interpolate
import scipy.optimize as spo
from scipy.ndimage.filters import gaussian_filter1d
import copy
#import matplotlib.pyplot as plt
import logging

import despasito.utils.general_toolbox as gtb
from . import fund_constants as constants

logger = logging.getLogger(__name__)

######################################################################
#                                                                    #
#                     Calculate Critical Parameters                  #
#                                                                    #
######################################################################
def calc_CC_Pguess(xilist, Tlist, CriticalProp):
    r"""
    Computes the Mie parameters of a mixture from the mixed critical properties of the pure components. 

    From: Mejia, A., C. Herdes, E. Muller. Ind. Eng. Chem. Res. 2014, 53, 4131-4141
    
    Parameters
    ----------
    xilist : list[list[xi]]
        List of sets of component mole fraction, where sum(xi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist.
    Tlist : list[float]
       [K] Temperature of the system corresponding to composition in xilist
    CriticalProp : list[list]
        List of critical properties :math:`T_C`, :math:`P_C`, :math:`\omega`, :math:`\rho_{0.7}`, :math:`Z_C`, :math:`V_C`, and molecular weight, where each of these properties is a list of values for each bead.
    
    Returns
    -------
    Psatm : list[float]
        [Pa] A list of guesses in pressure based on critical properties, of the same length as xilist and Tlist
    """

    Tc, Pc, omega, rho_7, Zc, Vc, M = CriticalProp

    ############## Calculate Mixed System Mie Parameters
    flag = 0
    if all(-0.847 > x > 0.2387 for x in omega):
        flag = 1
        logger.warning("Omega is outside of the range that these correlations are valid")

    # Fit parameters from: Mejia, A., C. Herdes, E. Muller. Ind. Eng. Chem. Res. 2014, 53, 4131-4141
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
                Psat_tmp, _, _ = calc_Psat(Tlist[kk], np.array([1.0]), eos)
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
def PvsRho(T, xi, eos, minrhofrac=(1.0 / 500000.0), rhoinc=5.0, vspacemax=1.0E-4, maxrho = None, **kwargs):

    r"""
    Give an array of density values, calculates the associated pressure given an eos.
    
    Parameters
    ----------
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    minrhofrac : float, Optional, default: (1.0/500000.0)
        Fraction of the maximum density used to calculate, and is equal to, the minimum density of the density array. The minimum density is the reciprocal of the maximum specific volume used to calculate the roots. Passed from inputs to through the dictionary density_dict.
    rhoinc : float, Optional, default: 5.0
        The increment between density values in the density array. Passed from inputs to through the dictionary density_dict.
    vspacemax : float, Optional, default: 1.0E-4
        Maximum increment between specific volume array values. After conversion from density to specific volume, the increment values are compared to this value. Passed from inputs to through the dictionary density_dict.
    maxrho : float, Optional, default: None
        [mol/m^3] Maximum molar density defined, if default of None is used then the eos object method, density_max is used.

    Returns
    -------
    vlist : numpy.ndarray
        [:math:`m^3`/mol] Specific volume array. Length depends on values in density_dict
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    """

    if type(xi) == list:
        xi = np.array(xi)

    #estimate the maximum density based on the hard sphere packing fraction, part of EOS
    if not maxrho:
        maxrho = eos.density_max(xi, T, **kwargs)
    elif type(maxrho) in [list, np.ndarray]:
        logger.error("Maxrho should be type float. Given value: {}".format(maxrho))
  
    if maxrho > 1e+5:
        raise ValueError("Max density of {} mol/m^3 is not feasible, check parameters.".format(maxrho))

    #min rho is a fraction of max rho, such that minrho << rhogassat
    minrho = maxrho * minrhofrac
    #list of densities for P,rho and P,v
    if (maxrho-minrho) < rhoinc:
        raise ValueError("Density range, {}, is less than incement, {}. Check parameters used in eos.density_max().".format((maxrho-minrho),rhoinc))
 
    rholist = np.arange(minrho, maxrho, rhoinc)
    #check rholist to see when the spacing
    vspace = (1.0 / rholist[:-1]) - (1.0 / rholist[1:])
    if np.amax(vspace) > vspacemax:
        vspaceswitch = np.where(vspace > vspacemax)[0][-1]
        rholist_2 = 1.0 / np.arange(1.0 / rholist[vspaceswitch + 1], 1.0 / minrho, vspacemax)[::-1]
        rholist = np.append(rholist_2, rholist[vspaceswitch + 2:])

    #compute Pressures (Plist) for rholist
    Plist = eos.pressure(rholist, T, xi)

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
        [:math:`m^3`/mol] Specific volume array. Length depends on values in density_dict
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    
    Returns
    -------
    Pvspline : obj
        Function object of pressure vs. specific volume
    roots : list
        List of specific volume roots. Subtract a system pressure from the output of Pvsrho to find density of vapor and/or liquid densities.
    extrema : list
        List of specific volume values corresponding to local minima and maxima.
    """

    # Larger sigma value 
    Psmoothed = gaussian_filter1d(Plist, sigma=1.0e-2)

    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed)
    roots = Pvspline.roots().tolist()
    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed, k=4)
    extrema = Pvspline.derivative().roots().tolist()
    if extrema: 
        if len(extrema) > 2: extrema = extrema[0:2]

    #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

    if np.any(np.isnan(Plist)):
        roots = [np.nan]

    return Pvspline, roots, extrema

######################################################################
#                                                                    #
#                      Pressure-Volume Spline                        #
#                                                                    #
######################################################################
def interp_vroot(v0, vlist, Plist):
    r"""
    Take estimate of the specific volume root in P vs. specific volume curve, and find the closest root in the array to interpolate the value. If the root already lies between the closest points, then the same value is reported. This helps improve accuracy of liquid roots where the spline may not correctly represent the near vertical trend because we used a cublic spline to allow use of the derivative method.
    
    Parameters
    ----------
    v0 : float
        [:math:`m^3`/mol] This guess in the specific volume roots is tested
    vlist : numpy.ndarray
        [:math:`m^3`/mol] Specific volume array. Length depends on values in density_dict
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    
    Returns
    -------
    v0_new : float
        [:math:`m^3`/mol] This is either the same value as given, or a new estimate interpolated from the closest points that change sign.
    """

    # Find roots through change in sign
    ind_array = np.zeros(4,int)
    for i in range(3):
        if ind_array[i]+1 < len(Plist):
            tmp = np.where(Plist[ind_array[i]:]*(-1)**(i) < 0)[0]
            if any(tmp):
                ind_array[i+1] = tmp[0]+ind_array[i]
            else:
                break
    ind_array = ind_array[ind_array>0]

    # Find which root is closest to the estimate given
    Nind = len(ind_array)
    if Nind == 0:
        logger.warning("No roots found in given Plist. Return given v_root")
    elif Nind == 1:
        ind = ind_array[0]
    else:
         tmp = np.abs(vlist[ind_array]-v0)
         ind_tmp = np.where(tmp==np.min(tmp))[0][0]
         ind = ind_array[ind_tmp]

    # Assess and possibly reestimate v_root
    if (v0 < vlist[ind-1] or v0 > vlist[ind]):
        m = (Plist[ind]-Plist[ind-1])/(vlist[ind]-vlist[ind-1])
        v0_new = -m/(Plist[ind]-m*vlist[ind]) 
        logger.debug("    Reestimate root: v0={}, v0_new={}".format(v0,v0_new))
    else:
        logger.debug("    v_root is within bounds, ({},{}). Return given v_root, {}.".format(vlist[ind-1],vlist[ind],v0))
        v0_new = v0

    return v0_new

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
        [:math:`m^3`/mol] Specific volume array. Length depends on values in density_dict
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    Pvspline : obj
        Function object of pressure vs. specific volume
    markers : list, Optional, default: []
        List of plot markers used in plot
    """

    plt.figure(1)
    plt.plot(vlist,Plist,label="Orig.")
    plt.plot(vlist,Pvspline(vlist),label="Smoothed")
    plt.plot([vlist[0], vlist[-1]],[0,0],"k")
    for k in range(len(markers)):
        plt.plot([markers[k], markers[k]],[min(Plist),max(Plist)],"k")
    plt.xlabel("Specific Volume [$m^3$/mol]"), plt.ylabel("Pressure [Pa]")
#    plt.ylim(min(Plist)/2,np.abs(min(Plist))/2)
    plt.xlim(0.0,0.001)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

######################################################################
#                                                                    #
#                              Calc Psat                             #
#                                                                    #
######################################################################
def calc_Psat(T, xi, eos, density_dict={}):
    r"""
    Computes the saturated pressure, gas and liquid densities for a single component system.
    
    Parameters
    ----------
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    Psat : float
        [Pa] Saturation pressure given system information
    rhov : float
        [mol/:math:`m^3`] Density of vapor at saturation pressure
    rhol : float
        [mol/:math:`m^3`] Density of liquid at saturation pressure
    """

    if np.count_nonzero(xi) != 1:
        if np.count_nonzero(xi>0.1) != 1:
            raise ValueError("Multiple components have compositions greater than 10%, check code for source")
        else:
            ind = np.where((xi>0.1)==True)[0]
            raise ValueError("Multiple components have compositions greater than 0. Do you mean to obtain the saturation pressure of {} with a mole fraction of {}?".format(eos.beads[ind],xi[ind]))

    vlist, Plist = PvsRho(T, xi, eos, **density_dict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    if (not extrema or len(extrema)<2 or np.any(np.isnan(roots))):
        logger.warning('Error: One of the components is above its critical point, add an exception to setPsat')
        Psat = np.nan
        roots = [1.0, 1.0, 1.0]

    else:
        ind_Pmin1 = np.argwhere(np.diff(Plist) > 0)[0][0]
        ind_Pmax1 = np.argmax(Plist[ind_Pmin1:]) + ind_Pmin1

        Pmaxsearch = Plist[ind_Pmax1]

        Pconverged = 10 # If the pressure is negative (under tension), we search from a value just above vacuum 
        Pminsearch = max(Pconverged, np.amin(Plist[ind_Pmin1:ind_Pmax1]))

        #search Pressure that gives equal area in maxwell construction
      #  try:
        Psat = spo.minimize_scalar(eq_area,
                               args=(Plist, vlist),
                               bounds=(Pminsearch, Pmaxsearch),
                               method='bounded')

        #Using computed Psat find the roots in the maxwell construction to give liquid (first root) and vapor (last root) densities
        Psat = Psat.x
        Pvspline, roots, extrema = PvsV_spline(vlist, Plist-Psat)
      #  except:
      #      PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

        logger.debug("    Psat found: {} Pa, obj value: {}, with {} roots and {} extrema".format(Psat,eq_area(Psat,Plist,vlist),np.size(roots),np.size(extrema)))

        if len(roots) ==2:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:]-Psat, 1)
            vroot = -yroot/slope
            if vroot < 0.0:
                vroot = np.finfo(float).eps
            rho_tmp = spo.minimize(Pdiff, 1.0/vroot, args=(Psat, T, xi, eos), bounds=[(1.0/(vroot*1e+2), 1.0/(1.1*roots[-1]))])
            roots = np.append(roots,[1.0/rho_tmp.x])

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
        [Pa] Guess in Psat value used to translate the pressure vs. specific volume curve
    Pv : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    vlist : numpy.ndarray
        [mol/:math:`m^3`] Specific volume array. Length depends on values in density_dict

    Returns
    -------
    obj_value : float
        Output of objective function, the addition of the positive area between first two roots, and negative area between second and third roots, quantity squared.

    """

    Pvspline, roots, extrema = PvsV_spline(vlist, Pv-shift)

    if len(roots) >=3:
        a = Pvspline.integral(roots[0], roots[1])
        b = Pvspline.integral(roots[1], roots[2])
    elif len(roots) == 2:
        a = Pvspline.integral(roots[0], roots[1])
        # If the curve hasn't decayed to 0 yet, estimate the remaining area as a triangle. This isn't super accurate but we are just using the saturation pressure to get started.
        slope, yroot = np.polyfit(vlist[-4:], Pv[-4:]-shift, 1)
        b = Pvspline.integral(roots[1], vlist[-1]) + (Pv[-1]-shift)*(-yroot/slope-vlist[-1])/2
    elif np.any(np.isnan(roots)):
        logger.warning("Pressure curve without cubic properties has wrongly been accepted. Try decreasing pressure")
    else:
        logger.warning("Pressure curve without cubic properties has wrongly been accepted. Try decreasing minrhofrac")
        #PvsV_plot(vlist, Pv-shift, Pvspline, markers=extrema)

    return (a + b)**2

######################################################################
#                                                                    #
#                              Calc Rho V Full                       #
#                                                                    #
######################################################################
def calc_rhov(P, T, xi, eos, density_dict={}):
    r"""
    Computes vapor density under system conditions.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    rhov : float
        [mol/:math:`m^3`] Density of vapor at system pressure
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    """

    vlist, Plist = PvsRho(T, xi, eos, **density_dict)
    Plist = Plist-P
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    logger.debug("    Find rhov: P {} Pa, roots {} m^3/mol".format(P,roots))

    flag_NoOpt = False
    l_roots = len(roots)
    if np.any(np.isnan(roots)):
        rho_tmp = np.nan
        flag = 3
        logger.warning("    Flag 3: The T and yi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(T,xi))
    elif l_roots == 0:
        if Pvspline(1/vlist[-1]) < 0:
            try:
                rho_tmp = spo.least_squares(Pdiff, 1/vlist[0], args=(P, T, xi, eos), bounds=(np.finfo("float").eps, eos.density_max(xi, T, maxpack=0.99)))
                rho_tmp = rho_tmp.x
                if not len(extrema):
                    flag = 2
                    logger.debug("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
                else:
                    flag = 1
                    logger.debug("    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(T,xi))
            except:
                rho_tmp = np.nan
                flag = 3
                logger.warning("    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure, without density greater than max, {}".format(T,xi,eos.density_max(xi, T, maxpack=0.99)))
            flag_NoOpt = True
        elif min(Plist)+P > 0:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot/slope
            try:
                rho_tmp = spo.least_squares(Pdiff, 1/vroot, args=(P, T, xi, eos), bounds=(np.finfo("float").eps, 1.0/(1.1*roots[-1])))
                rho_tmp = rho_tmp.x
                flag = 0
            except:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                logger.debug("    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
        else:
            logger.warning("    Flag 3: The T and yi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(T,xi))
            flag = 3
            rho_tmp = np.nan
    elif l_roots == 1:
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
        elif (Pvspline(roots[0])+P) > (Pvspline(max(extrema))+P):
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(T,xi))
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
    elif l_roots == 2:
        if (Pvspline(roots[0])+P) < 0.:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 1: This T and yi, {} {}, combination produces a liquid under tension at this pressure".format(T,xi))
        else:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot/slope
            try:
                rho_tmp = spo.least_squares(Pdiff, 1/vroot, args=(P, T, xi, eos), bounds=(np.finfo("float").eps, 1.0/(1.1*roots[-1])))
                rho_tmp = rho_tmp.x
                flag = 0
            except:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug("    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                logger.debug("    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
    else: # 3 roots
        logger.debug("    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure.".format(T,xi))
        rho_tmp = 1.0 / roots[2]
        flag = 0

    if flag in [0,2]: # vapor or critical fluid
        tmp = [rho_tmp*.99, rho_tmp*1.01]
        if (rho_tmp*1.01 > eos.density_max(xi, T, maxpack=0.99)):
            tmp[1] = eos.density_max(xi, T, maxpack=0.99)

        if (Pdiff(tmp[0],P, T, xi, eos)*Pdiff(tmp[1],P, T, xi, eos))<0:
            rho_tmp = spo.brentq(Pdiff, tmp[0], tmp[1], args=(P, T, xi, eos), rtol=0.0000001)
        else:
            if Plist[0] < 0:
                logger.warning("Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(tmp))
            elif not flag_NoOpt:
                rho_tmp = spo.least_squares(Pdiff, rho_tmp, args=(P, T, xi, eos), bounds=(np.finfo("float").eps, eos.density_max(xi, T, maxpack=0.99)))
                rho_tmp = rho_tmp.x

    logger.info("    Vapor Density: {} mol/m^3, flag {}".format(rho_tmp,flag))

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    return rho_tmp, flag


######################################################################
#                                                                    #
#                              Calc Rho L Full                       #
#                                                                    #
######################################################################
def calc_rhol(P, T, xi, eos, density_dict={}):
    r"""
    Computes liquid density under system conditions.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    rhol : float
        [mol/:math:`m^3`] Density of liquid at system pressure
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    # Get roots and local minima and maxima 
    vlist, Plist = PvsRho(T, xi, eos, **density_dict)
    Plist = Plist-P
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    logger.debug("    Find rhol: P {} Pa, roots {} m^3/mol".format(P,str(roots)))
    flag_NoOpt = False

    if extrema:
        if len(extrema) == 1:
            logger.warning("    One extrema at {}, assume weird minima behavior. Check your parameters.".format(1/extrema[0])) 

    # Assess roots, what is the liquid density
    l_roots = len(roots)
    if np.any(np.isnan(roots)):
        rho_tmp = np.nan
        flag = 3
        logger.warning("    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(T,xi))
    elif l_roots == 0:
        if Pvspline(1/vlist[-1]):
            try:
                bounds = (1/vlist[0], eos.density_max(xi, T, maxpack=0.99))
                rho_tmp = spo.least_squares(Pdiff, np.mean(bounds), args=(P, T, xi, eos), bounds=bounds)
                rho_tmp = rho_tmp.x
                if not len(extrema):
                    flag = 2
                    logger.debug("    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
                else:
                    flag = 1
                    logger.debug("    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(T,xi))
            except:
                rho_tmp = np.nan
                flag = 3
                logger.warning("    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure, without density greater than max, {}".format(T,xi,eos.density_max(xi, T, maxpack=0.99)))
            flag_NoOpt = True
        elif min(Plist)+P > 0:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot/slope
            try:
                rho_tmp = spo.least_squares(Pdiff, 1.0/vroot, args=(P, T, xi, eos), bounds=(np.finfo("float").eps, 1.0/(1.1*roots[-1])))
                rho_tmp = rho_tmp.x
                flag = 0
            except:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug("    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
            else:
                logger.debug("    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
        else:
            flag = 3
            logger.error("    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(str(T),str(xi)))
            rho_tmp = np.nan
            #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)
    elif l_roots == 2: # 2 roots
        if (Pvspline(roots[0])+P) < 0.:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 1: This T and xi, {} {}, combination produces a liquid under tension at this pressure".format(T,xi))
        else: # There should be three roots, but the values of specific volume don't go far enough to pick up the last one
            flag = 1
            rho_tmp = 1.0 / roots[0]
    elif l_roots == 1: # 1 root
        if not len(extrema):
            flag = 2
# NoteHere
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(T,xi))
        elif (Pvspline(roots[0])+P) > (Pvspline(max(extrema))+P):
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(T,xi))
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.debug("    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(T,xi))
    else: # 3 roots
        rho_tmp = 1.0 / roots[0]
        flag = 1
        logger.debug("    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(T,xi))

    if flag in [1,2]: # liquid or critical fluid
        tmp = [rho_tmp*.99, rho_tmp*1.01]
        P_tmp = [Pdiff(tmp[0],P, T, xi, eos), Pdiff(tmp[1],P, T, xi, eos)]
        if (P_tmp[0]*P_tmp[1])<0:
            rho_tmp = spo.brentq(Pdiff, tmp[0], tmp[1], args=(P, T, xi, eos), rtol=1e-7)
        else:
            if P_tmp[0] < 0:
                logger.warning("Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(tmp))
            elif not flag_NoOpt:
                rho_tmp = spo.least_squares(Pdiff, rho_tmp, args=(P, T, xi, eos), bounds=(np.finfo("float").eps, eos.density_max(xi, T, maxpack=0.99)))
                rho_tmp = rho_tmp.x[0]
    logger.info("    Liquid Density: {} mol/m^3, flag {}".format(rho_tmp,flag))

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    return rho_tmp, flag

######################################################################
#                                                                    #
#                              Calc Pdiff                            #
#                                                                    #
######################################################################
def Pdiff(rho, Pset, T, xi, eos):
    """
    Calculate difference between set point pressure and computed pressure for a given density.
    
    Parameters
    ----------
    rho : float
        [mol/:math:`m^3`] Density of system
    Pset : float
        [Pa] Guess in pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    
    Returns
    -------
    Pdiff : float
        [Pa] Difference in set pressure and predicted pressure given system conditions.
    """

    Pguess = eos.pressure(rho, T, xi)

    return (Pguess - Pset)

######################################################################
#                                                                    #
#                          Calc phi vapor                            #
#                                                                    #
######################################################################
def calc_phiv(P, T, yi, eos, density_dict={}):
    r"""
    Computes vapor fugacity coefficient under system conditions.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    yi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    phiv : float
        Fugacity coefficient of vapor at system pressure
    rhov : float
        [mol/:math:`m^3`] Density of vapor at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    rhov, flagv = calc_rhov(P, T, yi, eos, density_dict)
    if flagv == 4:
        phiv = np.ones_like(yi)
        rhov = 0.
        logger.info("    rhov set to 0.")
    elif flagv == 3:
        phiv = np.array([np.nan,np.nan])
    else:
        phiv = eos.fugacity_coefficient(P, np.array([rhov]), yi, T)

        #muiv = eos._chemicalpotential_old(P, np.array([rhov]), yi, T)
        #phiv = np.exp(muiv)

    #logger.debug("    Vapor Fugacity Coefficients {}".format(phiv))

    return phiv, rhov, flagv

######################################################################
#                                                                    #
#                         Calc phi liquid                            #
#                                                                    #
######################################################################
def calc_phil(P, T, xi, eos, density_dict={}):
    r"""
    Computes liquid fugacity coefficient under system conditions.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    phil : float
        Fugacity coefficient of liquid at system pressure
    rhol : float
        [mol/:math:`m^3`] Density of liquid at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true.
    """

    rhol, flagl = calc_rhol(P, T, xi, eos, density_dict)
    if flagl == 3:
        phil = np.array([np.nan,np.nan])
    else:
        phil = eos.fugacity_coefficient(P, np.array([rhol]), xi, T)

    #logger.debug("    Liquid Fugacity Coefficients {}".format(phil))

    return phil, rhol, flagl

######################################################################
#                                                                    #
#                          Calc xi                                   #
#                                                                    #
######################################################################
def calc_yi(xi, phil, phiv):
    r"""
    Computes vapor fugacity coefficient under system conditions.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    phiv : float
        Fugacity coefficient of vapor at system pressure

    Returns
    -------
    yi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    """

    yi = np.zeros(len(xi))
    ind = np.where(xi != 0.0)[0]
    for i in ind:
        yi[i] = xi[i] * phil[i] / phiv[i]

    return yi

######################################################################
#                                                                    #
#                          Calc yi                                   #
#                                                                    #
######################################################################
def calc_xi(yi, phiv, phil):
    r"""
    Computes vapor fugacity coefficient under system conditions.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Mole fraction of each component, sum(yi) should equal 1.0
    phiv : float
        Fugacity coefficient of vapor at system pressure
    phil : float
        Fugacity coefficient of liquid at system pressure

    Returns
    -------
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    """

    xi = np.zeros(len(yi))
    ind = np.where(yi != 0.0)[0]
    for i in ind:
        xi[i] = yi[i] * phiv[i] / phil[i]

    return xi

######################################################################
#                                                                    #
#                   Clean Plot Data                                  #
#                                                                    #
######################################################################
def _clean_plot_data(x_old, y_old):
    r"""
    Reorder array and remove duplicates, then repeat process for the corresponding array.

    Parameters
    ----------
    x_old : numpy.ndarray
        Original independent variable
    y_old : numpy.ndarray
        Original dependent variable

    Returns
    -------
    x_new : numpy.ndarray
        New independent variable
    y_new : numpy.ndarray
        New dependent variable
    """

    x_new = np.sort(np.array(list(set(x_old))))
    y_new = np.array([y_old[np.where(np.array(x_old)==x)[0][0]] for x in x_new])

    return x_new, y_new

######################################################################
#                                                                    #
#                              Calc P range                          #
#                                                                    #
######################################################################
def calc_Prange_xi(T, xi, yi, eos, density_dict={}, Pmin=10000, mole_fraction_options={}):
    r"""
    Obtain min and max pressure values.

    The liquid mole fraction is set and the objective function at each of those values is of opposite sign.
    
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
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    Pmin : float, Optional, default: 1000.0
        [Pa] Minimum pressure in pressure range that restricts searched space.
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    Prange : list
        List of min and max pressure range
    """

    global yi_global

    tol = 1e-2
    flag_liqu = False

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = PvsRho(T, xi, eos, **density_dict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

    if not len(extrema):
        Pmax = 100000 # 1 MPa
    else:
        Pmax = max(Pvspline(extrema))
    Prange = np.array([Pmin, Pmax])

    #################### Find Pressure range and Objective Function values

    # Root of min from liquid curve is absolute minimum
    ObjRange = np.zeros(2)
    yi_range = yi

    maxiter = 200
    flag_min = False
    p = Prange[0]
    for z in range(maxiter):
        # Find Obj Function for Min pressure above
        phil, rhol, flagl = calc_phil(p, T, xi, eos, density_dict=density_dict)
        if any(np.isnan(phil)):
            logger.error("Estimated minimum pressure is too low.")
            flag_liqu = False
            flag_min=True
            ObjRange[0] = np.inf
            Prange[0] = Pmin
            p = 2*Pmin
            continue

        if flagl in [1,2]:
            yi_range, phiv_min, flagv_min = solve_yi_xiT(yi_range, xi, phil, p, T, eos, density_dict=density_dict, **mole_fraction_options)
            obj = (np.nansum(xi * phil / phiv_min) - 1.0)
            logger.debug("Liquid / Vapor Phi: {} /  {}".format(phil,phiv_min))
            if np.any(np.isnan(yi_range)):
                logger.info("Estimated Minimum Pressure produces NaN")
                Prange[1] = p
                ObjRange[1] = obj
                p = (Prange[1]-Prange[0])/2.0 + Prange[0]
            elif (np.sum(np.abs(xi-yi_range)/xi) < 0.02 and flagv_min==2): # If within 2% of liquid mole fraction
                logger.info("Estimated Minimum Pressure Reproduces xi: {},  Obj. Func: {}, Range {}".format(p,obj,Prange))
                if p < 5: # Less than a Pa
                    flag_liqu = False
                    flag_min=True
                    ObjRange[0] = np.inf
                    Prange[0] = Pmin
                    p = 2*Pmin
                elif flag_min:
                    ObjRange[0] = obj
                    Prange[0] = p
                    p = 2*p
                else:
                    flag_liqu = True
                    ObjRange[1] = obj
                    Prange[1] = p
                    phiv_max, flagv_max = phiv_min, flagv_min
                    p = p/2
            elif obj > 0:
                Prange[0] = p
                ObjRange[0] = obj
                logger.info("Estimated Minimum Pressure: {},  Obj. Func: {}, Range {}".format(p,obj,Prange))
                break
            elif obj < 0:
                flag_liqu = True
                logger.info("Estimated Minimum Pressure too High: {},  Obj. Func: {}, Range {}".format(p,obj,Prange))
                ObjRange[1] = obj
                Prange[1] = p
                phiv_max, flagv_max = phiv_min, flagv_min
                p /= 2
            else:
                logger.info("Estimated Minimum Pressure Produced Vapor: {}, Range {}".format(p,Prange))
                Prange[0] = p
                ObjRange[0] = obj
                p = 2*Prange[0]
        else:
            logger.info("Estimated Minimum Pressure Produced Vapor: {}, Range {}".format(p,Prange))
            Prange[0] = p
            ObjRange[0] = np.nan
            p = 2*Prange[0]

        if Prange[0] > Prange[1]:
            Prange[1] = 2*Prange[0]
            ObjRange[1] = ObjRange[0]

        if (flag_min and flag_liqu and p not in range(Prange[0],Prange[1])):
        #if (p < Prange[0] and Prange[0] != Prange[1]) or (flag_liqu and p > Prange[1]):
            p = (Prange[1]-Prange[0])*np.random.rand(1)[0] + Prange[0]

        if p <= 0.:
            raise ValueError("Pressure, {}, cannot be equal to or less than zero. Given composition, {}, and T {}, results in a supercritical value without a coexistent fluid.".format(p,xi,T))

    if z == maxiter-1:
        raise ValueError("Maximum Number of Iterations Reached: Proper minimum pressure for liquid density could not be found")
            
    # A flag value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas

    # Be sure guess in pressure is larger than lower bound
    if Prange[1] <= Prange[0]:
        Prange[1] = Prange[0]*1.1
        if z ==0:
            ObjRange[1] == 0.

    flag_min = False
    p = Prange[1]
    Parray = [Prange[1]]
    ObjArray = [ObjRange[1]]
    for z in range(maxiter):

        phil, rhol, flagl = calc_phil(p, T, xi, eos, density_dict=density_dict)
        if any(np.isnan(phil)):
            logger.info("Liquid fugacity coefficient should not be NaN, pressure could be too high.")
            flag_liqu = True
            Prange[1] = p
            ObjRange[1] = obj
            p = (Prange[1]-Prange[0])/2.0 + Prange[0]
            continue
            
        yi_range, phiv_max, flagv_max = solve_yi_xiT(yi_range, xi, phil, p, T, eos, density_dict=density_dict, **mole_fraction_options)
        obj = np.nansum(xi * phil / phiv_max) - 1.0

        if (flagv_max not in [0,2,4] or np.any(np.isnan(yi_range))):
            flag_liqu = True
            Prange[1] = p
            ObjRange[1] = obj
            logger.info("New Max Pressure: {} isn't vapor, flag={}, Obj Func: {}, Range {}".format(Prange[1],flagv_max,ObjRange[1],Prange))
            p = (Prange[1]-Prange[0])/2.0 + Prange[0]
        elif np.sum(np.abs(xi-yi_range)/xi) < 0.01: # If less than 2%
            flag_liqu = True
            ObjRange[1] = obj
            Prange[1] = p
            logger.info("Estimated Maximum Pressure Reproduces xi: {},  Obj. Func: {}".format(p,obj))
            p = (Prange[1]-Prange[0])/2.0 + Prange[0]
        elif obj < 0:
            if Prange[1] < p:
                Prange[0] = Prange[1]
                ObjRange[0] = ObjRange[1]
            Prange[1] = p
            ObjRange[1] = obj
            logger.info("New Max Pressure: {}, flag={}, Obj Func: {}, Range {}".format(Prange[1],flagv_max,ObjRange[1],Prange))
            logger.info("Got the pressure range!")
            slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
            intercept = ObjRange[1] - slope * Prange[1]
            Pguess = -intercept / slope
            flag_min = False
            break
        else:
            Parray.append(p)
            ObjArray.append(obj)
            if (z > 0 and ObjArray[-1] > 1.1*ObjArray[-2]) or flag_min:
                if not flag_min:
                    flag_min = True

                if np.any(np.abs(Parray[-1]-np.array(Parray[:-1])) < tol):
                    break
                else:
                    if obj > ObjRange[0]:
                        Prange[1] = p
                        ObjRange[1] = obj 
                        p = (Prange[1]-Prange[0])/2 + Prange[0]
                    else:

                        ind = np.where(np.logical_and(Prange[0]<=Parray, Parray<=Prange[1]))[0]
                        if np.size(ind) > 3:
                            logger.info("Pressure Obj starts to increase, let's find a lower bound.") 
                            p_array = np.linspace(Prange[0],Prange[1],10)
                            obj_array = np.zeros(len(p_array))
                            for ii in range(len(p_array)):
                                obj_array[ii] = solve_P_xiT(p_array[ii], xi, T, eos, density_dict=density_dict)
                            spline = interpolate.UnivariateSpline(p_array, obj_array, k=4, s=0)
                            p_min = spline.derivative().roots()
                            if len(p_min) > 1:
                                p_min = p_min[0]
                            elif len(p_min) == 0:
                                logger.error("Could not find minimum in pressure range:\n    Pressure: {}\n    Obj Value: {}".format(p_array,obj_array))
                            obj  = solve_P_xiT(p_min, xi, T, eos, density_dict=density_dict)
                            Prange[1] = p_min
                            ObjRange[1] = obj 
                            logger.info("New Max Pressure: {}, Obj Func: {}, Range {}".format(Prange[1],ObjRange[1],Prange))
                            if obj < 0:
                                logger.info("Got the pressure range!")
                                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                                intercept = ObjRange[1] - slope * Prange[1]
                                Pguess = -intercept / slope
                                flag_min = False
                            break

                            #x, y = _clean_plot_data([Parray[i] for i in ind], [ObjArray[i] for i in ind])
                            #xroot = np.sort(np.roots(np.polyder(np.polyfit(x,y,2))))
                            #if np.size(xroot) == 0:
                            #    p = (Prange[1]-Prange[0])*np.random.rand(1)[0] + Prange[0]
                            #else:
                            #    p = xroot[0]
                        else:
                            p = (Prange[1]-Prange[0])*np.random.rand(1)[0] + Prange[0]
                        
                    logger.info("New Max Pressure: (increases obj) {},  Obj. Func: {}, Range {}".format(Prange[1],obj,Prange))
                    if p < Prange[0]:
                        p = (Prange[1]-Prange[0])*np.random.rand(1)[0] + Prange[0]
            elif flag_liqu:
                Prange[0] = p
                ObjRange[0] = obj
                p = (Prange[1]-Prange[0])/2.0 + Prange[0]
                logger.info("New Min Pressure: {},  Obj. Func: {}, Range {}".format(Prange[0],ObjRange[0],Prange))
            else:
                if Prange[1] < p:
                    Prange[0] = Prange[1]
                    ObjRange[0] = ObjRange[1]
                Prange[1] = p
                ObjRange[1] = obj
                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                intercept = ObjRange[1] - slope * Prange[1]
                p = np.nanmax([-intercept / slope, 2*Prange[1]])
                logger.info("New Max Pressure: {},  Obj. Func: {}, Range {}".format(Prange[1],ObjRange[1],Prange))

    if (z == maxiter-1 or flag_min):
        if flag_min:
            logger.error("Cannot reach objective value of zero. Final Pressure: {}, Obj. Func: {}".format(p,obj))
        else:
            logger.error('Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress')
        Prange = np.array([np.nan, np.nan])
        Pguess = np.nan
    else:
        logger.info("[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange),str(ObjRange)))
        logger.info("Initial guess in pressure: {} Pa".format(Pguess))

        yi_global = yi_range

    return Prange, Pguess

######################################################################
#                                                                    #
#                              Calc P range                          #
#                                                                    #
######################################################################
def calc_Prange_yi(T, xi, yi, eos, density_dict={}, mole_fraction_options={}):
    r"""
    Obtain min and max pressure values.

    The vapor mole fraction is set and the objective function at each of those values is of opposite sign.
    
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
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    Prange : list
        List of min and max pressure range
    """

    global xi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = PvsRho(T, yi, eos, **density_dict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    # Calculation the highest pressure possible
    if len(extrema) > 1:
        Pmin = min(Pvspline(extrema))
        Pmax = max(Pvspline(extrema))
    else:
        Pmin = min(Plist)
        if Pmin < 0.:
            Pmin = 100
        Pmax = 10*Pmin
    Prange = [Pmin, Pmax]
    ObjRange = [0, 0]
    xi_range = xi

    flag_sol = False
    flag_vapor = False
    p = Prange[0]

    maxiter = 200
    for i in range(maxiter):
        # Calculate objective value
        phiv, rhov, flagv = calc_phiv(p, T, yi, eos, density_dict=density_dict)
        xi_range, phil, flagl = solve_xi_yiT(xi_range, yi, phiv, p, T, eos, density_dict=density_dict, **mole_fraction_options)
        obj = (np.nansum(yi * phiv / phil) - 1.0)

        if i == 0:
            if flagv not in [0,2,4]:
                logger.warning('A vapor is not produced at the minimum pressure.')
                Prange = np.array([np.nan, np.nan])
                Pguess = np.nan
                break
            else:
                ObjRange[0] = obj
                logger.info("Estimate Minimum pressure: {},  Obj. Func: {}".format(p,ObjRange[0]))
                p = Prange[1]
        else:
            if i == 1:
                ObjRange[1] = obj
                logger.info("Estimate Maximum pressure: {},  Obj. Func: {}".format(p,ObjRange[1]))

            if flagv not in [0,2,4]: # Ensure vapor is produced
                flag_vapor = True
                Prange[1] = p
                ObjRange[1] = p
                logger.info("New Max Pressure: {} doesn't produce vapor, flag={}, Obj Func: {}, Range {}".format(Prange[1],flagv,ObjRange[1],Prange))
                p = (Prange[1]-Prange[0])/2.0 + Prange[0]
            elif obj > 0:  # Check pressure range
                if Prange[1] < p:
                    Prange[0] = Prange[1]
                    ObjRange[0] = ObjRange[1]
                Prange[1] = p
                ObjRange[1] = obj
                logger.info("New Max Pressure: {}, flag={}, Obj Func: {}, Range {}".format(Prange[1],flagv,ObjRange[1],Prange))
                logger.info("Got the pressure range!")
                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                intercept = ObjRange[1] - slope * Prange[1]
                Pguess = -intercept / slope
                flag_sol = True
                break
            elif flag_vapor:
                Prange[0] = p
                ObjRange[0] = obj
                p = (Prange[1]-Prange[0])/2.0 + Prange[0]
                logger.info("New Max Pressure: {},  Obj. Func: {}, Range {}".format(Prange[0],ObjRange[0],Prange))
            else:
                if Prange[1] < p:
                    Prange[0] = Prange[1]
                    ObjRange[0] = ObjRange[1]
                Prange[1] = p
                ObjRange[1] = obj
                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                intercept = ObjRange[1] - slope * Prange[1]
                p = np.nanmax([-intercept / slope, 2*Prange[1]])
                logger.info("New Max Pressure: {},  Obj. Func: {}, Range {}".format(Prange[1],ObjRange[1],Prange))
            
    if i == maxiter-1:
        logger.error('Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress')
        Prange = np.array([np.nan, np.nan])
        Pguess = np.nan
    elif flag_sol:
        logger.info("[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange),str(ObjRange)))
        logger.info("Initial guess in pressure: {} Pa".format(Pguess))
    else:
        logger.error('Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress')

        xi_global = xi_range

    return Prange, Pguess


######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_yi_xiT(yi, xi, phil, P, T, eos, density_dict={}, maxiter=50, tol=1e-6):
    r"""
    Find vapor mole fraction given pressure, liquid mole fraction, and temperature.

    Objective function is the sum of the predicted "mole numbers" predicted by the computed fugacity coefficients. Note that by "mole number" we mean that the prediction will only sum to 1 when the correct pressure is chosen in the outer loop. In this inner loop, we seek to find a mole fraction that is converged to reproduce itself in a prediction. If it hasn't, the new "mole numbers" are normalized into mole fractions and used as the next guess.
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
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
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

    global yi_global

    yi_total = [np.sum(yi)]
    yi /= np.sum(yi)
    flag_check_vapor = True # Make sure we only search for vapor compositions once
    flag_trivial_sol = True # Make sure we only try to find alternative to trivial solution once
    logger.info("    Solve yi: P {}, T {}, xi {}, phil {}".format(P, T, xi, phil))

    for z in range(maxiter):

        yi_tmp = yi/np.sum(yi)

        # Try yi
        phiv, rhov, flagv = calc_phiv(P, T, yi_tmp, eos, density_dict=density_dict)

        if ((any(np.isnan(phiv)) or flagv==1) and flag_check_vapor): # If vapor density doesn't exist
            flag_check_vapor = False
            if (all(yi_tmp != 0.) and len(yi_tmp)==2):
                logger.info("    Composition doesn't produce a vapor, let's find one!")
                yi_tmp = find_new_yi(P, T, phil, xi, eos, density_dict=density_dict)
                flag_trivial_sol = False
                if np.any(np.isnan(yi_tmp)):
                    phiv, rhov, flagv = [np.nan, np.nan, 3]
                    yinew = yi_tmp
                    break
                else:
                    phiv, rhov, flagv = calc_phiv(P, T, yi_tmp, eos, density_dict=density_dict)
                    yinew = xi * phil / phiv
            else:
                logger.info("    Composition doesn't produce a vapor, we need a function to search compositions for more than two components.")
                yinew = yi
        elif (np.sum(np.abs(xi-yi_tmp)/xi) < 0.05 and flag_trivial_sol):
            flag_trivial_sol = False
            if (all(yi_tmp != 0.) and len(yi_tmp)==2):
                logger.info("    Composition produces trivial solution, let's find a different one!")
                yi_tmp = find_new_yi(P, T, phil, xi, eos, density_dict=density_dict)
                flag_check_vapor = False
            else:
                logger.info("    Composition produces trivial solution, using random guess to reset")
                yi_tmp = np.random.rand(len(yi_tmp))
                yi_tmp /= np.sum(yi_tmp)

            if np.any(np.isnan(yi_tmp)):
                phiv, rhov, flagv = [np.nan, np.nan, 3]
                yinew = yi_tmp
                break
            else:
                phiv, rhov, flagv = calc_phiv(P, T, yi_tmp, eos, density_dict=density_dict)
                yinew = xi * phil / phiv
        else:
            yinew = xi * phil / phiv

        yinew[np.isnan(yinew)] = 0.
        yi2 =  yinew/np.sum(yinew)
        phiv2, _, flagv2 = calc_phiv(P, T, yi2, eos, density_dict=density_dict)

        if any(np.isnan(phiv)):
            phiv = np.nan
            logger.error("Fugacity coefficient of vapor should not be NaN, pressure could be too high.")

        # Check for bouncing between values
        if len(yi_total) > 3:
            tmp1 =  (np.abs(np.sum(yinew)-yi_total[-2]) + np.abs(yi_total[-1]-yi_total[-3]))
            if (tmp1 < np.abs(np.sum(yinew)-yi_total[-1]) and flagv != flagv2):
                logger.info("    Composition bouncing between values, let's find the answer!")
                bounds = np.sort([yi_tmp[0], yi2[0]])
                yi2, obj = bracket_bounding_yi(P, T, phil, xi, eos, bounds=bounds, density_dict=density_dict)
                phiv2, _, flagv2 = calc_phiv(P, T, yi2, eos, density_dict=density_dict)
                yi_global = yi2
                logger.info("    Inner Loop Final (from bracketing bouncing values) yi: {}, Final Error on Smallest Fraction: {}".format(yi2,obj))
                break

        logger.debug("    yi guess {}, yi calc {}, phiv {}, flag {}".format(yi_tmp,yinew,phiv,flagv))
        logger.debug("    Old yi_total: {}, New yi_total: {}, Change: {}".format(yi_total[-1],np.sum(yinew),np.sum(yinew)-yi_total[-1])) 

        # Check convergence
        if abs(np.sum(yinew)-yi_total[-1]) < tol:
            ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp>0]))[0] 
            if np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp] < tol:
                yi_global = yi2
                logger.info("    Inner Loop Final yi: {}, Final Error on Smallest Fraction: {}%".format(yi2,np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp]*100))
                break

        if z < maxiter-1:
            yi_total.append(np.sum(yinew))
            yi = yinew

    ## If yi wasn't found in defined number of iterations
    ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp>0.]))[0]
    if flagv==3:
        yi2 = yinew/np.sum(yinew)
        logger.info("    Could not converged mole fraction")
        phiv2 = np.full(len(yi_tmp), np.nan)
        flagv2 = np.nan
    elif z == maxiter - 1:
        yi2 = yinew/np.sum(yinew)
        tmp = (np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp])
        logger.warning('    More than {} iterations needed. Error in Smallest Fraction: {}%'.format(maxiter, tmp*100))
        if tmp > .1: # If difference is greater than 10%
            yinew = find_new_yi(P, T, phil, xi, eos, density_dict=density_dict)
            yi2 = yinew/np.sum(yinew)
        y1 = spo.least_squares(obj_yi, yi2[0], bounds=(0.,1.), args=(P, T, phil, xi, eos, density_dict))
        yi = y1.x[0]
        yi2 = np.array([yi,1-yi])
        phiv, rhov, flagv = calc_phiv(P, T, yi2, eos, density_dict=density_dict)
        obj = obj_yi(yi2, P, T, phil, xi, eos, density_dict=density_dict)
        logger.warning('    Find yi with root algorithm, yi {}, obj {}'.format(yi2,obj))
        if obj > tol:
            logger.error("Could not converge mole fraction")
            phiv = np.full(len(yi_tmp),np.nan)
            flagv = 3

    return yi2, phiv2, flagv2

######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_xi_yiT(xi, yi, phiv, P, T, eos, density_dict={}, maxiter=20, tol=1e-6):
    r"""
    Find liquid mole fraction given pressure, vapor mole fraction, and temperature. 

    Objective function is the sum of the predicted "mole numbers" predicted by the computed fugacity coefficients. Note that by "mole number" we mean that the prediction will only sum to 1 when the correct pressure is chosen in the outer loop. In this inner loop, we seek to find a mole fraction that is converged to reproduce itself in a prediction. If it hasn't, the new "mole numbers" are normalized into mole fractions and used as the next guess.
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
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
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

    global xi_global

    xi /= np.sum(xi)
    xi_total = [np.sum(xi)]
    logger.info("    Solve xi: P {}, T {}, yi {}, phiv {}".format(P, T, yi, phiv))
    for z in range(maxiter):

        xi_tmp = xi/np.sum(xi)

        # Try xi
        phil, rhol, flagl = calc_phil(P, T, xi_tmp, eos, density_dict=density_dict)

        if (any(np.isnan(phil)) or flagl==0): # If liquid density doesn't exist
            logger.info("    Composition doesn't produce a vapor, let's find one!")
            if all(xi != 0.):
                xinew = find_new_xi(P, T, phiv, yi, eos, density_dict=density_dict)
                phil, rhol, flagl = calc_phil(P, T, xinew, eos, density_dict=density_dict)
                xinew = yi * phiv / phil
            else:
                xinew = xi
            
            if any(np.isnan(phil)):
                phil = np.nan
                logger.error("Fugacity coefficient of liquid should not be NaN")
        else:
            xinew = yi * phiv / phil
        xinew[np.isnan(xinew)] = 0.

        logger.info("    xi guess {}, xi calc {}, phil {}".format(xi_tmp,xinew,phil))
        logger.info("    Old xi_total: {}, New xi_total: {}, Change: {}".format(xi_total[-1],np.sum(xinew),np.sum(xinew)-xi_total[-1]))

        # Check convergence
        if abs(np.sum(xinew)-xi_total[-1]) < tol:
            ind_tmp = np.where(xi_tmp == min(xi_tmp[xi_tmp>0]))[0]
            xi2 = xinew/np.sum(xinew)
            if np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp] < tol:
                xi_global = xi_tmp
                logger.info("    Found xi")
                break

        if z < maxiter-1:
            xi_total.append(np.sum(xinew))
            xi = xinew

    ## If xi wasn't found in defined number of iterations
    xinew /= np.sum(xinew)

    ind_tmp = np.where(xi == min(xi[xi>0]))[0]
    if z == maxiter - 1:
        xi2 = xinew/np.sum(xinew)
        tmp = (np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp])
        logger.warning('    More than {} iterations needed. Error in Smallest Fraction: {} %%'.format(maxiter, tmp*100))
        if tmp > .1: # If difference is greater than 10%
            xinew = find_new_xi(P, T, phiv, yi, eos, density_dict=density_dict)
        xinew = spo.least_squares(obj_xi, xinew[0], bounds=(0.,1.), args=(P, T, phiv, yi, eos, density_dict))
        xi = xinew.x[0]
        xi = np.array([xi,1-xi])
        obj = obj_xi(xi, P, T, phiv, yi, eos, density_dict=density_dict)
        logger.warning('    Find xi with root algorithm, xi {}, obj {}'.format(xi,obj))
    else:
        logger.info("    Inner Loop Final xi: {}, Final Error on Smallest Fraction: {}".format(xi_tmp,np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp]*100))

    return xi_tmp, phil, flagl

######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def find_new_yi(P, T, phil, xi, eos, bounds=(0.01, 0.99), npoints=30, density_dict={}):
    r"""
    Search vapor mole fraction combinations for a new estimate that produces a vapor density.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    phil : float
        Fugacity coefficient of liquid at system pressure
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    bounds : tuple, Optional, default: (0.01, 0.99)
        These bounds dictate the lower and upper boundary for the first component in a binary system.
    npoints : float, Optional, default: 30
        Number of points to test between the bounds.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """


    yi_ext = np.linspace(bounds[0],bounds[1],npoints) # Guess for yi
    obj_ext = np.zeros(len(yi_ext))
    flag_ext = np.zeros(len(yi_ext))

    for i, yi in enumerate(yi_ext):
        yi = np.array([yi, 1-yi])
        obj, flagv = obj_yi(yi, P, T, phil, xi, eos, density_dict=density_dict, return_flag=True)
        flag_ext[i] = flagv
        obj_ext[i] = obj        

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    logger.debug("    Number of valid mole fractions: {}".format(tmp))
    if tmp == 0:
        yi_tmp = np.nan
        obj_tmp = np.nan
    else:
        # Remove any NaN
        obj_tmp  =  obj_ext[~np.isnan(obj_ext)]
        yi_tmp   =   yi_ext[~np.isnan(obj_ext)]
        flag_tmp = flag_ext[~np.isnan(obj_ext)]
 
        # Assess vapor values
        # A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        #ind = [i for i in range(len(flag_tmp)) if flag_tmp[i] not in [1,3]]
        #if ind:
        #    obj_tmp = [obj_tmp[i] for i in ind]
        #    yi_tmp = [yi_tmp[i] for i in ind]

        # Fit spline
        spline = interpolate.UnivariateSpline(yi_tmp, obj_tmp, k=4, s=0)
        yi_min = spline.derivative().roots()

        if len(yi_min) > 1:
            yi_concav = spline.derivative(n=2)(yi_min)
            yi_min = [yi_min[i] for i in range(len(yi_min)) if yi_concav[i]>0.0]
            if len(yi_tmp) > 1:
                if obj_tmp[0] < obj_tmp[-1]:
                    yi_min.insert(0,yi_tmp[0])
                elif obj_tmp[-1] < obj_tmp[-2]:
                    yi_min.append(yi_tmp[-1])
            yi_min = np.array(yi_min)
            obj_trivial = np.abs(yi_min-xi[0])/xi[0]
            ind = np.where(obj_trivial==min(obj_trivial))[0][0]
            logger.debug('    Found multiple minima: {}, discard {} as trivial solution'.format(yi_min, yi_min[ind]))
            yi_min = np.array([yi_min[ii] for ii in range(len(yi_min)) if ii != ind])
            if len(yi_min) > 1:
                lyi = len(yi_min)
                obj_tmp2 = np.zeros(lyi)
                flagv_tmp2 = np.zeros(lyi)
                for ii in range(lyi):
                    obj_tmp2[ii], flagv_tmp2[ii] = obj_yi(yi_min[ii], P, T, phil, xi, eos, density_dict=density_dict, return_flag=True)
                yi_tmp2 = [yi_min[ii] for ii in range(len(yi_min)) if flagv_tmp2 [ii] != 1]
                if len(yi_tmp2):
                    obj_tmp2 =  [obj_tmp2[ii] for ii in range(len(obj_tmp2)) if flagv_tmp2 [ii] != 1]
                    yi_min = [yi_tmp2[np.where(obj_tmp2==min(obj_tmp2))[0][0]]]
                else:
                    yi_min = [yi_min[np.where(obj_tmp2==min(obj_tmp2))[0][0]]]

        if not len(yi_min):
            # Choose values with lowest objective function
            ind = np.where(np.abs(obj_tmp)==min(np.abs(obj_tmp)))[0][0]
            obj_final = obj_tmp[ind]
            yi_final = yi_tmp[ind]
        else:
            yi_final = yi_min[0]
            obj_final = spline(yi_min[0])
        
    logger.info("    Found new guess in yi: {}, Obj: {}".format(yi_final,obj_final))
    if type(yi_final) not in [list,np.ndarray]:
        yi_final = np.array([yi_final, 1-yi_final])

    return yi_final

######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def bracket_bounding_yi(P, T, phil, xi, eos, bounds=(0.01, 0.99), maxiter=50, tol=1e-7, density_dict={}):
    r"""
    Search vapor mole fraction combinations for a new estimate that produces a vapor density.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    phil : float
        Fugacity coefficient of liquid at system pressure
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    bounds : tuple, Optional, default: (0.01, 0.99)
        These bounds dictate the lower and upper boundary for the first component in a binary system.
    maxiter : int, Optional, default: 50
        Maximum number of iterations
    tol : float, Optional, default: 1e-7
        Tolerance to quit search for yi
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """

    if np.size(bounds) != 2:
        raise ValueError("Given bounds on y1 must be of length two.")

    bounds = np.array(bounds)
    obj_bounds = np.zeros(2)
    flag_bounds = np.zeros(2)
    obj_bounds[0], flag_bounds[0] = obj_yi(bounds[0], P, T, phil, xi, eos, density_dict=density_dict, return_flag=True)
    obj_bounds[1], flag_bounds[1] = obj_yi(bounds[1], P, T, phil, xi, eos, density_dict=density_dict, return_flag=True)

    if flag_bounds[0] == flag_bounds[1]:
        logger.error("    Both mole fractions have flag, {}, continue seeking convergence".format(flag_bounds[0]))
        y1 = bounds[1]
        flagv = flag_bounds[1]
        i = maxiter - 1

    else:
        flag_high_vapor = False
        for i in np.arange(maxiter):

            y1 = np.mean(bounds)
            obj, flagv = obj_yi(y1, P, T, phil, xi, eos, density_dict=density_dict, return_flag=True)

            if not flag_high_vapor:
                ind = np.where(flag_bounds==flagv)[0][0]
                if flagv == 0 and obj > 1/tol:
                    flag_high_vapor = True
                    bounds[0], obj_bounds[0], flag_bounds[0] = bounds[ind], obj_bounds[ind], flag_bounds[ind]
                    ind = 1
            else:
                if obj < obj_bounds[0]:
                    ind = 0
                else:
                    ind = 1 

            bounds[ind], obj_bounds[ind], flag_bounds[ind] = y1, obj, flagv
            logger.debug("    Bouncing mole fraction new bounds: {}, obj: {}, flag: {}".format(bounds,obj_bounds,flag_bounds))
             
            # Check convergence
            if np.abs(bounds[1]-bounds[0]) < tol:
                break

    ind_array = np.where(flag_bounds==0)[0]
    if np.size(ind_array) == 1:
        ind = ind_array[0]
    else:
        ind = np.where(obj_bounds==np.min(obj_bounds))[0][0]  

    y1, flagv = bounds[ind], flag_bounds[ind]
    if i == maxiter - 1:
        logger.info("    Bouncing mole fraction, max iterations ended with, y1={}, flagv={}".format(y1,flagv))
    else:
        logger.info("    Bouncing mole fractions converged to y1={}, flagv={}".format(y1,flagv))
        

    return np.array([y1, 1-y1]), flagv

######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def obj_yi(yi, P, T, phil, xi, eos, density_dict={}, return_flag=False):
    r"""
    Objective function for solving for stable vapor mole fraction.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    P : float
        [Pa] Pressure of the system 
    T : float
        [K] Temperature of the system
    phil : float
        Fugacity coefficient of liquid at system pressure
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    obj : numpy.ndarray
        Objective function for solving for vapor mole fractions
    """

    if type(yi) == float or np.size(yi) == 1:
        if type(yi) in [list, np.ndarray]:
            yi = np.array([yi[0], 1-yi[0]])
        else:
            yi = np.array([yi, 1-yi])
    elif type(yi) == list:
        yi = np.array(yi)
    yi /= np.sum(yi)

    phiv, _, flagv = calc_phiv(P, T, yi, eos, density_dict=density_dict)

    tmp1 = yi*phiv
    #tmp2 = xi*phil
    yinew = xi * phil / phiv
    yi2 = yinew/np.sum(yinew)
    phiv2, _, flagv2 = calc_phiv(P, T, yi2, eos, density_dict=density_dict)
    tmp2 = yi2*phiv2
    obj = np.sum(np.abs(yinew - xi * phil / phiv2))
    logger.debug("    Guess yi: {}, calc yi: {}, diff={}, flagv {}".format(yi,yi2,obj,flagv))
    
    if return_flag:
        return obj, flagv
    else:
        return obj

######################################################################
#                                                                    #
#                       Find new Xi                                  #
#                                                                    #
######################################################################


def find_new_xi(P, T, phiv, yi, eos, density_dict={}):
    r"""
    Search liquid mole fraction combinations for a new estimate that produces a liquid density.
        
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    phiv : float
        Fugacity coefficient of vapor at system pressure
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole
        
    Returns
    -------
    xi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """
    
    xi_ext = np.linspace(0.001,.999,30) # Guess for yi
    obj_ext = []
    flag_ext = [[],[]]
    
    for xi in xi_ext:
        xi = np.array([xi, 1-xi])
        ####
        phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)
        xinew = yi * phiv / phil
        xinew_total_1 = np.sum(xinew)
        
        xi2 = xinew/xinew_total_1
        phil2, rhol2, flagl2 = calc_phil(P, T, xi2, eos, density_dict=density_dict)
        xinew = yi * phiv / phil2
        xinew_total_2 = np.sum(xinew)
        
        logger.debug("    xi_totals {} {}".format(xinew_total_1,xinew_total_2))
        
        obj = xinew_total_1 - xinew_total_2
        
        flag_ext[0].append(flagl)
        flag_ext[1].append(flagl2)
        ######
        #    obj = obj_yi(yi, P, T, phil, xi, eos, density_dict=density_dict)
        obj_ext.append(abs(obj))
        logger.debug("    Obj xi {} total1 - total2 = {}".format(xi,obj))

    obj_ext = np.array(obj_ext)
    flag_ext = np.array(flag_ext)

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    logger.debug("    Number of valid mole fractions: {}".format(tmp))
    if tmp == 0:
        xi_tmp = np.nan
        obj_tmp = np.nan
    else:
        # Remove any NaN
        obj_tmp  =  obj_ext[~np.isnan(obj_ext)]
        xi_tmp   =   xi_ext[~np.isnan(obj_ext)]
        flag_tmp = flag_ext[1][~np.isnan(obj_ext)]
    
        # Assess vapor values
        # A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        ind = [i for i in range(len(flag_tmp)) if flag_tmp[i] not in [0,3,4]]
        if ind:
            obj_tmp = [obj_tmp[i] for i in ind]
            xi_tmp = [xi_tmp[i] for i in ind]
    
        # Choose values with lowest objective function
        ind = np.where(np.abs(obj_tmp)==min(np.abs(obj_tmp)))[0][0]
        obj_tmp = obj_tmp[ind]
        xi_tmp = xi_tmp[ind]
    
    logger.info("    Found new guess in xi: {}, Obj: {}".format(xi_tmp,obj_tmp))

    if type(xi_tmp) not in [list,np.ndarray]:
        xi_tmp = np.array([xi_tmp, 1-xi_tmp])
    
    return xi_tmp

######################################################################
#                                                                    #
#                       Find new Xi                                  #
#                                                                    #
######################################################################


def obj_xi(xi, P, T, phiv, yi, eos, density_dict={}):
    r"""
    Objective function for solving for stable vapor mole fraction.
        
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    phiv : float
        Fugacity coefficient of vapor at system pressure
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole
        
    Returns
    -------
    obj : numpy.ndarray
        Objective function for solving for liquid mole fractions
    """
    
    if type(xi) == float or len(xi) == 1:
        if type(xi) in [list, np.ndarray]:
            xi = np.array([xi[0], 1-xi[0]])
        else:
            xi = np.array([xi, 1-xi])

    phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)
    xinew = yi * phiv / phil
    xinew_total_1 = np.sum(xinew)
    
    xi2 = xinew/xinew_total_1
    phil2, rhol2, flagl2 = calc_phil(P, T, xi2, eos, density_dict=density_dict)
    xinew = yi * phiv / phil2
    xinew_total_2 = np.sum(xinew)
    
    logger.debug("    xi_totals {} {}".format(xinew_total_1,xinew_total_2))
    
    obj = np.abs(xinew_total_1 - xinew_total_2)
    
    return obj

######################################################################
#                                                                    #
#                              Solve P xT                            #
#                                                                    #
######################################################################
def solve_P_xiT(P, xi, T, eos, density_dict={}, mole_fraction_options={}):
    r"""
    Objective function used to search pressure values and solve outer loop of P bubble point calculations.
    
    Parameters
    ----------
    P : float
        [Pa] Guess in pressure of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    

    Returns
    -------
    obj_value : float
        :math:`\sum\frac{x_{i}\{phi_l}{\phi_v}-1`
    """

    global yi_global

    if P < 0:
        return 10.0

    logger.info("P Guess: {} Pa".format(P))

    #find liquid density
    phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)

    yinew, phiv, flagv = solve_yi_xiT(yi_global, xi, phil, P, T, eos, density_dict=density_dict, **mole_fraction_options)
    yi_global = yinew / np.sum(yinew)

    #given final yi recompute
    phiv, rhov, flagv = calc_phiv(P, T, yi_global, eos, density_dict=density_dict)

    Pv_test = eos.pressure(rhov, T, yi_global)
    obj_value = float((np.nansum(xi * phil / phiv) - 1.0))
    logger.info('Obj Func: {}, Pset: {}, Pcalc: {}'.format(obj_value, P, Pv_test[0]))

    return obj_value

######################################################################
#                                                                    #
#                              Solve P yT                            #
#                                                                    #
######################################################################
def solve_P_yiT(P, yi, T, eos, density_dict={}, mole_fraction_options={}):
    r"""
    Objective function used to search pressure values and solve outer loop of P dew point calculations.
    
    Parameters
    ----------
    P : float
        [Pa] Guess in pressure of the system
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    obj_value : list
        :math:`\sum\frac{y_{i}\{phi_v}{\phi_l}-1`
    """

    global xi_global

    if P < 0:
        return 10.0

    #find liquid density
    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, density_dict=density_dict)

    xinew, phil, flagl = solve_xi_yiT(xi_global, yi, phiv, P, T, eos, density_dict=density_dict, **mole_fraction_options)
    xi_global = xinew / np.sum(xinew)

    #given final yi recompute
    phil, rhol, flagl = calc_phil(P, T, xi_global, eos, density_dict=density_dict)

    Pv_test = eos.pressure(rhol, T, xi_global)
    obj_value = float((np.nansum(yi * phiv / phil) - 1.0)) 
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
        [Pa] Dummy value of saturation pressure
    NaNbead : str
        Bead name of the component that is above it's critical point
    """

    for j in range(np.size(eos.nui[ind])):
        if eos.nui[ind][j] > 0.0 and eos.beads[j] == "CO2":
            Psat = 10377000.0
        elif eos.nui[ind][j] > 0.0 and eos.beads[j] == "N2":
            Psat = 7377000.0
        elif eos.nui[ind][j] > 0.0 and ("CH4" in eos.beads[j]):
            Psat = 6377000.0
        elif eos.nui[ind][j] > 0.0 and ("CH3CH3" in eos.beads[j]):
            Psat = 7377000.0
        elif eos.nui[ind][j] > 0.0:
            #Psat = np.nan
            Psat = 7377000.0
            NaNbead = eos.beads[j]
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
def calc_yT_phase(yi, T, eos, density_dict={}, mole_fraction_options={}, Pguess=-1, method="hybr", pressure_options={}):
    r"""
    Calculate dew point mole fraction and pressure given system vapor mole fraction and temperature.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default: -1
        [Pa] Guess the system pressure at the dew point. A negative value will force an estimation based on the saturation pressure of each component.
    method : str, Optional, default: "broyden1"
        Choose the method used to solve the dew point calculation
    pressure_options : dict, Optional, default: {}
        Options used in the given method, "method", to solve the outer loop in the solving algorithm

    Returns
    -------
    P : float
        [Pa] Pressure of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    flagl : int
        Flag identifying the fluid type for the liquid mole fractions, expected is liquid, 1. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    flagv : int
        Flag identifying the fluid type for the vapor mole fractions, expected is vapor or 0. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    obj : float
        Objective function value
    """

    global xi_global

    # Estimate pure component vapor pressures
    Psat = np.zeros_like(yi)
    for i in range(np.size(yi)):
        yi_tmp = np.zeros_like(yi)
        yi_tmp[i] = 1.0
        Psat[i], _, _ = calc_Psat(T, yi_tmp, eos, density_dict)
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

    Prange, Pguess = calc_Prange_yi(T, xi, yi, eos, density_dict, mole_fraction_options=mole_fraction_options)
    P = Pguess

    P = gtb.solve_root(solve_P_yiT, args=(yi, T, eos, density_dict, mole_fraction_options), x0=P, method=method, bounds=Prange, options=pressure_options)

    #find vapor density and fugacity
    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, density_dict=density_dict)
    phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)
    if "tol" in mole_fraction_options:
        if mole_fraction_options["tol"] > 1e-10:
            mole_fraction_options["tol"] = 1e-10

    obj = solve_P_yiT(P, yi, T, eos, density_dict=density_dict, mole_fraction_options=mole_fraction_options)

    logger.info("Final Output: Obj {}, P {} Pa, flagl {}, xi {}".format(obj,P,flagl,xi_global))

    return P, xi, flagl, flagv, obj

######################################################################
#                                                                    #
#                              Calc xT phase                         #
#                                                                    #
######################################################################
def calc_xT_phase(xi, T, eos, density_dict={}, mole_fraction_options={}, Pguess=None, Pmin=1e+4, method="bisect", pressure_options={}):
    r"""
    Calculate bubble point mole fraction and pressure given system liquid mole fraction and temperature.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        [K] Temperature of the system
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    mole_fraction_options : dict, Optional, default: {}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default: None
        [Pa] Guess the system pressure at the dew point. A value of None will force an estimation based on the saturation pressure of each component.
    Pmin : float, Optional, default: 10000
        [Pa] Guess the minimum system pressure at the dew point. A value of None will used the calc_Prange_xi default.
    method : str, Optional, default: "bisect"
        Choose the method used to solve the dew point calculation
    pressure_options : dict, Optional, default: {}
        Options used in the given method, "method", to solve the outer loop in the solving algorithm

    Returns
    -------
    P : float
        [Pa] Pressure of the system
    yi : numpy.ndarray
        Mole fraction of each component, sum(yi) should equal 1.0
    flagv : int
        Flag identifying the fluid type for the vapor mole fractions, expected is vapor or 0. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    flagl : int
        Flag identifying the fluid type for the liquid mole fractions, expected is liquid, 1. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    obj : float
        Objective function value
    """

    global yi_global

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], _, _ = calc_Psat(T, xi_tmp, eos, density_dict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                logger.error("Component, {}, is beyond it's critical point. Add an exception to setPsat".format(NaNbead))

    # Estimate initial pressure
    if Pguess is None:
        P=1.0/np.sum(xi/Psat)
    else:
        P = Pguess

    if ("yi_global" not in globals() or any(np.isnan(yi_global))):
        yi_global = xi * Psat / P
        yi_global /= np.nansum(yi_global)
        yi_global = copy.deepcopy(yi_global)
        logger.info("Guess yi in calc_xT_phase with Psat: {}".format(yi_global))
    yi = yi_global

    Prange, Pestimate = calc_Prange_xi(T, xi, yi, eos, density_dict, mole_fraction_options=mole_fraction_options, Pmin=Pmin)
    if np.any(np.isnan(Prange)):
        raise ValueError("Neither a suitable pressure range, or guess in pressure could be found nor was given.")
    else:
        if Pguess is not None:
            if Pguess > Prange[1] or Pguess < Prange[0]:
                 logger.warning("Given guess in pressure, {}, is outside of the identified pressure range, {}. Using estimated pressure, {}.".format(Pguess,Prange,Pestimate))
                 P = Pestimate
            else:
                 logger.warning("Using given guess in pressure, {}, that is inside identified pressure range.".format(Pguess,Prange,Pestimate))
                 P = Pguess
        else:
            P = Pestimate
        P = gtb.solve_root(solve_P_xiT, args=(xi, T, eos, density_dict, mole_fraction_options), x0=P, method=method, bounds=Prange, options=pressure_options)

    #find liquid density and fugacity
    phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)
    phiv, rhov, flagv = calc_phiv(P, T, yi, eos, density_dict=density_dict)
    if "tol" in mole_fraction_options:
        if mole_fraction_options["tol"] > 1e-10:
            mole_fraction_options["tol"] = 1e-10

    obj = solve_P_xiT(P, xi, T, eos, density_dict=density_dict, mole_fraction_options=mole_fraction_options)

    logger.info("Final Output: Obj {}, P {} Pa, flagv {}, yi {}".format(obj,P,flagv,yi_global))

    return P, yi_global, flagv, flagl, obj

######################################################################
#                                                                    #
#                              Calc xT phase                         #
#                                                                    #
######################################################################
def hildebrand_solubility(rhol, xi, T, eos, dT=.1, tol=1e-4, density_dict={}):
    r"""
    Calculate the solubility parameter based on temperature and composition. This function is based on the method used in Zeng, Z., Y. Xi, and Y. Li "Calculation of Solubility Parameter Using Perturbed-Chain SAFT and Cubic-Plus-Association Equations of State" Ind. Eng. Chem. Res. 2008, 47, 9663–9669.
    
    Parameters
    ----------
    rhol : float
        Liquid molar density [mol/m^3]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    dT : float
        Change in temperature used in calculating the derivative with central difference method 
    tol : float
        This cutoff value evaluates the extent to which the integrand of the calculation has decayed. If the last value if the array is greater than tol, then the remaining area is estimated as a triangle, where the intercept is estimated from an interpolation of the previous four points.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole

    Returns
    -------
    delta : float
        Solubility parameter [Pa^(1/2)], ratio of cohesive energy and molar volume
    """

    R = constants.Nav * constants.kb
    RT = T * R

    if type(rhol) in [np.ndarray,list]:
        logger.info("rhol should be a float, not {}".format(rhol))

    # Find dZdT
    vlist, Plist1 = PvsRho(T-dT, xi, eos, **density_dict, maxrho=rhol)
    vlist2, Plist2 = PvsRho(T+dT, xi, eos, **density_dict, maxrho=rhol)
    vlist, Plist = PvsRho(T, xi, eos, **density_dict, maxrho=rhol)
    if any(vlist != vlist2):
        logger.error("Dependant variable vectors must be identical.")

    int_tmp = (Plist2-Plist1)/(2*dT)/R - Plist/(RT)
    integrand_list = gaussian_filter1d(int_tmp, sigma=0.1)

    # Calculat U_res
    integrand_spline = interpolate.InterpolatedUnivariateSpline(vlist, integrand_list,ext=1)
    U_res = -RT*integrand_spline.integral(1/rhol,vlist[-1])

    # Check if function converged before taking integral, if not, correct area
    if integrand_list[-1] > tol:
        slope, yroot = np.polyfit(vlist[-4:], integrand_list[-4:], 1)
        xroot = -yroot/slope
        U_res += -RT*integrand_list[-1]*(xroot-vlist[-1])/2

    if (U_res) > 0.:
        raise ValueError("The solubility parameter can not be imaginary")
    else:
        delta = np.sqrt(-(U_res)*rhol)
        logger.info("When T={}, xi={}, delta={}".format(T,xi,delta))

    return delta

######################################################################
#                                                                    #
#                              Calc PT phase                         #
#                                                                    #
######################################################################
def calc_flash(P, T, eos, density_dict={}, maxiter=200, tol=1e-9):
    r"""
    Flash calculation of vapor and liquid mole fractions. Only binary systems are currently supported
    
    Parameters
    ----------
    P : numpy.ndarray
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 
    maxiter : int, Optional, default: 50
        Maximum number of iterations in updating Ki values
    tol : float, Optional, tol: 1e-09
        Tolerance to break loop. The error is defined as the absolute value of the summed difference in Ki values between iterations.

    Returns
    -------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    flagl : int
        Flag identifying the fluid type for the liquid mole fractions, expected is liquid, 1. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    flagv : int
        Flag identifying the fluid type for the vapor mole fractions, expected is vapor or 0. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    obj : float
        Objective function value
    """

    # Initialize Variables
    lx = len(eos.nui)

    if lx != 2:
        raise ValueError("Only binary systems are currently supported for flash calculations, {} were given.".format(lx))

    Psat, Ki0, xi, yi, phil, phiv = [np.zeros(lx) for i in np.arange(6)]

    # Calculate Psat and Ki
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], _, _ = calc_Psat(T, xi_tmp, eos, density_dict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                logger.error("Component, {}, is beyond it's critical point. Add an exception to setPsat, otherwise assumed 1".format(NaNbead))
        Ki0[i] = Psat[i]/P

    tmp = Ki0 - 1.0
    if np.abs(np.sum(tmp)) == np.sum(np.abs(tmp)):
        if tmp[0] > 0:
            Ki0[0] = np.sqrt(np.finfo(float).eps)
        else:
            Ki0[0] = Ki0[0] + 1.0

    Ki = np.copy(Ki0)
    err = 1
    flag_critical = 0
    for i in np.arange(maxiter):
        
        # Mole Fraction
        xi[0] = (1-Ki[1])/(Ki[0]-Ki[1])
        xi[1] = 1-xi[0]
        if any(xi<0.0):
            ind = np.where(xi<0.0)[0][0]
            xi[ind] = np.sqrt(np.finfo(float).eps)
            if ind == 0:
                xi[1] = 1-xi[0]
            elif ind == 0:
                xi[0] = 1-xi[1]
       
        yi = Ki*xi
        if np.sum(yi) != 1.0:
            if np.abs(np.sum(yi) != 1.0) < np.sqrt(np.finfo(float).eps):
                raise ValueError("Vapor mole fractions do not add up to 1. Ki {}, xi {} produces {} = {}".format(Ki, xi, yi, np.sum(yi)))
            else:
                yi /= np.sum(yi)

        # Fugacity Coefficients and New Ki values
        phil, rhol, flagl = calc_phil(P, T, xi, eos, density_dict=density_dict)
        phiv, rhov, flagv = calc_phiv(P, T, yi, eos, density_dict=density_dict)
        logger.info("        xi: {}, phil: {}".format(xi,phil))
        logger.info("        yi: {}, phiv: {}".format(yi,phiv))
        Kinew = phil/phiv

        err = np.sum(np.abs(Kinew-Ki))
        logger.info("  Guess {} Ki: {}, New Ki: {}, Error: {}".format(i,Ki,Kinew,err))

        # Check Objective function
        if np.all(np.abs(Ki-1.0) < 1e-6) and flag_critical < 2:
            eps = np.sqrt(np.finfo(float).eps)
            ind = 1-flag_critical
            if flag_critical == 0:
                Ki[ind] = eps
                Ki[flag_critical] = 1/eps
            else:
                Ki[ind] = 1/eps
                Ki[flag_critical] = eps
            flag_critical += 1
            logger.info("    Liquid and vapor mole fractions are equal, let search from Ki = {}".format(Ki))
        elif err < tol:
            ind = np.where(Ki == min(Ki[Ki>0]))[0][0]
            err = np.abs(Kinew[ind] - Ki[ind]) / Ki[ind]
            logger.info("    Percent Error on smallest Ki value: {}".format(err))
            if err < tol:
                logger.info("    Found Ki")
                break
            Kiprev = Ki
            Ki = Kinew
        else:
            Kiprev = Ki
            Ki = Kinew

    if i == maxiter - 1:
        ind = np.where(Kiprev == min(Kiprev[Kiprev>0]))[0][0]
        err = np.abs(Ki[ind] - Kiprev[ind]) / Kiprev[ind]
        logger.warning('    More than {} iterations needed. Remaining error, {}.'.format(maxiter,err))

    flag_switch = False
    if flagl in [0, 4] or flagv == 1:
        if flagl == 1 or flagv in [0, 4]:
            if xi[0] > yi[0]:
                flag_switch = True
        else:
            flag_switch = True

    if flag_switch:
        zi, flag = xi, flagl
        xi, flagl = yi, flagv
        yi, flagv = zi, flag

    logger.info("Final Output: Obj {}, xi {} flagl {}, yi {} flagv {}".format(err,xi,flagl,yi,flagv))

    return xi, flagl, yi, flagv, err

######################################################################
#                                                                    #
#                              Calc PT phase                         #
#                                                                    #
######################################################################
def _calc_PT_phase(xi, T, eos, density_dict={}):
    r"""
    *Not Complete*
    Calculate the PT phase diagram given liquid mole fraction and temperature
    
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    P : float
        Pressure of the system [Pa]
    yi : numpy.ndarray
        Mole fraction of each component, sum(yi) should equal 1.0
    """

    logger.error("The function, calc_PT_phase, is not yet available.")

  #  Psat = np.zeros_like(xi)
  #  for i in range(np.size(xi)):
  #      xi_tmp = np.zeros_like(xi)
  #      xi_tmp[i] = 1.0
  #      Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, density_dict)
  #      if np.isnan(Psat[i]):
  #          Psat[i], NaNbead = setPsat(i, eos)
  #          if np.isnan(Psat[i]):
  #              logger.error("Component, {}, is beyond it's critical point. Add an exception to setPsat".format(NaNbead))

  #  zi = np.array([0.5, 0.5])

  #  #estimate ki
  #  ki = Psat / P

  #  #estimate beta (not thermodynamic) vapor frac
  #  beta = (1.0 - np.sum(ki * zi)) / np.prod(ki - 1.0)


######################################################################
#                                                                    #
#                              Calc dadT                             #
#                                                                    #
######################################################################
def calc_dadT(rho, T, xi, eos, density_dict={}):
    r"""
    Calculate the derivative of the Helmholtz energy with respect to temperature, :math:`\frac{dA}{dT}`.
    
    Parameters
    ----------
    rho : numpy.ndarray
        [mol/:math:`m^3`] Density array. Length depends on values in density_dict
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_dict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole 

    Returns
    -------
    dadT : numpy.ndarray
        Array of derivative values of Helmholtz energy with respect to temperature
    """

    step = np.sqrt(np.finfo(float).eps) * T * 1000.0

    #computer rho+step and rho-step for better a bit better performance
    Ap = eos.helmholtz_energy(rho, T + step, xi)
    Am = eos.helmholtz_energy(rho, T - step, xi)

    return (Ap - Am) / (2.0 * step)

######################################################################
#                                                                    #
#                          EOS Fugacity Test 1                       #
#                                                                    #
######################################################################
def fugacity_test_1(P, T, xi, rho, eos, step_size=1e-5):
    r"""
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    rho : float
        [mol/:math:`m^3`] Density array. Length depends on values in density_dict
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.

    Returns
    -------
    Residual : float
        
    """

    if type(rho) not in [list, np.ndarray]:
        rho = np.array([rho])

    Z = P / (rho * T * constants.R)
    dP = P * step_size
    log_phi_1 = np.sum(xi*np.log(eos.fugacity_coefficient(P+dP, rho, xi, T)))
    log_phi_2 = np.sum(xi*np.log(eos.fugacity_coefficient(P-dP, rho, xi, T)))
    residual = (log_phi_1-log_phi_2)/(2*dP) - (Z-1)/P

    return residual

######################################################################
#                                                                    #
#                          EOS Fugacity Test 2                       #
#                                                                    #
######################################################################
def fugacity_test_2(P, T, xi, rho, eos, fractional_change=1e-1):
    r"""
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    rho : float
        [mol/:math:`m^3`] Density array. Length depends on values in density_dict
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.

    Returns
    -------
    Residual : float
        
    """

    ncomp = len(xi)
    if type(rho) not in [list, np.ndarray]:
        rho = np.array([rho])

#    drho = rho * step_size
#    log_phi_1 = np.log(eos.fugacity_coefficient(P, rho+drho, xi, T))
#    log_phi_2 = np.log(eos.fugacity_coefficient(P, rho-drho, xi, T))
#    residual = np.sum(xi*(log_phi_1-log_phi_2)/(2*drho))

    ind = np.where(xi>np.finfo("float").eps)[0]
    if len(ind) == 1:
        logger.error("Fugacity test two is for multicomponent systems.")
    elif len(ind) != ncomp:
        logger.info("There is not a significant amount of components {} in solution".format(np.setdiff1d(range(ncomp),ind)))

#    dy = step_size
#    dphi = np.zeros((2,ncomp))
#    for j, delta in enumerate((dy, -dy)):
#        y_tmp = np.copy(xi)
#        y_tmp[ind[0]] += delta
#        y_tmp[ind[-1]] -= delta
#        dphi[j,:] = np.log(eos.fugacity_coefficient(P, rho, y_tmp, T))
#    dphidx = (dphi[0] - dphi[1]) / (2.0 * dy)

    log_phi = np.zeros((2,ncomp))
    for i,factor in enumerate([1.0, (1-fractional_change)]):
        log_phi[i,:] = np.log(eos.fugacity_coefficient(P*factor, rho, xi, T))
    dphidx = log_phi[0] - log_phi[1]

    residual = np.sum(xi*dphidx)

    return residual

######################################################################
#                                                                    #
#                  Activity Coefficient                              #
#                                                                    #
######################################################################
def activity_coefficient(P, T, xi, yi, eos, step_size=1e-2):
    r"""
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    rho : float
        [mol/:math:`m^3`] Density array. Length depends on values in density_dict
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.

    Returns
    -------
    Residual : float
        
    """
    ncomp = len(xi)
    Psat = np.zeros(ncomp)
    for i in range(ncomp):
        tmp = np.zeros(ncomp)
        tmp[i] = 1.
        Psat[i], _, _ = calc_Psat(T, tmp, eos, **opts)

    activity_coefficient = yi*P/(Psat*xi)

    return activity_coefficient, Psat
