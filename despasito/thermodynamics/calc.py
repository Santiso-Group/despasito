"""
    This module contains our thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS. The thermo module contains a series of wrapper to handle the inputs and outputs of these functions.
    
"""

import numpy as np
import sys
from scipy import interpolate
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import fmin
from scipy.optimize import newton
from scipy.optimize import root
from scipy.optimize import brentq
from scipy.optimize import bisect
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.ndimage.filters import gaussian_filter1d
import random
import deap
import copy
import time
import matplotlib.pyplot as plt

from . import fund_constants as const

######################################################################
#                                                                    #
#                              Calc CC Params                        #
#                                                                    #
######################################################################
def calc_CC_Pguess(xilist, Tlist, CriticalProp):
    """
       Computes the mie parameters of a mixture from the mixed critical properties of the pure components. 
       From: Mejia, A., C. Herdes, E. Muller. Ind. Eng. Chem. Res. 2014, 53, 4131-4141

    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    Tc, Pc, omega, rho_7, Zc, Vc, M = CriticalProp

    ############## Calculate Mixed System Mie Parameters
    if all(-0.847 > x > 0.2387 for x in omega):
        print("Omega is outside of the range that these correlations are valid")

    a = [14.8359, 22.2019, 7220.9599, 23193.4750, -6207.4663, 1732.9600]
    b = [0.0, -6.9630, 468.7358, -983.6038, 914.3608, -1383.4441]
    c = [0.1284, 1.6772, 0.0, 0.0, 0.0, 0.0]
    d = [0.0, 0.4049, -0.1592, 0.0, 0.0, 0.0]
    j = [1.8966, -6.9808, 10.6330, -9.2041, 4.2503, 0.0]
    k = [0.0, -1.6205, -0.8019, 1.7086, -0.5333, 1.0536]

    Tcm, Pcm, sigma, epsilon, Psatm = [[] for x in range(5)]

    #Nmol = len(Tc)
    #for i in range(Nmol):
    #    for j in range(Nmol-i-1,Nmol):
    i = 0
    jj = 1

    ## Mixture alpha
    #omegaij = (omega[i]+omega[jj])/2.
    #tmp1 = np.sum([a[ii]*omegaij**ii for ii in range(6)])
    #tmp2 = np.sum([b[ii]*omegaij**ii for ii in range(6)])
    #l_r = tmp1/(1.+tmp2)
    #C = (l_r/(l_r-6.))*(l_r/6.)**(6./(l_r-6.))
    #al_tmp = C*(1./3. - 1./(l_r-3.))

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
            print("Prausnitz!")
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
def PvsRho(T, xi, eos, minrhofrac=(1.0 / 200000.0), rhoinc=5.0, vspacemax=1.0E-4, Pmax=1000.0 * 101325, maxpack=0.65):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

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
    Plist = eos.P(rholist * const.Nav, T, xi)

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

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    Psmoothed = gaussian_filter1d(Plist, sigma=.1)

    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed)
    roots = Pvspline.roots().tolist()
    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Psmoothed, k=4)
    extrema = Pvspline.derivative().roots().tolist()

    #PvsV_plot(vlist, Plist, Pvspline, markers=extrema)

    return Pvspline, roots, extrema

######################################################################
#                                                                    #
#                      Pressure-Volume Spline                        #
#                                                                    #
######################################################################
def PvsV_plot(vlist, Plist, Pvspline, markers=[]):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

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

    """
    Computes the saturated pressure, gas and liquid densities for a single component system given Temperature and Mie parameters
    T: Saturated Temperature in Kelvin
    minrhofrac: Fraction of maximum hard sphere packing fraction for gas density
    rhoinc: spacing densities for rholist in mol/m^3. Smaller values will generate a more accurate curve at increasing computational cost
    Pmax: maximum needed pressure in Pascals
    Returns Saturated Pressure in Pa, liquid denisty, and gas density in mol/m^3

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    if np.count_nonzero(xi) != 1:
        if np.count_nonzero(xi>0.1) != 1:
            raise ValueError("Multiple components have compositions greater than 10%, check code for source")
        else:
            ind = np.where((xi>0.1)==True)[0]
            raise ValueError("Multiple components have compositions greater than 0. Do you mean to obtain the saturation pressure of %s with a mole fraction of %g?" % (eos._beads[ind],xi[ind]))

    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    tmp = np.argwhere(np.diff(Plist) > 0)

    if not tmp.any():
        print('Error: One of the components is above its critical point, add exception to functions calc_xT_phase, calc_PT_phase, or calc_yT_phase')
        Psat = np.nan
        roots = [1.0, 1.0, 1.0]

    else:
        Pmin1 = np.argwhere(np.diff(Plist) > 0)[0][0]
        Pmax1 = np.argmax(Plist[Pmin1:]) + Pmin1

        Pmaxsearch = Plist[Pmax1]

        Pminsearch = max(Plist[-1], np.amin(Plist[Pmin1:Pmax1]))
        #print("Pmin", Pmin1, "Pminsearch", Pminsearch, "Pmax", Pmax1, "Pmaxsearch", Pmaxsearch)
        #search Pressure that gives equal area in maxwell construction
        Psat = minimize_scalar(eq_area,
                               args=(Plist, vlist),
                               bounds=(Pminsearch * 1.0001, Pmaxsearch * .9999),
                               method='bounded')

        #Using computed Psat find the roots in the maxwell construction to give liquid (first root) and vapor (last root) densities
        Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Plist - Psat.x)
        roots = Pvspline.roots()
        Psat = Psat.x

    #Psat,rholsat,rhogsat
    return Psat, 1.0 / roots[0], 1.0 / roots[2]

######################################################################
#                                                                    #
#                              Eq Area                               #
#                                                                    #
######################################################################
def eq_area(shift, Pv, vlist):

    """
    Computes the area below and above Psat guess line
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    Pvspline = interpolate.InterpolatedUnivariateSpline(vlist, Pv - shift)

    roots = Pvspline.roots()

    a = Pvspline.integral(roots[0], roots[1])
    b = Pvspline.integral(roots[1], roots[2])

    return (a + b)**2

######################################################################
#                                                                    #
#                              Calc Rho V Full                       #
#                                                                    #
######################################################################
def calc_rhov(P, T, xi, eos, rhodict={}):

    """
    Computes the saturated pressure, gas and liquid densities for a single component system given Temperature and Mie parameters
    T: Saturated Temperature in Kelvin
    minrhofrac: Fraction of maximum hard sphere packing fraction for gas density
    rhoinc: spacing densities for rholist in mol/m^3. Smaller values will generate a more accurate curve at increasing computational cost
    Pmax: maximum needed pressure in Pascals
    Returns Saturated Pressure in Pa, liquid denisty, and gas density in mol/m^3

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Plist = Plist-P
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    #print("Find rhov, P [Pa], roots [m^3/mol]:",P, roots)

    l_roots = len(roots)
    if l_roots == 0:
        flag = 4
        rho_tmp = np.nan
        print("This temperature and composition won't produce a fluid (vapor or liquid) at this pressure")
        PvsV_plot(vlist, Plist, Pvspline)
    elif l_roots == 1:
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            print("This T and xi combination produces a critical fluid at this pressure")
        elif (Pvspline(roots[0])+P) > (Pvspline(max(extrema))+P):
            print(extrema)
            print(roots)
            flag = 1
            rho_tmp = np.nan
            print("This T and xi combination produces a liquid at this pressure")
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            print("This T and xi combination produces a vapor at this pressure. Warning! approaching critical fluid")
    elif l_roots == 2:
        if (Pvspline(roots[0])+P) < 0.:
            flag = 1
            rho_tmp = np.nan
            print("This T and xi combination produces a liquid under tension at this pressure")
        else:
            flag = 3
            rho_tmp = np.nan
            print("There should be a third root! Assume ideal gass P:", P)
            #PvsV_plot(vlist, Plist, Pvspline)
    else: # 3 roots
        rho_tmp = 1.0 / roots[2]
        flag = 0

    if flag in [0,2]: # vapor or critical fluid
        tmp = [rho_tmp*.99, rho_tmp*1.01]
        if not (Pdiff(tmp[0],P, T, xi, eos)*Pdiff(tmp[1],P, T, xi, eos))<0:
            print("rhomin, rhomax:",tmp)
            PvsV_plot(vlist, Plist, Pvspline)
        rho_tmp = brentq(Pdiff, tmp[0], tmp[1], args=(P, T, xi, eos), rtol=0.0000001)

    # Flag: 0 is gas, 1 is liquid, 2 mean a critical fluid, 3 means we should assume ideal, 4 means that neither is true
    return rho_tmp, flag


######################################################################
#                                                                    #
#                              Calc Rho L Full                       #
#                                                                    #
######################################################################
def calc_rhol(P, T, xi, eos, rhodict={}):

    """
    Computes the saturated pressure, gas and liquid densities for a single component system given Temperature and Mie parameters
    T: Saturated Temperature in Kelvin
    minrhofrac: Fraction of maximum hard sphere packing fraction for gas density
    rhoinc: spacing densities for rholist in mol/m^3. Smaller values will generate a more accurate curve at increasing computational cost
    Pmax: maximum needed pressure in Pascals
    Returns Saturated Pressure in Pa, liquid denisty, and gas density in mol/m^3

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    # Get roots and local minima and maxima 
    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist-P)

    #print("Find rhol, P [Pa], roots [m^3/mol]:",P, roots)

    # Assess roots, what is the liquid density
    l_roots = len(roots)
    if l_roots in [1, 2]:
        rho_tmp = 1.0 / roots[0]
        flag = 0
        if (Pvspline(roots[0])+P) < 0.:
            print("This T and xi combination produces a liquid under tension")
            if l_roots == 1:
                print("Warning! There should be two roots if it is under tension")
                PvsV_plot(vlist, Plist, Pvspline)
    elif len(roots) == 0:
        rho_tmp = np.nan
        flag = 2
        print("This temperature and composition won't produce a fluid (vapor or liquid)")
        PvsV_plot(vlist, Plist, Pvspline)
    else:
        rho_tmp = 1.0 / roots[0]
        flag = 0

    if flag == 0:
        rho_tmp = brentq(Pdiff, rho_tmp*0.75, rho_tmp*1.5, args=(P, T, xi, eos), rtol=0.0000001)

    # Flag: 0 is liquid, 1 is gas, 2 means that neither is true
    return rho_tmp, flag

######################################################################
#                                                                    #
#                              Calc Pdiff                            #
#                                                                    #
######################################################################
def Pdiff(rho, Pset, T, xi, eos):
    """
    Calculate difference between setpoint pressure and computed pressure for a given density
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    Pguess = eos.P(rho * const.Nav, T, xi)

    return (Pguess - Pset)

######################################################################
#                                                                    #
#                          Calc phi vapor                            #
#                                                                    #
######################################################################
def calc_phiv(P, T, yi, eos, rhodict={}):

    """
    Calculate fugacity coefficient
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    rhov, flagv = calc_rhov(P, T, yi, eos, rhodict)
    if flagv == 3:
        phiv = np.ones_like(yi)
    else:
        muiv = eos.chemicalpotential(P, np.array([rhov]), yi, T)
        phiv = np.exp(muiv)

    return phiv, rhov

######################################################################
#                                                                    #
#                         Calc phi liquid                            #
#                                                                    #
######################################################################
def calc_phil(P, T, xi, eos, rhodict={}):

    """
    Calculate fugacity coefficient
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    rhol, flagl = calc_rhol(P, T, xi, eos, rhodict)
    muil = eos.chemicalpotential(P, np.array([rhol]), xi, T)
    phil = np.exp(muil)

    return phil, rhol

######################################################################
#                                                                    #
#                              Calc P range                          #
#                                                                    #
######################################################################
def calc_Prange(T, xi, yi, eos, rhodict={}, Pmin=1000):

    """
    Computes the pressure range given Temperature and Mie parameters
    T: Saturated Temperature in Kelvin
    minrhofrac: Fraction of maximum hard sphere packing fraction for gas density
    rhoinc: spacing densities for rholist in mol/m^3. Smaller values will generate a more accurate curve at increasing computational cost
    Pmax: maximum needed pressure in Pascals
    Returns Saturated Pressure in Pa, liquid denisty, and gas density in mol/m^3

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = PvsRho(T, xi, eos, **rhodict)
    Pvspline, roots, extrema = PvsV_spline(vlist, Plist)

    Pmax = max(Pvspline(extrema))
    Parray = [Pmin, Pmax]

    #################### Find Pressure range and Objective Function values

    # Root of min from liquid curve is absolute minimum
    ObjArray = [0, 0]
    yi_range = yi

    ind = 0
    maxiter = 200
    for z in range(maxiter):
        if z == 0:
            # Find Obj Function for Min pressure above
            p = Parray[0]
            phil, rhol = calc_phil(p, T, xi, eos, rhodict={})
            yi_range, phiv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, maxitr=50)
            ObjArray[0] = (np.sum(xi * phil / phiv) - 1.0)
            print("!!!!!!!!!!!!!!!! Pmin", Parray[0], "Obj. Func", ObjArray[0])
        elif z == 1:
            # Find Obj function for Max Pressure above
            p = Parray[1]
            phil, rhol = calc_phil(p, T, xi, eos, rhodict={})
            yi_range, phiv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, maxitr=50)
            ObjArray[1] = (np.sum(xi * phil / phiv) - 1.0)
            print("!!!!!!!!!!!!!!!! Estimate Pmax", Parray[1], "Obj. Func", ObjArray[1])
        else:
            tmp_sum = np.abs(ObjArray[-2] + ObjArray[-1])
            tmp_dif = np.abs(ObjArray[-2] - ObjArray[-1])
            if tmp_dif > tmp_sum:
                print("Got the pressure range!")
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
            elif z == maxiter:
                print('A change in sign for the objective function could not be found, inspect progress')
                plt.plot(Parray, ObjArray)
                plt.plot([Parray[0], Parray[-1]], [0, 0], 'k')
                plt.ylabel("Obj. Function")
                plt.xlabel("Pressure / Pa")
                plt.show()
                sys.exit('Error: A change in sign for the objective function could not be found')
            else:
                p = 2 * Parray[-1]
                Parray.append(p)
                phil, rhol = calc_phil(p, T, xi, eos, rhodict={})
                yi_range, phiv = solve_yi_xiT(yi_range, xi, phil, p, T, eos, rhodict=rhodict, maxitr=50)
                ObjArray.append(np.sum(xi * phil / phiv) - 1.0)

    Prange = Parray[-2:]
    ObjRange = ObjArray[-2:]
    print("[Pmin, Pmax]", Prange, "Obj. Values", ObjRange)

    yi_global = yi_range

    return Prange, Pguess

######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_yi_xiT(yi, xi, phil, P, T, eos, rhodict={}, maxitr=50):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global
    yi /= np.sum(yi)

# Option 1
#    yimin=root(solve_yi_root,yi,args=(xi,phil,P,T,eos,rhodict),method='broyden1',options={'fatol':0.0001,'maxiter':15})
#    yi = yimin.x
#    print "!!!! Output", yimin.x
#    yi/=np.sum(yi)

#    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})


    yi_tmp = []
    for z in range(maxitr):

        print("yi guess", yi)

        yi /= np.sum(yi)
# Please
        # Try yi
        phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
        
        if any(np.isnan(phiv)): # If vapor density doesn't exist
            print("Let's find it!")
            yinew = find_new_yi(P, T, phil, xi, eos, rhodict=rhodict, maxitr=1000)
            phiv, rhov = calc_phiv(P, T, yinew, eos, rhodict={})
            if any(np.isnan(yinew)): 
                phiv = np.nan
                sys.exit("This shouldn't be happening")

        yinew = xi * phil / phiv
        yinew = yinew / np.sum(yinew)
        
        # Check convergence
        ind_tmp = np.where(yi == min(yi))[0]
        if np.abs(yinew[ind_tmp] - yi[ind_tmp]) / yi[ind_tmp] < 1e-5:
            yi_global = yi
            print("!!!!!!! Found yi !!!!!!")
            break

        # Check for bouncing between values, then updated yi to yinew
        if len(yi_tmp) > 4:
            if all(np.abs((yi - yi_tmp[-3]) / yi + (yinew - yi_tmp[-2]) / yinew) < 1e-2):
                yi = (yi + yinew) / 2
                print("New guess:", yi)
            else:
                yi = yinew
        else:
            yi = yinew
        yi_tmp.append(yi)

    ## If yi wasn't found in defined number of iterations
    if z == maxitr - 1:
        print('More than ', maxitr, ' iterations needed, % error: ', np.sum(np.abs((yinew - yi)[0:] / yi[0:])))
        yi_tmp = np.array(yi_tmp).T
        #NoteHere Benzene
        for i in range(len(yi)):
            plt.plot(yi_tmp[i], label="$y_{%g}$" % i)
            plt.xlabel("Iteration")
            plt.ylabel("Vapor Fraction")
            plt.legend(loc="best")
            plt.show()

    print("Final yi: ", yi)

    return yi, phiv

######################################################################
#                                                                    #
#                       Solve Yi for xi and T                        #
#                                                                    #
######################################################################
def solve_xi_yiT(xi, yi, phiv, P, T, eos, rhodict={}, maxitr=50):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global xi_global
    xi /= np.sum(xi)

    xi_tmp = []
    for z in range(maxitr):

        print("xi guess", xi)

        xi /= np.sum(xi)
# Please
        # Try xi
        phil, rhol = calc_phil(P, T, xi, eos, rhodict={})
        
        if any(np.isnan(phil)): # If vapor density doesn't exist
            raise ValueError("Contingency methods of determining xi haven't been written yet")
            xinew = find_new_xi(P, T, phiv, yi, eos, rhodict=rhodict, maxitr=1000)
            phil, rhol = calc_phil(P, T, xinew, eos, rhodict={})
            if any(np.isnan(xinew)): 
                phil = np.nan
                sys.exit("This shouldn't be happening")

        xinew = yi * phiv / phil
        xinew = xinew / np.sum(xinew)
        
        # Check convergence
        ind_tmp = np.where(xi == min(xi))[0]
        if np.abs(xinew[ind_tmp] - xi[ind_tmp]) / xi[ind_tmp] < 1e-5:
            xi_global = xi
            print("!!!!!!! Found xi !!!!!!")
            break

        # Check for bouncing between values, then updated xi to xinew
        if len(xi_tmp) > 4:
            if all(np.abs((xi - xi_tmp[-3]) / xi + (xinew - xi_tmp[-2]) / xinew) < 1e-2):
                xi = (xi + xinew) / 2
                print("New guess:", xi)
            else:
                xi = xinew
        else:
            xi = xinew
        xi_tmp.append(xi)

    ## If xi wasn't found in defined number of iterations
    if z == maxitr - 1:
        print('More than ', maxitr, ' iterations needed, % error: ', np.sum(np.abs((xinew - xi)[0:] / xi[0:])))
        xi_tmp = np.array(xi_tmp).T
        #NoteHere Benzene
        for i in range(len(xi)):
            plt.plot(xi_tmp[i], label="$x_{%g}$" % i)
            plt.xlabel("Iteration")
            plt.ylabel("Liquid Fraction")
            plt.legend(loc="best")
            plt.show()

    print("Final xi: ", xi)

    return xi, phil

######################################################################
#                                                                    #
#                       Sum of Yi                                    #
#                                                                    #
######################################################################


def sum_yi(P, yi, xi, T, phil, eos, rhodict={}, maxitr=50):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global

    y_total = np.sum(yi)
    for i in range(maxitr):
        phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
        yinew = xi * phil / phiv

        print("P:", P, " yi:", yi, " yinew:", yinew, " yinew tot:", np.sum(yinew))
        obj_out = np.sum(yinew) - 1
        if (y_total - np.sum(yinew)) < 1e-3:
            yi = yinew
            break
        else:
            yi = yinew / np.sum(yinew)
            y_total = np.sum(yinew)
        print(yi)

    if i == maxitr - 1:
        sys.exit("yi didn't converged in sum_yi()")
    yi_global = yi

    return obj_out


######################################################################
#                                                                    #
#                       Find new Yi                                  #
#                                                                    #
######################################################################


def find_new_yi(P, T, phil, xi, eos, rhodict={}, maxitr=50):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    #    # have three functions, make sum close to zero, mui isn't np.nan, and rho is a vapor
    #    deap.creator.create("FitnessMulti",deap.base.Fitness,weights=(-1.0, -1.0, 0.5))
    #    deap.creator.create("Inividual", list, fitness=deap.creator.FitnessMax)
    #    toolbox = deap.base.Toolbox()
    # # NoteHere, make individuals add up to 1?
    #    toolbox.register("attr_bool", random.randit, 0, 1)
    #    toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attr_bool, n=l_yi)
    #    toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
    #
    #    def obj_func(individual):
    #        return np.sum(individual)
    #
    #    toolbox.register("evaluate", obj_func)
    #    toolbox.register("mate", tools.cxTwoPoint)
    #    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    #    toolbox.register("select", tools.selTournament, tournsize=3)
    #
    #    population = toolbox.population(n=300)
    #
    #    NGEN=40
    #    for gen in range(NGEN):
    #        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #        fits = toolbox.map(toolbox.evaluate, offspring)
    #        for fit, ind in zip(fits, offspring):
    #            ind.fitness.values = fit
    #        population = toolbox.select(offspring, k=len(population))
    #    top10 = tools.selBest(population, k=10)

   # # My attempt at using a heuristic approach
   # # Make new random guesses for yi, and test to find a feasible one
   # l_yi = len(xi)
   # Nguess = 10
   # yiguess = eos.chemicalpotential(P, np.array([rhov_tmp]), yi_g_tmp, T)[]
   # rhov_guess = []
   # obj_guess = []
   # for j in range(maxitr):  # iterate until 10 feasible options are found
   #     yi_g_tmp = np.zeros(l_yi)
   #     # Make guesses for yi
   #     yi_g_tmp[0:-1] = np.random.random((1, l_yi - 1))[0]
   #     yi_g_tmp[-1] = 1 - np.sum(yi_g_tmp)
   #     # Test guess
   #     phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
   #     if all(np.isnan(muiv_tmp) == False):

   #         yiguess.append(yi_g_tmp)
   #         rhov_guess.append(rhov_tmp)

   #         phiv = np.exp(muiv_tmp)
   #         obj_tmp = np.sum(xi * phil / phiv) - 1
   #         print(rhov_tmp, muiv, phiv, obj_tmp)

   #         obj_guess.append(np.abs(obj_tmp))

   #     if len(yiguess) == Nguess:
   #         break
   # # Choose the yi value to continue with based on new guesses
   # ind = np.where(obj_guess == min(obj_guess))[0]
   # print(obj_guess, ind)
   # yi = yiguess[ind]
   # rhov = rhov_guess[ind]

    yi_ext = np.linspace(0,1,10) # Guess for yi
    obj_ext = []
    for yi in yi_ext:
        obj_tmp = yi_obj(yi,P,T,phil,xi,eos,rhodict)
        obj_ext.append(obj_tmp)

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    print(tmp)
    if tmp == 0:
        yi_tmp = np.nan
        obj_tmp = np.nan
    elif tmp == 1:
        yi_tmp = yi_ext[np.where(~np.isnan(obj_ext))]
        obj_tmp = obj_ext[np.where(~np.isnan(obj_ext))]
    else:
        #yi_tmp = brentq(yi_obj,yi_ext[ind[0]],yi_ext[ind[-1]],args=(P,T,phil,xi,eos,rhodict),rtol=0.0000001)
        obj_tmp = [obj_ext[ii] for ii in ind]
        iind = np.where(obj_tmp==min(np.abs(obj_tmp)))[0][0]
        yi_tmp = yi_ext[iind]
    print("Found yi:", yi_tmp, "Obj:",obj_tmp)
    yi = yi_tmp
    if type(yi) != list:
        yi = [yi, 1-yi]

    return yi

######################################################################
#                                                                    #
#                   Yi Obj Func                                      #
#                                                                    #
######################################################################
def yi_obj(yi,P,T,phil,xi,eos,rhodict={}):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    if type(yi) != list:
        yi = [yi, 1-yi]

    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
    obj = np.sum(xi*phil/phiv)-1

    print(yi, obj)

    return obj

######################################################################
#                                                                    #
#                              Solve Xi in root finding              #
#                                                                    #
######################################################################
def solve_xi_root(xi0, yi, phiv, P, T, eos, rhodict):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    # !!!!!!!!!!!!!!! This isn't working !!!!!!!!!!!!!!!!
    # Check calc_phase_test_3.py for old version named "solve_xi"

    for i in range(np.size(xi0)):
        if xi0[i] < 0.0:
            return -1.0
    if np.sum(xi0) > 1.0:
        return 1.0
    xi = np.zeros_like(yi)
    #xi0=abs(xi0)

    xi[0:np.size(xi0)] = xi0
    xi[-1] = 1.0 - np.sum(xi0)

    #xi=abs(xi)
    xi /= np.sum(xi)
    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})
    xinew = yi * phiv / phil
    xinew = xinew / np.sum(xinew)

    #   ind_tmp = np.where(yi==min(yi))[0]
    #   print 'xinew',xinew,xi,' Percent Error: ',np.abs(xinew[ind_tmp]-xi[ind_tmp])/xi[ind_tmp]
    #   return np.abs(xinew[ind_tmp]-xi[ind_tmp])/xi[ind_tmp]

    print('xinew', xinew, xi, ' Percent Error: ', (xinew - xi) / xi * 100)
    return np.abs(xinew - xi) / xi


######################################################################
#                                                                    #
#                       Solve Yi in Root Finding                     #
#                                                                    #
######################################################################

# !!!!!!!!!!!!!!! This isn't working !!!!!!!!!!!!!!!!


def solve_yi_root(yi0, xi, phil, P, T, eos, rhodict={}, maxitr=50):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    yi0 /= np.sum(yi0)
    yi = yi0

    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
    yinew = xi * phil / phiv
    yinew = yinew / np.sum(yinew)

    # ind_tmp = np.where(yi==min(yi))[0]
    # print 'yinew',yinew,yi,' Percent Error: ',np.abs(yinew[ind_tmp]-yi[ind_tmp])/yi[ind_tmp]
    #   return np.abs(yinew[ind_tmp]-yi[ind_tmp])/yi[ind_tmp]

    print('yinew', yinew, yi, ' Percent Error: ', (yinew - yi) / yi * 100)
    return yinew - yi


#   return np.abs(yinew-yi)
#   return (yinew-yi)/yi
#   return np.abs(yinew-yi)/yi

######################################################################
#                                                                    #
#                              Solve P xT                            #
#                                                                    #
######################################################################
def solve_P_xiT(P, xi, T, eos, rhodict):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global

    if P < 0:
        return 10.0

    #yi=np.array([0.99,0.01])
    #find liquid density
    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})

    yinew, phiv = solve_yi_xiT(yi_global, xi, phil, P, T, eos, rhodict=rhodict, maxitr=50)
    yi_global = yi_global / np.sum(yi_global)

    #given final yi recompute
    phiv, rhov = calc_phiv(P, T, yi_global, eos, rhodict={})

    Pv_test = eos.P(rhov*const.Nav, T, yi_global)
    print('Obj Func', (np.sum(xi * phil / phiv) - 1.0), "Pset", P, "Pcalcv",Pv_test[0])

    return (np.sum(xi * phil / phiv) - 1.0)

######################################################################
#                                                                    #
#                              Solve P yT                            #
#                                                                    #
######################################################################
def solve_P_yiT(P, yi, T, eos, rhodict):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global xi_global

    if P < 0:
        return 10.0

    #yi=np.array([0.99,0.01])
    #find liquid density
    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})

    xinew, phil = solve_xi_yiT(xi_global, yi, phiv, P, T, eos, rhodict=rhodict, maxitr=50)
    xi_global = xi_global / np.sum(xi_global)

    #given final yi recompute
    phil, rhol = calc_phil(P, T, xi_global, eos, rhodict={})

    Pv_test = eos.P(rhov*const.Nav, T, xi_global)
    print('Obj Func', (np.sum(yi * phiv / phil) - 1.0), "Pset", P, "Pcalcv",Pv_test[0])

    return (np.sum(yi * phiv / phil) - 1.0)


######################################################################
#                                                                    #
#                              Solve P xiT inerp                     #
#                                                                    #
######################################################################
def solve_P_xiT_inerp(P, Psat, xi, T, eos, rhodict):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    #print 'P',P
    if P < 0:
        return 10.0

    if P > 30000000.0:
        return 10.0

    yi = xi * Psat / P
    yi /= np.sum(yi)
    #yi=np.array([0.99,0.01])
    #find liquid density
    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})

    #print phiv
    #print xi[0]
    #test=solve_xi_root(xi[0],yi,phiv,P,T,eos,rhodict)
    #print test
    #yimin=fsolve(solve_yi,yi[0:-1],args=(xi,phil,P,T,eos,rhodict)
    try:
        yimin = root(solve_yi,
                     yi[0:-1],
                     args=(xi, phil, P, T, eos, rhodict),
                     method='broyden1',
                     options={
                         'fatol': 0.0001,
                         'maxiter': 15
                     })
        yi = np.zeros_like(xi)
        yi[0:np.size(yimin.x)] = yimin.x
        yi[-1] = 1.0 - np.sum(yi)
    except:
        return 10.0
        #for i in range(10):
        #    yidiff,yi=solve_yi_iter(yi[0:-1],xi,phil,P,T,eos,rhodict=rhodict,maxpack)
        #    if np.sum(abs(yidiff))< 0.0001: break
        #yi=np.zeros_like(xi)
        #yi[0:np.size(yimin.x)]=yimin
        #yi[-1]=1.0-np.sum(yi)

    #yimin=root(solve_yi,yi[0:-1],args=(xi,phil,P,T,eos,rhodict=rhodict,maxpack))

    #yi=np.zeros_like(xi)
    #yi[0:np.size(yimin.x)]=yimin.x
    #yi[-1]=1.0-np.sum(yi)
    #yi=np.array([1.0])
    #massi=massi[0]
    #eos._nui=np.array([eos._nui[0]])
    #given final yi recompute
    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
    print('Pconv', (np.sum(xi * phil / phiv) - 1.0), P, rhov)
    return (np.sum(xi * phil / phiv) - 1.0)

######################################################################
#                                                                    #
#                   Set Psat for Critical Components                 #
#                                                                    #
######################################################################
def setPsat(ind, eos):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

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
            Psat = np.nan
            NaNbead = eos._beads[j]
    try:
        NaNbead
    except:
       NaNbead = "No NaNbead"

    return Psat, NaNbead 

######################################################################
#                                                                    #
#                              Calc yT phase                         #
#                                                                    #
######################################################################
def calc_yT_phase(yi, T, eos, rhodict, Pguess=[],meth="broyden1"):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

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
                sys.exit("Component, %s, is beyond it's critical point at %g K. Add an exception to setPsat" % (NaNbead,T))

    # Estimate initial pressure
    if not Pguess:
        P=1.0/np.sum(yi/Psat)
    else:
        P = Pguess

    # Estimate initial xi
    if ("xi_global" not in globals() or "True" in np.isnan(xi_global)):
        xi_global = P * (yi / Psat)
        xi_global /= np.sum(xi_global)
        xi_global = copy.deepcopy(xi_global)
    xi = xi_global 

    Pfinal = root(solve_P_yiT,
                  P,
                  args=(yi, T, eos, rhodict),
                  method='broyden1',
                  options={
                      'fatol': 0.0001,
                      'maxiter': 15
                  })

    #Given final P estimate
    P = Pfinal.x

    phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})

    ximin = root(solve_xi_root,
                 xi[0:-1],
                 args=(yi, phiv, P, T, eos, rhodict),
                 method='broyden1',
                 options={
                     'fatol': 0.00001,
                     'maxiter': 20
                 })

    xi = np.zeros_like(yi)
    xi[0:np.size(ximin.x)] = ximin.x
    xi[-1] = 1.0 - np.sum(xi)
    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})

    #xitol=0.1
    #xitol=0.0001
    #for z in range(50):
    #find rhol
    #phil, rhol = calc_phil(P, T, xi, eos, rhodict={})
    #xinew=yi*phiv/phil
    #xinew=xinew/np.sum(xinew)
    #if np.sum(abs(xinew-xi)) < xitol:
    #xi=xinew
    #break
    #else:
    #xi=xinew
    return P, xi

######################################################################
#                                                                    #
#                              Calc xT phase                         #
#                                                                    #
######################################################################
def calc_xT_phase(xi, T, eos, rhodict={}, Pguess=[],meth="broyden1"):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                sys.exit("Component, %s, is beyond it's critical point. Add an exception to setPsat" % (NaNbead))

#estimate initial pressure
    if not Pguess:
        Pguess = np.sum(xi * Psat)
        P = np.sum(xi * Psat)
    else:
        P = Pguess

    if ("yi_global" not in globals() or "True" in np.isnan(yi_global)):
        print("Guess yi in calc_xT_phase with Psat")
        yi_global = xi * Psat / P
        yi_global /= np.sum(yi_global)
        yi_global = copy.deepcopy(yi_global)
    yi = yi_global

    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})

#    print("Initial: P ", Pguess, "   yi ", yi)
#    Pguess, yi = bubblepoint_guess(Pguess, yi, xi, T, phil, eos, rhodict)
#    print("Updated: P ", Pguess, "   yi ", yi, "\n\n\n\n")

    #P=4130879.792
    #Pfinal=root(solve_P_xiT,P,args=(Psat,xi,T,eos,rhodict),method='broyden1',options={'fatol':0.0001,'maxiter':25,'jac_options': {'reduction_method': 'simple'}})

    Prange, Pguess = calc_Prange(T, xi, yi, eos, rhodict)
    print("Given Pguess:", P, "Suggested", Pguess)
    P = Pguess

    print("Method:", meth)
    #################### Root Finding without Boundaries ###################
    if meth in ['broyden1', 'broyden2']:
        Pfinal = root(solve_P_xiT,
                      P,
                      args=(xi, T, eos, rhodict),
                      method=meth,
                      options={
                          'fatol': 0.0001,
                          'maxiter': 25,
                          'jac_options': {
                              'reduction_method': 'simple'
                          }
                      })
    elif meth == 'hybr_broyden1':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="hybr")
        Pfinal = root(solve_P_xiT,
                      Pfinal.x,
                      args=(xi, T, eos, rhodict),
                      method="broyden1",
                      options={
                          'fatol': 0.0001,
                          'maxiter': 25,
                          'jac_options': {
                              'reduction_method': 'simple'
                          }
                      })
    elif meth == 'hybr_broyden2':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="hybr")
        Pfinal = root(solve_P_xiT,
                      Pfinal.x,
                      args=(xi, T, eos, rhodict),
                      method="broyden2",
                      options={
                          'fatol': 0.0001,
                          'maxiter': 25,
                          'jac_options': {
                              'reduction_method': 'simple'
                          }
                      })
    elif meth == 'anderson':
        Pfinal = root(solve_P_xiT,
                      P,
                      args=(xi, T, eos, rhodict),
                      method=meth,
                      options={
                          'fatol': 0.0001,
                          'maxiter': 25
                      })
    elif meth == 'hybr':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'lm':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'linearmixing':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'diagbroyden':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="hybr")
        Pfinal = root(solve_P_xiT, Pfinal.x, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'excitingmixing':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'krylov':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
    elif meth == 'df-sane':
        Pfinal = root(solve_P_xiT, P, args=(xi, T, eos, rhodict), method=meth)
#################### Minimization Methods with Boundaries ###################
    elif meth == "TNC":
        if len(Prange) == 2:
            print([tuple(Prange)], len((tuple(Prange))))
            Pfinal = minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="TNC", bounds=[tuple(Prange)])
        else:
            Pfinal = minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="TNC")
    elif meth == "L-BFGS-B":
        if len(Prange) == 2:
            print([tuple(Prange)], len((tuple(Prange))))
            Pfinal = minimize(solve_P_xiT,
                              P,
                              args=(xi, T, eos, rhodict),
                              method="L-BFGS-B",
                              bounds=[tuple(Prange)])
        else:
            Pfinal = minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="L-BFGS-B")
    elif meth == "SLSQP":
        if len(Prange) == 2:
            print([tuple(Prange)], len((tuple(Prange))))
            Pfinal = minimize(solve_P_xiT,
                              P,
                              args=(xi, T, eos, rhodict),
                              method="SLSQP",
                              bounds=[tuple(Prange)],
                              options={
                                  'fatol': 0.0001,
                                  'maxiter': 25
                              })
        else:
            Pfinal = minimize(solve_P_xiT, P, args=(xi, T, eos, rhodict), method="SLSQP")
#################### Root Finding with Boundaries ###################
    elif meth == "brent":
        Pfinal = brentq(solve_P_xiT, Prange[0], Prange[1], args=(xi, T, eos, rhodict), rtol=0.0000001)

    #Given final P estimate
    if meth != "brent":
        P = Pfinal.x

    #find liquid density and fugacity
    phil, rhol = calc_phil(P, T, xi, eos, rhodict={})
    yi, phiv = solve_yi_xiT(yi_global, xi, phil, P, T, eos, rhodict)
    yi_global = yi

    return P, yi_global


######################################################################
#                                                                    #
#                              Calc xT phase dir                     #
#                                                                    #
######################################################################
def calc_xT_phase_dir(xi, T, eos, rhodict={}, Pguess=[]):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    global yi_global

    Psat = np.zeros_like(xi)
    saft_args_tmp = saft_args
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                sys.exit("Component, %s, is beyond it's critical point. Add an exception to setPsat" % (NaNbead))

    #estimate initial pressure
    if not Pguess:
        P = np.sum(xi * Psat)
    else:
        P = Pguess

    # guess yi
    yi = xi * Psat / P
    yi /= np.sum(yi)

    maxitr = 25
    for zz in range(maxitr):
        ########## replace solve_P_xiT #############
        #####def solve_P_xiT(P,Psat,xi,T,eos,rhodict):

        #print 'P',P
        if P < 0:
            #return 10.0 # change the objective function (difference in error) to much higher, pushing the P way down
            P = np.nan
            break
        elif P > 1000.0 * 101325.0:
            #return 10.0
            P = np.nan
            break

        #find liquid density
        phil, rhol = calc_phil(P, T, xi, eos, rhodict={})
        yinew, phiv = solve_yi_xiT(yi, xi, phil, P, T, eos, rhodict)
        yi_global = yi
        #given final yi recompute
        phiv, rhov = calc_phiv(P, T, yi, eos, rhodict={})
        Pv_test = eos.P(rhov * const.Nav, T, yi)
        print(Pv_test, 'Pa')
        print('Pconv', (np.sum(xi * phil / phiv) - 1.0), P, rhov, rhov_full, rhol)
        P = Pv_test
        if (np.sum(xi * phil / phiv) - 1.0) < 0.0001:
            break
    if zz == maxitr - 1:
        print('More than ', maxitr, ' iterations needed for P, abs error: ', (np.sum(xi * phil / phiv) - 1.0))
#########################
    return P, yi


######################################################################
#                                                                    #
#                              Calc PT phase                         #
#                                                                    #
######################################################################
def calc_PT_phase(xi, T, eos, rhodict={}):

    """
    Placeholder function to show example docstring (NumPy format)
    
    Replace this function and doc string for your own project
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], rholsat, rhogsat = calc_Psat(T, xi_tmp, eos, rhodict)
        if np.isnan(Psat[i]):
            Psat[i], NaNbead = setPsat(i, eos)
            if np.isnan(Psat[i]):
                sys.exit("Component, %s, is beyond it's critical point. Add an exception to setPsat" % (NaNbead))

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
    """
    Given rho N/m3 and T compute denstiy given SAFT parameters
    
    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from
    
    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    step = np.sqrt(np.finfo(float).eps) * T * 1000.0
    nrho = np.size(rho)

    #computer rho+step and rho-step for better a bit better performance
    Ap = calchelmholtz.calc_A(np.array([rho]), xi, T + step, eos)
    Am = calchelmholtz.calc_A(np.array([rho]), xi, T - step, eos)

    return (Ap - Am) / (2.0 * step)

