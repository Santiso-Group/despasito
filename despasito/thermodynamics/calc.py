"""
This module contains our thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an Eos object so that these functions can be used with any EOS. The thermo module contains a series of wrapper to handle the inputs and outputs of these functions.
    
"""

import sys
import numpy as np
from scipy import interpolate
import scipy.optimize as spo
from scipy.ndimage.filters import gaussian_filter1d
import copy
import logging

import despasito.utils.general_toolbox as gtb
from despasito import fundamental_constants as constants
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


def pressure_vs_volume_arrays(
    T,
    xi,
    Eos,
    min_density_fraction=(1.0 / 500000.0),
    density_increment=5.0,
    max_volume_increment=1.0e-4,
    pressure_min=100,
    maxiter=25,
    multfactor=2,
    extended_npts=20,
    max_density=None,
    density_max_opts={},
    **kwargs
):

    r"""
    Give arrays with specific volume and pressure calculated from the given an EOS. 

    Options for this functions are provided in other functions with the keyword variable `density_opts`
    
    Parameters
    ----------
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    min_density_fraction : float, Optional, default=(1.0/500000.0)
        Fraction of the maximum density used to calculate, and is equal to, the minimum density of the density array. The minimum density is the reciprocal of the maximum specific volume used to calculate the roots.
    density_increment : float, Optional, default=5.0
        The increment between density values in the density array.
    max_volume_increment : float, Optional, default=1.0E-4
        Maximum increment between specific volume array values. After conversion from density to specific volume, the increment values are compared to this value.
    pressure_min : float, Optional, default=100
        Ensure pressure curve reaches down to this value
    multfactor : int, Optional, default=2
        Multiplication factor to extend range
    extended_npts : int, Optional, default=20
        Number of points in extended range
    maxiter : int, Optional, default=25
        Number of times to multiply range by to obtain full pressure vs. specific volume curve
    max_density : float, Optional, default=None
        [mol/m^3] Maximum molar density defined, if default of None is used then the Eos object method, density_max is used.
    density_max_opts : dict, Optional, default={}
        Keyword arguments for density_max method for EOS object

    Returns
    -------
    vlist : numpy.ndarray
        [:math:`m^3`/mol] Specific volume array.
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'pressure_vs_volume_arrays' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    if np.any(np.isnan(xi)):
        raise ValueError("Given mole fractions are NaN")

    if isinstance(xi, list):
        xi = np.array(xi)

    # estimate the maximum density based on the hard sphere packing fraction, part of EOS
    if not max_density:
        max_density = Eos.density_max(xi, T, **density_max_opts)
    elif gtb.isiterable(max_density):
        logger.error(
            "    Maxrho should be type float. Given value: {}".format(max_density)
        )
        max_density = max_density[0]

    if max_density > 1e5:
        raise ValueError(
            "Max density of {} mol/m^3 is not feasible, check parameters.".format(
                max_density
            )
        )

    # min rho is a fraction of max rho, such that minrho << rhogassat
    minrho = max_density * min_density_fraction
    # list of densities for P,rho and P,v
    if (max_density - minrho) < density_increment:
        raise ValueError(
            "Density range, {}, is less than increment, {}. Check parameters used in Eos.density_max().".format(
                (max_density - minrho), density_increment
            )
        )

    rholist = np.arange(minrho, max_density, density_increment)
    # check rholist to see when the spacing
    vspace = (1.0 / rholist[:-1]) - (1.0 / rholist[1:])
    if np.amax(vspace) > max_volume_increment:
        vspaceswitch = np.where(vspace > max_volume_increment)[0][-1]
        rholist_2 = (
            1.0
            / np.arange(
                1.0 / rholist[vspaceswitch + 1], 1.0 / minrho, max_volume_increment
            )[::-1]
        )
        rholist = np.append(rholist_2, rholist[vspaceswitch + 2 :])

    # compute Pressures (Plist) for rholist
    Plist = Eos.pressure(rholist, T, xi)

    # Make sure enough of the pressure curve is obtained
    for i in range(maxiter):
        if Plist[0] > pressure_min:
            rhotmp = np.linspace(rholist[0] / 2, rholist[0], extended_npts)[:-1]
            Ptmp = Eos.pressure(rhotmp, T, xi)
            Plist = np.append(Ptmp, Plist)
            rholist = np.append(rhotmp, rholist)
        else:
            break

    # Flip Plist and rholist arrays
    Plist = Plist[:][::-1]
    rholist = rholist[:][::-1]
    vlist = 1.0 / rholist

    return vlist, Plist


def pressure_vs_volume_spline(vlist, Plist):
    r"""
    Fit arrays of specific volume and pressure values to a cubic Univariate Spline.
    
    Parameters
    ----------
    vlist : numpy.ndarray
        [:math:`m^3`/mol] Specific volume array.
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
        if len(extrema) > 2:
            extrema = extrema[0:2]

    # pressure_vs_volume_plot(vlist, Plist, Pvspline, markers=extrema)

    if np.any(np.isnan(Plist)):
        roots = [np.nan]

    return Pvspline, roots, extrema


def pressure_vs_volume_plot(vlist, Plist, Pvspline, markers=[], **kwargs):
    r"""
    Plot pressure vs. specific volume.
    
    Parameters
    ----------
    vlist : numpy.ndarray
        [:math:`m^3`/mol] Specific volume array.
    Plist : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    Pvspline : obj
        Function object of pressure vs. specific volume
    markers : list, Optional, default=[]
        List of plot markers used in plot
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'pressure_vs_volume_plot' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    try:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(vlist, Plist, label="Orig.")
        plt.plot(vlist, Pvspline(vlist), label="Smoothed")
        plt.plot([vlist[0], vlist[-1]], [0, 0], "k")
        for k in range(len(markers)):
            plt.plot([markers[k], markers[k]], [min(Plist), max(Plist)], "k")
        plt.xlabel("Specific Volume [$m^3$/mol]"), plt.ylabel("Pressure [Pa]")
        #        plt.ylim(min(Plist)/2,np.abs(min(Plist))/2)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
    except Exception:
        logger.error("Matplotlib package is not installed, could not plot")


def calc_saturation_properties(
    T, xi, Eos, density_opts={}, tol=1e-6, Pconverged=1, **kwargs
):
    r"""
    Computes the saturated pressure, gas and liquid densities for a single component system.
    
    Parameters
    ----------
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    tol : float, Optional, default=1e-6
        Tolerance to accept pressure value
    Pconverged : float, Optional, default=1.0
        If the pressure is negative (under tension), we search from a value just above vacuum

    Returns
    -------
    Psat : float
        [Pa] Saturation pressure given system information
    rhov : float
        [mol/:math:`m^3`] Density of vapor at saturation pressure
    rhol : float
        [mol/:math:`m^3`] Density of liquid at saturation pressure
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_saturation_properties' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    if np.count_nonzero(xi) != 1:
        if np.count_nonzero(xi > 0.1) != 1:
            raise ValueError(
                "Multiple components have compositions greater than 10%, check code for source"
            )
        else:
            ind = np.where((xi > 0.1) == True)[0]
            raise ValueError(
                "Multiple components have compositions greater than 0. Do you mean to obtain the saturation pressure of {} with a mole fraction of {}?".format(
                    Eos.beads[ind], xi[ind]
                )
            )

    vlist, Plist = pressure_vs_volume_arrays(T, xi, Eos, **density_opts)
    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist)

    if not extrema or len(extrema) < 2 or np.any(np.isnan(roots)):
        logger.warning("    The component is above its critical point")
        Psat, rhol, rhov = np.nan, np.nan, np.nan

    else:
        ind_Pmin1 = np.argwhere(np.diff(Plist) > 0)[0][0]
        ind_Pmax1 = np.argmax(Plist[ind_Pmin1:]) + ind_Pmin1

        Pmaxsearch = Plist[ind_Pmax1]
        Pminsearch = max(Pconverged, np.amin(Plist[ind_Pmin1:ind_Pmax1]))

        # Using computed Psat find the roots in the maxwell construction to give liquid (first root) and vapor (last root) densities
        Psat = spo.minimize_scalar(
            objective_saturation_pressure,
            args=(Plist, vlist),
            bounds=(Pminsearch, Pmaxsearch),
            method="bounded",
        )
        Psat = Psat.x
        obj_value = objective_saturation_pressure(Psat, Plist, vlist)

        Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist - Psat)
        #      pressure_vs_volume_plot(vlist, Plist, Pvspline, markers=extrema)

        if obj_value < tol:
            logger.debug(
                "    Psat found: {} Pa, obj value: {}, with {} roots and {} extrema".format(
                    Psat, obj_value, np.size(roots), np.size(extrema)
                )
            )

            if len(roots) == 2:
                slope, yroot = np.polyfit(vlist[-4:], Plist[-4:] - Psat, 1)
                vroot = -yroot / slope
                if vroot < 0.0:
                    vroot = np.finfo(float).eps
                rho_tmp = spo.minimize(
                    pressure_spline_error,
                    1.0 / vroot,
                    args=(Psat, T, xi, Eos),
                    bounds=[(1.0 / (vroot * 1e2), 1.0 / (1.1 * roots[-1]))],
                )
                roots = np.append(roots, [1.0 / rho_tmp.x])

            rhol = 1.0 / roots[0]
            rhov = 1.0 / roots[2]

        else:
            logger.warning(
                "    Psat NOT found: {} Pa, obj value: {}, consider decreasing 'pressure_min' option in density_opts".format(
                    Psat, obj_value
                )
            )
            Psat, rhol, rhov = np.nan, np.nan, np.nan

    tmpv, _, _ = calc_vapor_fugacity_coefficient(
        Psat, T, xi, Eos, density_opts=density_opts
    )
    tmpl, _, _ = calc_liquid_fugacity_coefficient(
        Psat, T, xi, Eos, density_opts=density_opts
    )
    logger.debug("    phiv: {}, phil: {}".format(tmpv, tmpl))

    return Psat, rhol, rhov


def objective_saturation_pressure(shift, Pv, vlist):
    r"""
    Objective function used to calculate the saturation pressure. Note that if 

    Parameters
    ----------
    shift : float
        [Pa] Guess in Psat value used to translate the pressure vs. specific volume curve
    Pv : numpy.ndarray
        [Pa] Pressure associated with specific volume of system with given temperature and composition
    vlist : numpy.ndarray
        [mol/:math:`m^3`] Specific volume array. Length depends on values in density_opts passed to :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    obj_value : float
        Output of objective function, the addition of the positive area between first two roots, and negative area between second and third roots, quantity squared.

    """

    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Pv - shift)

    if len(roots) >= 3:
        a = Pvspline.integral(roots[0], roots[1])
        b = Pvspline.integral(roots[1], roots[2])
    elif len(roots) == 2:
        a = Pvspline.integral(roots[0], roots[1])
        # If the curve hasn't decayed to 0 yet, estimate the remaining area as a triangle. This isn't super accurate but we are just using the saturation pressure to get started.
        slope, yroot = np.polyfit(vlist[-4:], Pv[-4:] - shift, 1)
        b = (
            Pvspline.integral(roots[1], vlist[-1])
            + (Pv[-1] - shift) * (-yroot / slope - vlist[-1]) / 2
        )
        # raise ValueError("Pressure curve only has two roots. If the curve hasn't fully decayed, either increase maximum specific volume or decrease 'pressure_min' in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`.")
    elif np.any(np.isnan(roots)):
        raise ValueError(
            "Pressure curve without cubic properties has wrongly been accepted. Try decreasing pressure."
        )
    else:
        raise ValueError(
            "Pressure curve without cubic properties has wrongly been accepted. Try decreasing min_density_fraction"
        )
    # pressure_vs_volume_plot(vlist, Pv-shift, Pvspline, markers=extrema)

    return (a + b) ** 2


def calc_vapor_density(P, T, xi, Eos, density_opts={}, **kwargs):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    rhov : float
        [mol/:math:`m^3`] Density of vapor at system pressure
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_vapor_density' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    vlist, Plist = pressure_vs_volume_arrays(T, xi, Eos, **density_opts)
    Plist = Plist - P
    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist)

    logger.debug("    Find rhov: P {} Pa, roots {} m^3/mol".format(P, roots))

    flag_NoOpt = False
    l_roots = len(roots)
    if np.any(np.isnan(roots)):
        rho_tmp = np.nan
        flag = 3
        logger.warning(
            "    Flag 3: The T and yi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(
                T, xi
            )
        )
    elif l_roots == 0:
        if Pvspline(1 / vlist[-1]) < 0:
            try:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    1 / vlist[0],
                    args=(P, T, xi, Eos),
                    bounds=(
                        np.finfo("float").eps,
                        Eos.density_max(xi, T, maxpack=0.99),
                    ),
                )
                rho_tmp = rho_tmp.x
                if not len(extrema):
                    flag = 2
                    logger.debug(
                        "    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(
                            T, xi
                        )
                    )
                else:
                    flag = 1
                    logger.debug(
                        "    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(
                            T, xi
                        )
                    )
            except Exception:
                rho_tmp = np.nan
                flag = 3
                logger.warning(
                    "    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure, without density greater than max, {}".format(
                        T, xi, Eos.density_max(xi, T, maxpack=0.99)
                    )
                )
            flag_NoOpt = True
        elif min(Plist) + P > 0:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot / slope
            try:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    1 / vroot,
                    args=(P, T, xi, Eos),
                    bounds=(np.finfo("float").eps, 1.0 / (1.1 * roots[-1])),
                )
                rho_tmp = rho_tmp.x
                flag = 0
            except Exception:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug(
                    "    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(
                        T, xi
                    )
                )
            else:
                logger.debug(
                    "    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(
                        T, xi
                    )
                )
        else:
            logger.warning(
                "    Flag 3: The T and yi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(
                    T, xi
                )
            )
            flag = 3
            rho_tmp = np.nan
    elif l_roots == 1:
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(
                    T, xi
                )
            )
        elif (Pvspline(roots[0]) + P) > (Pvspline(max(extrema)) + P):
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 1: The T and yi, {} {}, combination produces a liquid at this pressure".format(
                    T, xi
                )
            )
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(
                    T, xi
                )
            )
    elif l_roots == 2:
        if (Pvspline(roots[0]) + P) < 0.0:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 1: This T and yi, {} {}, combination produces a liquid under tension at this pressure".format(
                    T, xi
                )
            )
        else:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot / slope
            try:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    1 / vroot,
                    args=(P, T, xi, Eos),
                    bounds=(np.finfo("float").eps, 1.0 / (1.1 * roots[-1])),
                )
                rho_tmp = rho_tmp.x
                flag = 0
            except Exception:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug(
                    "    Flag 2: The T and yi, {} {}, combination produces a critical fluid at this pressure".format(
                        T, xi
                    )
                )
            else:
                logger.debug(
                    "    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(
                        T, xi
                    )
                )
    else:  # 3 roots
        logger.debug(
            "    Flag 0: This T and yi, {} {}, combination produces a vapor at this pressure.".format(
                T, xi
            )
        )
        rho_tmp = 1.0 / roots[2]
        flag = 0

    if flag in [0, 2]:  # vapor or critical fluid
        tmp = [rho_tmp * 0.99, rho_tmp * 1.01]
        if rho_tmp * 1.01 > Eos.density_max(xi, T, maxpack=0.99):
            tmp[1] = Eos.density_max(xi, T, maxpack=0.99)

        if (
            pressure_spline_error(tmp[0], P, T, xi, Eos)
            * pressure_spline_error(tmp[1], P, T, xi, Eos)
        ) < 0:
            rho_tmp = spo.brentq(
                pressure_spline_error,
                tmp[0],
                tmp[1],
                args=(P, T, xi, Eos),
                rtol=0.0000001,
            )
        else:
            if Plist[0] < 0:
                logger.warning(
                    "    Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(
                        tmp
                    )
                )
            elif not flag_NoOpt:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    rho_tmp,
                    args=(P, T, xi, Eos),
                    bounds=(
                        np.finfo("float").eps,
                        Eos.density_max(xi, T, maxpack=0.99),
                    ),
                )
                rho_tmp = rho_tmp.x

    logger.debug("    Vapor Density: {} mol/m^3, flag {}".format(rho_tmp, flag))

    # pressure_vs_volume_plot(vlist, Plist, Pvspline, markers=extrema)

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
    return rho_tmp, flag


def calc_liquid_density(P, T, xi, Eos, density_opts={}, **kwargs):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    rhol : float
        [mol/:math:`m^3`] Density of liquid at system pressure
    flag : int
        A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_liquid_density' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    # Get roots and local minima and maxima
    vlist, Plist = pressure_vs_volume_arrays(T, xi, Eos, **density_opts)
    Plist = Plist - P
    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist)

    logger.debug("    Find rhol: P {} Pa, roots {} m^3/mol".format(P, str(roots)))
    flag_NoOpt = False

    if extrema:
        if len(extrema) == 1:
            logger.warning(
                "    One extrema at {}, assume weird minima behavior. Check your parameters.".format(
                    1 / extrema[0]
                )
            )

    # Assess roots, what is the liquid density
    l_roots = len(roots)
    if np.any(np.isnan(roots)):
        rho_tmp = np.nan
        flag = 3
        logger.warning(
            "    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(
                T, xi
            )
        )
    elif l_roots == 0:
        if Pvspline(1 / vlist[-1]):
            try:
                bounds = (1 / vlist[0], Eos.density_max(xi, T, maxpack=0.99))
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    np.mean(bounds),
                    args=(P, T, xi, Eos),
                    bounds=bounds,
                )
                rho_tmp = rho_tmp.x
                if not len(extrema):
                    flag = 2
                    logger.debug(
                        "    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(
                            T, xi
                        )
                    )
                else:
                    flag = 1
                    logger.debug(
                        "    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(
                            T, xi
                        )
                    )
            except Exception:
                rho_tmp = np.nan
                flag = 3
                logger.warning(
                    "    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure, without density greater than max, {}".format(
                        T, xi, Eos.density_max(xi, T, maxpack=0.99)
                    )
                )
            flag_NoOpt = True
        elif min(Plist) + P > 0:
            slope, yroot = np.polyfit(vlist[-4:], Plist[-4:], 1)
            vroot = -yroot / slope
            try:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    1.0 / vroot,
                    args=(P, T, xi, Eos),
                    bounds=(np.finfo("float").eps, 1.0 / (1.1 * roots[-1])),
                )
                rho_tmp = rho_tmp.x
                flag = 0
            except Exception:
                rho_tmp = np.nan
                flag = 4

            if not len(extrema):
                logger.debug(
                    "    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(
                        T, xi
                    )
                )
            else:
                logger.debug(
                    "    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(
                        T, xi
                    )
                )
        else:
            flag = 3
            logger.error(
                "    Flag 3: The T and xi, {} {}, won't produce a fluid (vapor or liquid) at this pressure".format(
                    str(T), str(xi)
                )
            )
            rho_tmp = np.nan
            # pressure_vs_volume_plot(vlist, Plist, Pvspline, markers=extrema)
    elif l_roots == 2:  # 2 roots
        if (Pvspline(roots[0]) + P) < 0.0:
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 1: This T and xi, {} {}, combination produces a liquid under tension at this pressure".format(
                    T, xi
                )
            )
        else:  # There should be three roots, but the values of specific volume don't go far enough to pick up the last one
            flag = 1
            rho_tmp = 1.0 / roots[0]
    elif l_roots == 1:  # 1 root
        if not len(extrema):
            flag = 2
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 2: The T and xi, {} {}, combination produces a critical fluid at this pressure".format(
                    T, xi
                )
            )
        elif (Pvspline(roots[0]) + P) > (Pvspline(max(extrema)) + P):
            flag = 1
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(
                    T, xi
                )
            )
        elif len(extrema) > 1:
            flag = 0
            rho_tmp = 1.0 / roots[0]
            logger.debug(
                "    Flag 0: This T and xi, {} {}, combination produces a vapor at this pressure. Warning! approaching critical fluid".format(
                    T, xi
                )
            )
    else:  # 3 roots
        rho_tmp = 1.0 / roots[0]
        flag = 1
        logger.debug(
            "    Flag 1: The T and xi, {} {}, combination produces a liquid at this pressure".format(
                T, xi
            )
        )

    if flag in [1, 2]:  # liquid or critical fluid
        tmp = [rho_tmp * 0.99, rho_tmp * 1.01]
        P_tmp = [
            pressure_spline_error(tmp[0], P, T, xi, Eos),
            pressure_spline_error(tmp[1], P, T, xi, Eos),
        ]
        if (P_tmp[0] * P_tmp[1]) < 0:
            rho_tmp = spo.brentq(
                pressure_spline_error, tmp[0], tmp[1], args=(P, T, xi, Eos), rtol=1e-7
            )
        else:
            if P_tmp[0] < 0:
                logger.warning(
                    "    Density value could not be bounded with (rhomin,rhomax), {}. Using approximate density value".format(
                        tmp
                    )
                )
            elif not flag_NoOpt:
                rho_tmp = spo.least_squares(
                    pressure_spline_error,
                    rho_tmp,
                    args=(P, T, xi, Eos),
                    bounds=(
                        np.finfo("float").eps,
                        Eos.density_max(xi, T, maxpack=0.99),
                    ),
                )
                rho_tmp = rho_tmp.x[0]
    logger.debug("    Liquid Density: {} mol/m^3, flag {}".format(rho_tmp, flag))

    # Flag: 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    return rho_tmp, flag


def pressure_spline_error(rho, Pset, T, xi, Eos):
    """
    Calculate difference between set point pressure and computed pressure for a given density. 

    Used to ensure an accurate value from the EOS rather than an estimate from a spline.
    
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    
    Returns
    -------
    pressure_spline_error : float
        [Pa] Difference in set pressure and predicted pressure given system conditions.
    """

    Pguess = Eos.pressure(rho, T, xi)

    return Pguess - Pset


def calc_vapor_fugacity_coefficient(P, T, yi, Eos, density_opts={}, **kwargs):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    phiv : float
        Fugacity coefficient of vapor at system pressure
    rhov : float
        [mol/:math:`m^3`] Density of vapor at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_vapor_fugacity_coefficient' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    rhov, flagv = calc_vapor_density(P, T, yi, Eos, density_opts)
    if flagv == 4:
        phiv = np.ones_like(yi)
        rhov = 0.0
        logger.info("    rhov set to 0.")
    elif flagv == 3:
        phiv = np.array([np.nan, np.nan])
    else:
        phiv = Eos.fugacity_coefficient(P, np.array([rhov]), yi, T)

    return phiv, rhov, flagv


def calc_liquid_fugacity_coefficient(P, T, xi, Eos, density_opts={}, **kwargs):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    phil : float
        Fugacity coefficient of liquid at system pressure
    rhol : float
        [mol/:math:`m^3`] Density of liquid at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true.
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_liquid_fugacity_coefficient' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    rhol, flagl = calc_liquid_density(P, T, xi, Eos, density_opts)
    if flagl == 3:
        phil = np.array([np.nan, np.nan])
    else:
        phil = Eos.fugacity_coefficient(P, np.array([rhol]), xi, T)

    return phil, rhol, flagl


def calc_new_mole_fractions(phase_1_mole_fraction, phil, phiv, phase=None):
    r"""

    Calculate the alternative phase composition given the composition and fugacity coefficients of one phase, and the fugacity coefficients of the target phase.
    
    Parameters
    ----------
    phase_1_mole_fraction : numpy.ndarray
        Mole fraction of each component, sum(mole fraction) must equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    phiv : float
        Fugacity coefficient of vapor at system pressure
    phase : str, default=None
        Use either 'vapor' or 'liquid' to define the mole fraction **being computed**. Default is None and it will fail to ensure the user specifies the correct phase

    Returns
    -------
    phase_2_mole_fraction : numpy.ndarray
        Mole fraction of each component computed from fugacity coefficients, sum(xi) should equal 1.0 when the solution is found, but the resulting values may not during an equilibrium calculation (e.g. bubble point).
    """

    if phase == None or phase not in ["vapor", "liquid"]:
        raise ValueError(
            "The user must specify the desired mole fraction as either 'vapor' or 'liquid'."
        )

    if np.sum(phase_1_mole_fraction) != 1.0:
        raise ValueError("Given mole fractions must add up to one.")

    if np.any(np.isnan(phiv)):
        raise ValueError("Vapor fugacity coefficients should not be NaN")

    if np.any(np.isnan(phil)):
        raise ValueError("Liquid fugacity coefficients should not be NaN")

    phase_2_mole_fraction = np.zeros(len(phase_1_mole_fraction))
    ind = np.where(phase_1_mole_fraction != 0.0)[0]
    if phase == "vapor":
        for i in ind:
            phase_2_mole_fraction[i] = phase_1_mole_fraction[i] * phil[i] / phiv[i]
    elif phase == "liquid":
        for i in ind:
            phase_2_mole_fraction[i] = phase_1_mole_fraction[i] * phiv[i] / phil[i]

    return phase_2_mole_fraction


def equilibrium_objective(phase_1_mole_fraction, phil, phiv, phase=None):
    r"""

    Computes the objective value used to determine equilibrium between phases. sum(phase_1_mole_fraction * phase_1_phi / phase_2_phi ) - 1.0, where `phase` is phase 2.
    
    Parameters
    ----------
    phase_1_mole_fraction : numpy.ndarray
        Mole fraction of each component, sum(mole fraction) must equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    phiv : float
        Fugacity coefficient of vapor at system pressure
    phase : str, default=None
        Use either 'vapor' or 'liquid' to define the mole fraction **being computed**. Default is None and it will fail to ensure the user specifies the correct phase

    Returns
    -------
    objective_value : numpy.ndarray
        Objective value indicating how close to equilibrium we are
    """

    if phase == None or phase not in ["vapor", "liquid"]:
        raise ValueError(
            "The user must specify the desired mole fraction as either 'vapor' or 'liquid'."
        )
    if np.sum(phase_1_mole_fraction) != 1.0:
        raise ValueError("Given mole fractions must add up to one.")

    if np.any(np.isnan(phiv)):
        raise ValueError("Vapor fugacity coefficients should not be NaN")

    if np.any(np.isnan(phil)):
        raise ValueError("Liquid fugacity coefficients should not be NaN")

    if phase == "vapor":
        objective_value = float((np.nansum(phase_1_mole_fraction * phil / phiv)) - 1.0)
    elif phase == "liquid":
        objective_value = float((np.nansum(phase_1_mole_fraction * phiv / phil)) - 1.0)

    return objective_value


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
    y_new = np.array([y_old[np.where(np.array(x_old) == x)[0][0]] for x in x_new])

    return x_new, y_new


def calc_Prange_xi(
    T,
    xi,
    yi,
    Eos,
    density_opts={},
    Pmin=None,
    Pmax=None,
    maxiter=200,
    mole_fraction_options={},
    ptol=1e-2,
    xytol=0.01,
    maxfactor=2,
    minfactor=0.5,
    Pmin_allowed=100,
    **kwargs
):
    r"""
    Obtain min and max pressure values for bubble point calculation.

    The liquid mole fraction is set and the objective function at each of those values is of opposite sign.
    
    Parameters
    ----------
    T : float
        Temperature of the system [K]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    maxiter : float, Optional, default=200
        Maximum number of iterations in both the loop to find Pmin and the loop to find Pmax
    Pmin : float, Optional, default=1000.0
        [Pa] Minimum pressure in pressure range that restricts searched space.
    Pmax : float, Optional, default=100000
        If no local minima or maxima are identified for the liquid composition at this temperature, this value is used as an initial estimate of the maximum pressure range.
    Pmin_allowed : float, Optional, default=100
        Minimum allowed pressure in search, before looking for a super critical fluid
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm
    ptol : float, Optional, default=1e-2
        If two iterations in the search for the maximum pressure are within this tolerance, the search is discontinued
    xytol : float, Optional, default=0.01
        If the sum of absolute relative difference between the vapor and liquid mole fractions are less than this total, the pressure is assumed to be super critical and the maximum pressure is sought at a lower value.
    maxfactor : float, Optional, default=2
        Factor to multiply by the pressure if it is too low (produces liquid or positive objective value). Not used if an unfeasible maximum pressure is found to bound the problem (critical for NaN result).
    minfactor : float, Optional, default=0.5
        Factor to multiply by the minimum pressure if it is too high (produces critical value).

    Returns
    -------
    Prange : list
        List of min and max pressure range
    Pguess : float
        An interpolated guess in the equilibrium pressure from Prange
    """

    if len(kwargs) > 0:
        logger.debug(
            "'calc_Prange_xi' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _yi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = pressure_vs_volume_arrays(T, xi, Eos, **density_opts)
    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist)

    flag_hard_min = False
    if Pmin != None:
        flag_hard_min = True
        if gtb.isiterable(Pmin):
            Pmin = Pmin[0]
    elif len(extrema):
        Pmin = min(Pvspline(extrema))
        if Pmin < 0:
            Pmin = 1e3
    else:
        Pmin = 1e3

    flag_hard_max = False
    if Pmax != None:
        flag_hard_max = True
        if gtb.isiterable(Pmax):
            Pmax = Pmax[0]
    elif len(extrema):
        Pmax = max(Pvspline(extrema))
    else:
        Pmax = 1e5
    if Pmax < Pmin:
        Pmax = Pmin * maxfactor

    Prange = np.array([Pmin, Pmax])

    #################### Find Minimum Pressure and Objective Function Value ###############

    # Root of min from liquid curve is absolute minimum
    ObjRange = np.zeros(2)
    yi_range = yi

    flag_max = False
    flag_min = False
    flag_critical = False
    flag_liquid = False
    flag_vapor = False
    p = Prange[0]
    for z in range(maxiter):

        # Liquid properties
        phil, rhol, flagl = calc_liquid_fugacity_coefficient(
            p, T, xi, Eos, density_opts=density_opts
        )

        if any(np.isnan(phil)):
            logger.error("Estimated minimum pressure is too high.")
            flag_max = True
            flag_liquid = True
            ObjRange[1] = np.inf
            Prange[1] = p
            if flag_hard_min:
                p = (Prange[1] - Prange[0]) / 2 + Prange[0]
            else:
                p = minfactor * p
                if p < Prange[0]:
                    Prange[0] = p
                    ObjRange[0] = np.nan
            continue

        if flagl in [1, 2]:  # 'liquid' phase is as expected

            # Calculate vapor phase properties and obj value
            yi_range, phiv_min, flagv_min = calc_vapor_composition(
                yi_range,
                xi,
                phil,
                p,
                T,
                Eos,
                density_opts=density_opts,
                **mole_fraction_options
            )
            obj = equilibrium_objective(xi, phil, phiv_min, phase="vapor")

            if np.any(np.isnan(yi_range)):
                logger.info("Estimated minimum pressure produces NaN")
                flag_max = True
                flag_liquid = True
                Prange[1] = p
                ObjRange[1] = obj
                phiv_max, flagv_max = phiv_min, flagv_min
                p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]

            # If within tolerance of liquid mole fraction
            elif np.sum(np.abs(xi - yi_range) / xi) < xytol and flagv_min == 2:
                logger.info(
                    "Estimated minimum pressure reproduces xi: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                if (
                    flag_max or flag_hard_max
                ) and flag_liquid:  # If a liquid phase exists at a higher pressure, this must bound the lower pressure
                    flag_min = True
                    ObjRange[0] = obj
                    Prange[0] = p
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                    if np.abs(Prange[1] - Prange[0]) < ptol:
                        flag_critical = True
                        flag_max = False
                        ObjRange = [np.inf, np.inf]
                        Prange = [Pmin, Pmax]
                        if flag_hard_max:
                            p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                        else:
                            p = maxfactor * Pmin
                            if p > Prange[1]:
                                Prange[1] = p
                                ObjRange[1] = np.nan
                elif (
                    flag_min or flag_hard_min
                ) and flag_vapor:  # If the 'liquid' phase is vapor at a lower pressure, this must bound the upper pressure
                    flag_max = True
                    ObjRange[1] = obj
                    Prange[1] = p
                    phiv_max, flagv_max = phiv_min, flagv_min
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                elif (
                    flag_critical
                ):  # Couldn't find phase by lowering pressure, now raise it
                    ObjRange[0] = obj
                    Prange[0] = p
                    if flag_hard_max:
                        p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                    else:
                        p = maxfactor * p
                        if p > Prange[1]:
                            Prange[1] = p
                            ObjRange[1] = np.nan
                else:
                    flag_max = True
                    ObjRange[1] = obj
                    Prange[1] = p
                    phiv_max, flagv_max = phiv_min, flagv_min
                    if flag_hard_min:
                        p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                    else:
                        p = minfactor * p
                        if p < Prange[0]:
                            Prange[0] = p
                            ObjRange[0] = np.nan

                if p < Pmin_allowed:  # Less than a kPa and can't find phase, go up
                    flag_critical = True
                    flag_max = False
                    ObjRange = [np.inf, np.inf]
                    Prange = [Pmin, Pmax]
                    if flag_hard_max:
                        p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                    else:
                        p = maxfactor * Pmin
                        if p > Prange[1]:
                            Prange[1] = p
                            ObjRange[1] = np.nan

            # If 'vapor' phase is liquid or unattainable
            elif flagv_min not in [0, 2, 4]:
                logger.info(
                    "Estimated minimum pressure produces liquid: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                if flag_hard_min and p <= Pmin:
                    flag_critical = True
                    if flag_max:
                        flag_max = False

                flag_liquid = True
                if flag_critical:  # Looking for a super critical fluid
                    Prange[0] = p
                    ObjRange[0] = obj
                    flag_min = True
                    if flag_hard_max:
                        p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                    else:
                        p = p * maxfactor
                        if p > Prange[1]:
                            Prange[1] = p
                            ObjRange[1] = np.nan
                else:  # Looking for a vapor
                    Prange[1] = p
                    ObjRange[1] = obj
                    flag_max = True
                    phiv_max, flagv_max = phiv_min, flagv_min
                    if flag_min or flag_hard_min:
                        p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                    else:
                        p = p * minfactor
                        if p < Prange[0]:
                            Prange[0] = p
                            ObjRange[0] = np.nan

            # Found minimum pressure!
            elif obj > 0:
                logger.info(
                    "Found estimated minimum pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                Prange[0] = p
                ObjRange[0] = obj
                break
            elif obj < 0:
                logger.info(
                    "Estimated minimum pressure too high: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                flag_liquid = True
                flag_max = True
                ObjRange[1] = obj
                Prange[1] = p
                phiv_max, flagv_max = phiv_min, flagv_min
                if flag_min or flag_hard_min:
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                else:
                    p = p * minfactor
                    if p < Prange[0]:
                        Prange[0] = p
                        ObjRange[0] = np.nan
            else:
                raise ValueError(
                    "This shouldn't happen: xi {}, phil {}, flagl {}, yi {}, phiv {}, flagv {}, obj {}, flags: {} {} {}".format(
                        xi,
                        phil,
                        flagl,
                        yi_range,
                        phiv_min,
                        flagv_min,
                        obj,
                        flag_min,
                        flag_max,
                        flag_critical,
                    )
                )
        else:
            logger.info(
                "Estimated minimum pressure produced vapor as a 'liquid' phase: {}, Range {}".format(
                    p, Prange
                )
            )
            flag_vapor = True
            flag_min = True
            Prange[0] = p
            ObjRange[0] = np.nan
            if flag_max or flag_hard_max:
                p = (Prange[1] - Prange[0]) / 2 + Prange[0]
            else:
                p = maxfactor * Prange[0]

        if (
            (flag_hard_min or flag_min)
            and (flag_hard_max or flag_max)
            and (p < Prange[0] or p > Prange[1])
        ):
            # if (p < Prange[0] and Prange[0] != Prange[1]) or (flag_max and p > Prange[1]):
            p = (Prange[1] - Prange[0]) / 1 + Prange[0]

        if p <= 0.0:
            raise ValueError(
                "Pressure, {}, cannot be equal to or less than zero. Given composition, {}, and T {}".format(
                    p, xi, T
                )
            )

        if flag_hard_min and Pmin == p:
            raise ValueError(
                "In searching for the minimum pressure, the range {}, converged without a solution".format(
                    Prange
                )
            )

    if z == maxiter - 1:
        raise ValueError(
            "Maximum Number of Iterations Reached: Proper minimum pressure for liquid density could not be found"
        )

    # A flag value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas

    #################### Find Maximum Pressure and Objective Function Value ###############

    # Be sure guess in upper bound is larger than lower bound
    if Prange[1] <= Prange[0]:
        Prange[1] = Prange[0] * maxfactor
        ObjRange[1] == 0.0

    flag_min = (
        False
    )  # Signals that the objective value starts to increase again and we must go back
    p = Prange[1]
    Parray = [Prange[1]]
    ObjArray = [ObjRange[1]]
    for z in range(maxiter):

        # Liquid properties
        phil, rhol, flagl = calc_liquid_fugacity_coefficient(
            p, T, xi, Eos, density_opts=density_opts
        )

        if any(np.isnan(phil)):
            logger.info(
                "Liquid fugacity coefficient should not be NaN, pressure could be too high."
            )
            flag_max = True
            Prange[1] = p
            ObjRange[1] = obj
            p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
            continue

        # Calculate vapor phase properties and obj value
        yi_range, phiv_max, flagv_max = calc_vapor_composition(
            yi_range,
            xi,
            phil,
            p,
            T,
            Eos,
            density_opts=density_opts,
            **mole_fraction_options
        )
        obj = equilibrium_objective(xi, phil, phiv_max, phase="vapor")

        # If 'vapor' phase is a liquid
        if flagv_max not in [0, 2, 4] or np.any(np.isnan(yi_range)):
            logger.info(
                "New Maximum Pressure: {} isn't vapor, flag={}, Obj Func: {}, Range {}".format(
                    p, flagv_max, obj, Prange
                )
            )
            if flag_critical:  # looking for critical fluid
                Prange[0] = p
                ObjRange[0] = obj
                if flag_hard_max:
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                else:
                    p = p * maxfactor
                    if p > Prange[1]:
                        Prange[1] = p
                        ObjRange[1] = np.nan
            else:  # Looking for vapor phase
                flag_max = True
                Prange[1] = p
                ObjRange[1] = obj
                p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]

        # If 'liquid' composition is reproduced
        elif np.sum(np.abs(xi - yi_range) / xi) < xytol:  # If less than 2%
            logger.info(
                "Estimated Maximum Pressure Reproduces xi: {},  Obj. Func: {}".format(
                    p, obj
                )
            )
            flag_max = True
            ObjRange[1] = obj
            Prange[1] = p
            p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
        # Suitable objective value found
        elif obj < 0:
            logger.info(
                "New Max Pressure: {}, flag={}, Obj Func: {}, Range {}".format(
                    p, flagv_max, obj, Prange
                )
            )
            if Prange[1] < p:
                Prange[0] = Prange[1]
                ObjRange[0] = ObjRange[1]
            Prange[1] = p
            ObjRange[1] = obj
            logger.info("Got the pressure range!")
            slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
            intercept = ObjRange[1] - slope * Prange[1]
            Pguess = -intercept / slope
            flag_min = False
            break
        else:
            Parray.append(p)
            ObjArray.append(obj)

            # In an objective value "well"
            if (z > 0 and ObjArray[-1] > 1.1 * ObjArray[-2]) or flag_min:
                if not flag_min:
                    flag_min = True

                Prange[1] = p
                ObjRange[1] = obj

                logger.info(
                    "Maximum Pressure (if it exists) between Pressure: {} and Obj Range: {}".format(
                        Prange, ObjRange
                    )
                )

                P0 = np.mean(Prange)
                scale_factor = 10 ** (np.ceil(np.log10(P0)))
                args = (xi, T, Eos, density_opts, mole_fraction_options, scale_factor)
                p = gtb.solve_root(
                    lambda x, xi, T, Eos, density_opts, mole_fraction_options, scale_factor: objective_bubble_pressure(
                        x * scale_factor,
                        xi,
                        T,
                        Eos,
                        density_opts,
                        mole_fraction_options,
                    ),
                    args=args,
                    x0=P0 / scale_factor,
                    method="TNC",
                    bounds=Prange / scale_factor,
                )
                p = p[0] * scale_factor
                obj = objective_bubble_pressure(
                    p,
                    xi,
                    T,
                    Eos,
                    density_opts=density_opts,
                    mole_fraction_options=mole_fraction_options,
                )
                logger.info(
                    "New Max Pressure: {}, Obj Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )

                if p < 0:
                    parray = np.linspace(Prange[0], Prange[1], 20)
                    obj_array = []
                    for ptmp in parray:
                        obj_tmp = objective_dew_pressure(
                            ptmp,
                            yi,
                            T,
                            Eos,
                            density_opts=density_opts,
                            mole_fraction_options=mole_fraction_options,
                        )
                        obj_array.append(obj_tmp)
                    spline = interpolate.Akima1DInterpolator(parray, obj_array)
                    p_min = spline.derivative().roots()
                    if len(p_min) > 1:
                        obj_tmp = []
                        for p_min_tmp in p_min:
                            obj_tmp.append(
                                objective_bubble_pressure(
                                    p_min_tmp, xi, T, Eos, density_opts=density_opts
                                )
                            )
                        p_min = p_min[obj_tmp == np.nanmin(obj_tmp)]
                    elif len(p_min) == 0:
                        logger.error(
                            "Could not find minimum in pressure range:\n    Pressure: {}\n    Obj Value: {}".format(
                                parray, obj_array
                            )
                        )
                    p = p_min
                    obj = objective_bubble_pressure(
                        p, xi, T, Eos, density_opts=density_opts
                    )
                    logger.info(
                        "New Max Pressure: {}, Obj Func: {}, Range {}".format(
                            p, obj, Prange
                        )
                    )

                if obj > 0:
                    Prange[1] = p
                    ObjRange[1] = obj
                    logger.info("Got the pressure range!")
                    slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                    intercept = ObjRange[1] - slope * Prange[1]
                    Pguess = -intercept / slope
                    flag_min = False
                else:
                    logger.error(
                        "Could not find maximum in pressure range:\n    Pressure range {} best {}\n    Obj Value range {} best {}".format(
                            Prange, p, ObjRange, obj
                        )
                    )
                break

            elif flag_max:
                logger.info(
                    "New Minimum Pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                Prange[0] = p
                ObjRange[0] = obj
                p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
            else:
                logger.info(
                    "New Maximum Pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                if not flag_hard_max:
                    if Prange[1] < p:
                        Prange[0] = Prange[1]
                        ObjRange[0] = ObjRange[1]
                    Prange[1] = p
                    ObjRange[1] = obj
                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                intercept = ObjRange[1] - slope * Prange[1]

                if flag_hard_max:
                    p = (Prange[1] - Prange[0]) * np.random.rand(1)[0] + Prange[0]
                else:
                    p = np.nanmax([-intercept / slope, maxfactor * Prange[1]])

        if p <= 0.0:
            raise ValueError(
                "Pressure, {}, cannot be equal to or less than zero. Given composition, {}, and T {}".format(
                    p, xi, T
                )
            )

        if np.abs(Prange[1] - Prange[0]) < ptol:
            raise ValueError(
                "In searching for the minimum pressure, the range {}, converged without a solution".format(
                    Prange
                )
            )

    if z == maxiter - 1 or flag_min:
        if flag_min:
            logger.error(
                "Cannot reach objective value of zero. Final Pressure: {}, Obj. Func: {}".format(
                    p, obj
                )
            )
        else:
            logger.error(
                "Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress"
            )
        Prange = np.array([np.nan, np.nan])
        Pguess = np.nan
    else:
        logger.info(
            "[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange), str(ObjRange))
        )
        logger.info("Initial guess in pressure: {} Pa".format(Pguess))

        _yi_global = yi_range

    return Prange, Pguess


def calc_Prange_yi(
    T,
    xi,
    yi,
    Eos,
    density_opts={},
    mole_fraction_options={},
    Pmin=None,
    Pmax=None,
    Pmin_allowed=100,
    maxiter=200,
    ptol=1e-2,
    xytol=0.01,
    maxfactor=2,
    minfactor=0.5,
    **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    maxiter : float, Optional, default=200
        Maximum number of iterations in both the loop to find Pmin and the loop to find Pmax
    Pmin : float, Optional, default=1000.0
        [Pa] Minimum pressure in pressure range that restricts searched space. Used if local minimum isn't available for pressure curve for vapor composition.
    Pmax : float, Optional, default=100000
        If no local minima or maxima are identified for the liquid composition at this temperature, this value is used as an initial estimate of the maximum pressure range.
    Pmin_allowed : float, Optional, default=100
        Minimum allowed pressure in search, before looking for a super critical fluid
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm
    ptol : float, Optional, default=1e-2
        If two iterations in the search for the maximum pressure are within this tolerance, the search is discontinued
    xytol : float, Optional, default=0.01
        If the sum of absolute relative difference between the vapor and liquid mole fractions are less than this total, the pressure is assumed to be super critical and the maximum pressure is sought at a lower value.
    maxfactor : float, Optional, default=2
        Factor to multiply by the pressure if it is too low (produces liquid or positive objective value). Not used if an unfeasible maximum pressure is found to bound the problem (critical for NaN result).
    minfactor : float, Optional, default=0.5
        Factor to multiply by the minimum pressure if it is too high (produces critical value).

    Returns
    -------
    Prange : list
        List of min and max pressure range
    Pguess : float
        An interpolated guess in the equilibrium pressure from Prange

    """

    if len(kwargs) > 0:
        logger.debug(
            "'calc_Prange_yi' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _xi_global

    # Guess a range from Pmin to the local max of the liquid curve
    vlist, Plist = pressure_vs_volume_arrays(T, yi, Eos, **density_opts)
    Pvspline, roots, extrema = pressure_vs_volume_spline(vlist, Plist)

    # Calculation the highest pressure possible
    flag_hard_min = False
    if Pmin != None:
        flag_hard_min = True
        if gtb.isiterable(Pmin):
            Pmin = Pmin[0]
    elif len(extrema):
        Pmin = min(Pvspline(extrema))
        if Pmin < 0:
            Pmin = 1e3
    else:
        Pmin = 1e3

    flag_hard_max = False
    if Pmax != None:
        flag_hard_max = True
        if gtb.isiterable(Pmax):
            Pmax = Pmax[0]
    elif len(extrema):
        Pmax = max(Pvspline(extrema))
    else:
        Pmax = 1e5
    if Pmax < Pmin:
        Pmax = Pmin * maxfactor

    Prange = np.array([Pmin, Pmax])

    ObjRange = np.zeros(2)
    xi_range = xi

    #################### Find Minimum Pressure and Objective Function Value ###############

    flag_min = False
    flag_max = False
    flag_critical = False
    flag_liquid = False
    flag_vapor = False
    p = Prange[0]
    for z in range(maxiter):

        # Vapor properties
        phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
            p, T, yi, Eos, density_opts=density_opts
        )
        if any(np.isnan(phiv)):
            logger.error("Estimated minimum pressure is too high.")
            flag_max = True
            flag_liquid = True
            ObjRange[1] = np.inf
            Prange[1] = p
            if flag_hard_min:
                p = (Prange[1] - Prange[0]) / 2 + Prange[0]
            else:
                p = minfactor * p
                if p < Prange[0]:
                    Prange[0] = p
                    ObjRange[0] = np.nan
            continue

        if flagv in [0, 2, 4]:

            # Calculate the liquid phase properties
            xi_range, phil_min, flagl_min = calc_liquid_composition(
                xi_range,
                yi,
                phiv,
                p,
                T,
                Eos,
                density_opts=density_opts,
                **mole_fraction_options
            )
            obj = equilibrium_objective(yi, phil_min, phiv, phase="liquid")

            if np.any(np.isnan(xi_range)):
                logger.info("Estimated Minimum Pressure produces NaN")
                flag_max = True
                flag_vapor = True
                Prange[1] = p
                ObjRange[1] = obj
                phiv_max, flagv_max = phiv_min, flagv_min
                if flag_hard_min:
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                else:
                    p = p * minfactor

            elif (
                np.sum(np.abs(yi - xi_range) / yi) < xytol and flagl_min == 2
            ):  # If within 2% of liquid mole fraction
                logger.info(
                    "Estimated Minimum Pressure Reproduces yi: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )

                if (
                    flag_critical
                ):  # Couldn't find phase by lowering pressure, now raise it
                    ObjRange[0] = obj
                    Prange[0] = p
                    if flag_hard_max:
                        p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                    else:
                        p = maxfactor * p
                        if p > Prange[1]:
                            Prange[1] = p
                            ObjRange[1] = np.nan
                else:
                    flag_max = True
                    ObjRange[1] = obj
                    Prange[1] = p
                    phil_max, flagl_max = phil_min, flagl_min
                    if flag_min or flag_hard_min:
                        p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                    else:
                        p = minfactor * p

                if p < Pmin_allowed:  # Less than a kPa and can't find phase, go up
                    flag_critical = True
                    flag_max = False
                    ObjRange = [np.inf, np.inf]
                    Prange = [Pmin, Pmax]
                    if flag_hard_max:
                        p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
                    else:
                        p = maxfactor * Pmin
                        if p > Prange[1]:
                            Prange[1] = p
                            ObjRange[1] = np.nan
            elif obj < 0:
                Prange[0] = p
                ObjRange[0] = obj
                logger.info(
                    "Obtained estimated Minimum Pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                break
            elif obj > 0:
                flag_max = True
                logger.info(
                    "Estimated Minimum Pressure too High: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                ObjRange[1] = obj
                Prange[1] = p
                phil_max, flagl_max = phil_min, flagl_min
                p = (Prange[1] - Prange[0]) * minfactor + Prange[0]
        else:
            logger.info(
                "Estimated Minimum Pressure Produced Liquid instead of Vapor Phase: {}, Range {}".format(
                    p, Prange
                )
            )
            if flag_hard_min and p <= Pmin:
                flag_critical = True
                if flag_max:
                    flag_max = False

            if flag_critical:  # Looking for a super critical fluid
                Prange[0] = p
                ObjRange[0] = obj
                flag_min = True
                if flag_hard_max:
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                else:
                    p = p * maxfactor
                    if p > Prange[1]:
                        Prange[1] = p
                        ObjRange[1] = np.nan
            else:  # Looking for a vapor
                Prange[1] = p
                ObjRange[1] = obj
                flag_max = True
                if flag_min or flag_hard_min:
                    p = (Prange[1] - Prange[0]) / 2 + Prange[0]
                else:
                    p = p * minfactor
                    if p < Prange[0]:
                        Prange[0] = p
                        ObjRange[0] = np.nan

        if Prange[0] > Prange[1]:
            if flag_max and not flag_min and not flag_hard_min:
                Prange[0] = minfactor * Prange[1]
                ObjRange[0] = ObjRange[1]
            elif not flag_hard_max:
                Prange[1] = maxfactor * Prange[0]
                ObjRange[1] = ObjRange[0]
            else:
                raise ValueError("Pmin should never be greater than Pmax")

        if (
            (flag_max or flag_hard_max)
            and (flag_min or flag_hard_min)
            and not Prange[0] <= p <= Prange[1]
        ):
            p = (Prange[1] - Prange[0]) * np.random.rand(1)[0] + Prange[0]

        if flag_hard_min and Pmin == p:
            raise ValueError(
                "In searching for the minimum pressure, the range {}, converged without a solution".format(
                    Prange
                )
            )

        if p <= 0.0:
            raise ValueError(
                "Pressure, {}, cannot be equal to or less than zero. Given composition, {}, and T {}, results in a supercritical value without a coexistent fluid.".format(
                    p, xi, T
                )
            )

    if z == maxiter - 1:
        raise ValueError(
            "Maximum Number of Iterations Reached: Proper minimum pressure for liquid density could not be found"
        )

    # Be sure guess in pressure is larger than lower bound
    if Prange[1] <= Prange[0]:
        Prange[1] = Prange[0] * 1.1
        if z == 0:
            ObjRange[1] == 0.0

    ## Check Pmax
    flag_sol = False
    flag_vapor = False
    flag_min = False
    p = Prange[1]
    Parray = [Prange[1]]
    ObjArray = [ObjRange[1]]
    for z in range(maxiter):
        # Calculate objective value
        phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
            p, T, yi, Eos, density_opts=density_opts
        )
        xi_range, phil, flagl = calc_liquid_composition(
            xi_range,
            yi,
            phiv,
            p,
            T,
            Eos,
            density_opts=density_opts,
            **mole_fraction_options
        )
        obj = equilibrium_objective(yi, phil, phiv, phase="liquid")

        if z == 0:
            ObjRange[1] = obj

        if flagv not in [0, 2, 4]:  # Ensure vapor is produced
            flag_vapor = True
            Prange[1] = p
            ObjRange[1] = obj
            logger.info(
                "New Max Pressure: {} doesn't produce vapor, flag={}, Obj Func: {}, Range {}".format(
                    Prange[1], flagv, ObjRange[1], Prange
                )
            )
            p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
        elif obj > 0:  # Check pressure range
            if Prange[1] < p:
                Prange[0] = Prange[1]
                ObjRange[0] = ObjRange[1]
            Prange[1] = p
            ObjRange[1] = obj
            logger.info(
                "New Max Pressure: {}, flag={}, Obj Func: {}, Range {}".format(
                    Prange[1], flagv, ObjRange[1], Prange
                )
            )
            logger.info("Got the pressure range!")
            slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
            intercept = ObjRange[1] - slope * Prange[1]
            Pguess = -intercept / slope
            flag_sol = True
            flag_min = False
            break
        elif flag_vapor:
            Prange[0] = p
            ObjRange[0] = obj
            p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
            logger.info(
                "New Max Pressure: {},  Obj. Func: {}, Range {}".format(
                    Prange[0], ObjRange[0], Prange
                )
            )
        else:
            Parray.append(p)
            ObjArray.append(obj)

            # In an objective value "well"
            if (z > 0 and ObjArray[-1] < 1.1 * ObjArray[-2]) or flag_min:
                if not flag_min:
                    flag_min = True

                Prange[1] = p
                ObjRange[1] = obj

                logger.info(
                    "Maximum Pressure (if it exists) between Pressure: {} and Obj Range: {}".format(
                        Prange, ObjRange
                    )
                )

                P0 = np.mean(Prange)
                scale_factor = 10 ** (np.ceil(np.log10(P0)))
                args = (yi, T, Eos, density_opts, mole_fraction_options, scale_factor)
                p = gtb.solve_root(
                    lambda x, yi, T, Eos, density_opts, mole_fraction_options, scale_factor: -objective_dew_pressure(
                        x * scale_factor,
                        yi,
                        T,
                        Eos,
                        density_opts,
                        mole_fraction_options,
                    ),
                    args=args,
                    x0=P0 / scale_factor,
                    method="TNC",
                    bounds=Prange / scale_factor,
                )
                p = p[0] * scale_factor
                obj = objective_dew_pressure(
                    p,
                    yi,
                    T,
                    Eos,
                    density_opts=density_opts,
                    mole_fraction_options=mole_fraction_options,
                )
                logger.info(
                    "New Max Pressure: {}, Obj Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )

                if p < 0:
                    parray = np.linspace(Prange[0], Prange[1], 20)
                    obj_array = []
                    for ptmp in parray:
                        obj_tmp = objective_dew_pressure(
                            ptmp,
                            yi,
                            T,
                            Eos,
                            density_opts=density_opts,
                            mole_fraction_options=mole_fraction_options,
                        )
                        obj_array.append(obj_tmp)
                    spline = interpolate.Akima1DInterpolator(parray, obj_array)
                    p_min = spline.derivative().roots()
                    if len(p_min) > 1:
                        obj_tmp = []
                        for p_min_tmp in p_min:
                            obj_tmp.append(
                                objective_bubble_pressure(
                                    p_min_tmp, xi, T, Eos, density_opts=density_opts
                                )
                            )
                        p_min = p_min[obj_tmp == np.nanmin(obj_tmp)]
                    elif len(p_min) == 0:
                        logger.error(
                            "Could not find minimum in pressure range:\n    Pressure: {}\n    Obj Value: {}".format(
                                parray, obj_array
                            )
                        )
                    p = p_min
                    obj = objective_bubble_pressure(
                        p, xi, T, Eos, density_opts=density_opts
                    )
                    logger.info(
                        "New Max Pressure: {}, Obj Func: {}, Range {}".format(
                            p, obj, Prange
                        )
                    )

                if obj > 0:
                    Prange[1] = p
                    ObjRange[1] = obj
                    logger.info("Got the pressure range!")
                    slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                    intercept = ObjRange[1] - slope * Prange[1]
                    Pguess = -intercept / slope
                    flag_min = False
                else:
                    logger.error(
                        "Could not find maximum in pressure range:\n    Pressure range {} best {}\n    Obj Value range {} best {}".format(
                            Prange, p, ObjRange, obj
                        )
                    )
                break

            elif flag_hard_max:
                logger.info(
                    "New Minimum Pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                Prange[0] = p
                ObjRange[0] = obj
                p = (Prange[1] - Prange[0]) / 2.0 + Prange[0]
            else:
                logger.info(
                    "New Maximum Pressure: {},  Obj. Func: {}, Range {}".format(
                        p, obj, Prange
                    )
                )
                if not flag_hard_max:
                    if Prange[1] < p:
                        Prange[0] = Prange[1]
                        ObjRange[0] = ObjRange[1]
                    Prange[1] = p
                    ObjRange[1] = obj
                slope = (ObjRange[1] - ObjRange[0]) / (Prange[1] - Prange[0])
                intercept = ObjRange[1] - slope * Prange[1]
                p = np.nanmax([-intercept / slope, maxfactor * Prange[1]])

    if z == maxiter - 1 or flag_min:
        if flag_min:
            logger.error(
                "Cannot reach objective value of zero. Final Pressure: {}, Obj. Func: {}".format(
                    p, obj
                )
            )
        else:
            logger.error(
                "Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress"
            )
        Prange = np.array([np.nan, np.nan])
        Pguess = np.nan
    elif flag_sol:
        logger.info(
            "[Pmin, Pmax]: {}, Obj. Values: {}".format(str(Prange), str(ObjRange))
        )
        logger.info("Initial guess in pressure: {} Pa".format(Pguess))
    else:
        logger.error(
            "Maximum Number of Iterations Reached: A change in sign for the objective function could not be found, inspect progress"
        )

        _xi_global = xi_range

    return Prange, Pguess


def calc_vapor_composition(
    yi,
    xi,
    phil,
    P,
    T,
    Eos,
    density_opts={},
    maxiter=50,
    tol=1e-6,
    tol_trivial=0.05,
    **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    maxiter : int, Optional, default=50
        Maximum number of iteration for both the outer pressure and inner vapor mole fraction loops
    tol : float, Optional, default=1e-6
        Tolerance in sum of predicted yi "mole numbers"
    tol_trivial : float, Optional, default=0.05
        If the vapor and liquid mole fractions are within this tolerance, search for a different composition
    kwargs : NA, Optional
        Other other keyword arguments for :func:`~despasito.thermodynamics.calc.find_new_yi`

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    phiv : float
        Fugacity coefficient of vapor at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    if np.any(np.isnan(phil)):
        raise ValueError(
            "Cannot obtain vapor mole fraction with fugacity coefficients of NaN"
        )

    global _yi_global

    yi_total = [np.sum(yi)]
    yi /= np.sum(yi)
    flag_check_vapor = True  # Make sure we only search for vapor compositions once
    flag_trivial_sol = (
        True
    )  # Make sure we only try to find alternative to trivial solution once
    logger.info("    Solve yi: P {}, T {}, xi {}, phil {}".format(P, T, xi, phil))

    for z in range(maxiter):

        yi_tmp = yi / np.sum(yi)

        # Try yi
        phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
            P, T, yi_tmp, Eos, density_opts=density_opts
        )

        if (
            any(np.isnan(phiv)) or flagv == 1
        ) and flag_check_vapor:  # If vapor density doesn't exist
            flag_check_vapor = False
            if all(yi_tmp != 0.0) and len(yi_tmp) == 2:
                logger.debug("    Composition doesn't produce a vapor, let's find one!")
                yi_tmp = find_new_yi(
                    P, T, phil, xi, Eos, density_opts=density_opts, **kwargs
                )
                flag_trivial_sol = False
                if np.any(np.isnan(yi_tmp)):
                    phiv, rhov, flagv = [np.nan, np.nan, 3]
                    yinew = yi_tmp
                    break
                else:
                    phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
                        P, T, yi_tmp, Eos, density_opts=density_opts
                    )
                    yinew = calc_new_mole_fractions(xi, phil, phiv, phase="vapor")
            else:
                logger.debug(
                    "    Composition doesn't produce a vapor, we need a function to search compositions for more than two components."
                )
                yinew = yi
        elif np.sum(np.abs(xi - yi_tmp) / xi) < tol_trivial and flag_trivial_sol:
            flag_trivial_sol = False
            if all(yi_tmp != 0.0) and len(yi_tmp) == 2:
                logger.debug(
                    "    Composition produces trivial solution, let's find a different one!"
                )
                yi_tmp = find_new_yi(
                    P, T, phil, xi, Eos, density_opts=density_opts, **kwargs
                )
                flag_check_vapor = False
            else:
                logger.debug(
                    "    Composition produces trivial solution, using random guess to reset"
                )
                yi_tmp = np.random.rand(len(yi_tmp))
                yi_tmp /= np.sum(yi_tmp)

            if np.any(np.isnan(yi_tmp)):
                phiv, rhov, flagv = [np.nan, np.nan, 3]
                yinew = yi_tmp
                break
            else:
                phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
                    P, T, yi_tmp, Eos, density_opts=density_opts
                )
                yinew = calc_new_mole_fractions(xi, phil, phiv, phase="vapor")
        else:
            yinew = calc_new_mole_fractions(xi, phil, phiv, phase="vapor")

        yinew[np.isnan(yinew)] = 0.0
        yi2 = yinew / np.sum(yinew)
        phiv2, _, flagv2 = calc_vapor_fugacity_coefficient(
            P, T, yi2, Eos, density_opts=density_opts
        )

        if any(np.isnan(phiv)):
            phiv = np.nan
            logger.error(
                "Fugacity coefficient of vapor should not be NaN, pressure could be too high."
            )

        # Check for bouncing between values
        if len(yi_total) > 3:
            tmp1 = np.abs(np.sum(yinew) - yi_total[-2]) + np.abs(
                yi_total[-1] - yi_total[-3]
            )
            if tmp1 < np.abs(np.sum(yinew) - yi_total[-1]) and flagv != flagv2:
                logger.debug(
                    "    Composition bouncing between values, let's find the answer!"
                )
                bounds = np.sort([yi_tmp[0], yi2[0]])
                yi2, obj = bracket_bounding_yi(
                    P, T, phil, xi, Eos, bounds=bounds, density_opts=density_opts
                )
                phiv2, _, flagv2 = calc_vapor_fugacity_coefficient(
                    P, T, yi2, Eos, density_opts=density_opts
                )
                _yi_global = yi2
                logger.info(
                    "    Inner Loop Final (from bracketing bouncing values) yi: {}, Final Error on Smallest Fraction: {}".format(
                        yi2, obj
                    )
                )
                break

        logger.debug(
            "    yi guess {}, yi calc {}, phiv {}, flag {}".format(
                yi_tmp, yinew, phiv, flagv
            )
        )
        logger.debug(
            "    Old yi_total: {}, New yi_total: {}, Change: {}".format(
                yi_total[-1], np.sum(yinew), np.sum(yinew) - yi_total[-1]
            )
        )

        # Check convergence
        if abs(np.sum(yinew) - yi_total[-1]) < tol:
            ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp > 0]))[0]
            if np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp] < tol:
                _yi_global = yi2
                logger.info(
                    "    Inner Loop Final yi: {}, Final Error on Smallest Fraction: {}%".format(
                        yi2,
                        np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp] * 100,
                    )
                )
                break

        if z < maxiter - 1:
            yi_total.append(np.sum(yinew))
            yi = yinew

    ## If yi wasn't found in defined number of iterations
    ind_tmp = np.where(yi_tmp == min(yi_tmp[yi_tmp > 0.0]))[0]
    if flagv == 3:
        yi2 = yinew / np.sum(yinew)
        logger.info("    Could not converged mole fraction")
        phiv2 = np.full(len(yi_tmp), np.nan)
        flagv2 = np.nan
    elif z == maxiter - 1:
        yi2 = yinew / np.sum(yinew)
        tmp = np.abs(yi2[ind_tmp] - yi_tmp[ind_tmp]) / yi_tmp[ind_tmp]
        logger.warning(
            "    More than {} iterations needed. Error in Smallest Fraction: {}%".format(
                maxiter, tmp * 100
            )
        )
        if tmp > 0.1:  # If difference is greater than 10%
            yinew = find_new_yi(
                P, T, phil, xi, Eos, density_opts=density_opts, **kwargs
            )
            yi2 = yinew / np.sum(yinew)
        y1 = spo.least_squares(
            objective_find_yi,
            yi2[0],
            bounds=(0.0, 1.0),
            args=(P, T, phil, xi, Eos, density_opts),
        )
        yi = y1.x[0]
        yi2 = np.array([yi, 1 - yi])
        phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
            P, T, yi2, Eos, density_opts=density_opts
        )
        obj = objective_find_yi(yi2, P, T, phil, xi, Eos, density_opts=density_opts)
        logger.warning(
            "    Find yi with root algorithm, yi {}, obj {}".format(yi2, obj)
        )
        if obj > tol:
            logger.error("Could not converge mole fraction")
            phiv2 = np.full(len(yi_tmp), np.nan)
            flagv2 = 3

    return yi2, phiv2, flagv2


def calc_liquid_composition(
    xi,
    yi,
    phiv,
    P,
    T,
    Eos,
    density_opts={},
    maxiter=20,
    tol=1e-6,
    tol_trivial=0.05,
    **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    maxiter : int, Optional, default=20
        Maximum number of iteration for both the outer pressure and inner vapor mole fraction loops
    tol : float, Optional, default=1e-6
        Tolerance in sum of predicted xi "mole numbers"
    tol_trivial : float, Optional, default=0.05
        If the vapor and liquid mole fractions are within this tolerance, search for a different composition
    kwargs : dict, Optional
        Optional keywords for :func:`~despasito.thermodynamics.calc.find_new_xi`

    Returns
    -------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    phil : float
        Fugacity coefficient of liquid at system pressure
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true
    """

    global _xi_global

    if np.any(np.isnan(phiv)):
        raise ValueError(
            "Cannot obtain liquid mole fraction with fugacity coefficients of NaN"
        )

    xi /= np.sum(xi)
    xi_total = [np.sum(xi)]
    flag_check_liquid = True  # Make sure we only search for liquid compositions once
    flag_trivial_sol = (
        True
    )  # Make sure we only try to find alternative to trivial solution once
    logger.info("    Solve xi: P {}, T {}, yi {}, phiv {}".format(P, T, yi, phiv))

    for z in range(maxiter):

        xi_tmp = xi / np.sum(xi)

        # Try xi
        phil, rhol, flagl = calc_liquid_fugacity_coefficient(
            P, T, xi_tmp, Eos, density_opts=density_opts
        )

        if (any(np.isnan(phil)) or flagl in [0, 4]) and flag_check_liquid:
            flag_check_liquid = False
            if all(xi_tmp != 0.0) and len(xi_tmp) == 2:
                logger.debug(
                    "    Composition doesn't produce a liquid, let's find one!"
                )
                xi_tmp = find_new_xi(
                    P, T, phiv, yi, Eos, density_opts=density_opts, **kwargs
                )
                flag_trivial_sol = False
                if np.any(np.isnan(xi_tmp)):
                    phil, rhol, flagl = [np.nan, np.nan, 3]
                    xinew = xi_tmp
                    break
                else:
                    phil, rhol, flagl = calc_liquid_fugacity_coefficient(
                        P, T, xi_tmp, Eos, density_opts=density_opts
                    )
                    xinew = calc_new_mole_fractions(yi, phil, phiv, phase="liquid")
            else:
                logger.debug(
                    "    Composition doesn't produce a liquid, we need a function to search compositions for more than two components."
                )
                xinew = xi
        elif np.sum(np.abs(yi - xi_tmp) / yi) < tol_trivial and flag_trivial_sol:
            flag_trivial_sol = False
            if all(xi_tmp != 0.0) and len(xi_tmp) == 2:
                logger.debug(
                    "    Composition produces trivial solution, let's find a different one!"
                )
                xi_tmp = find_new_xi(
                    P, T, phiv, yi, Eos, density_opts=density_opts, **kwargs
                )
                flag_check_vapor = False
            else:
                logger.debug(
                    "    Composition produces trivial solution, using random guess to reset"
                )
                xi_tmp = np.random.rand(len(xi_tmp))
                xi_tmp /= np.sum(xi_tmp)

            if np.any(np.isnan(xi_tmp)):
                phil, rhol, flagl = [np.nan, np.nan, 3]
                xinew = xi_tmp
                break
            else:
                phil, rhol, flagl = calc_liquid_fugacity_coefficient(
                    P, T, xi_tmp, Eos, density_opts=density_opts
                )
                xinew = calc_new_mole_fractions(yi, phil, phiv, phase="liquid")
        else:
            xinew = calc_new_mole_fractions(yi, phil, phiv, phase="liquid")
        xinew[np.isnan(xinew)] = 0.0

        logger.debug(
            "    xi guess {}, xi calc {}, phil {}".format(
                xi_tmp, xinew / np.sum(xinew), phil
            )
        )
        logger.debug(
            "    Old xi_total: {}, New xi_total: {}, Change: {}".format(
                xi_total[-1], np.sum(xinew), np.sum(xinew) - xi_total[-1]
            )
        )

        # Check convergence
        if abs(np.sum(xinew) - xi_total[-1]) < tol:
            ind_tmp = np.where(xi_tmp == min(xi_tmp[xi_tmp > 0]))[0]
            xi2 = xinew / np.sum(xinew)
            if np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp] < tol:
                _xi_global = xi2
                logger.info(
                    "    Inner Loop Final xi: {}, Final Error on Smallest Fraction: {}%".format(
                        xi2,
                        np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp] * 100,
                    )
                )
                break

        if z < maxiter - 1:
            xi_total.append(np.sum(xinew))
            xi = xinew

    xi2 = xinew / np.sum(xinew)

    ind_tmp = np.where(xi_tmp == min(xi_tmp[xi_tmp > 0]))[0]
    if z == maxiter - 1:
        tmp = np.abs(xi2[ind_tmp] - xi_tmp[ind_tmp]) / xi_tmp[ind_tmp]
        logger.warning(
            "    More than {} iterations needed. Error in Smallest Fraction: {} %%".format(
                maxiter, tmp * 100
            )
        )
        if tmp > 0.1:  # If difference is greater than 10%
            xinew = find_new_xi(
                P, T, phiv, yi, Eos, density_opts=density_opts, **kwargs
            )
        xinew = spo.least_squares(
            objective_find_xi,
            xinew[0],
            bounds=(0.0, 1.0),
            args=(P, T, phiv, yi, Eos, density_opts),
        )
        xi = xinew.x[0]
        xi_tmp = np.array([xi, 1 - xi])
        obj = objective_find_xi(xi_tmp, P, T, phiv, yi, Eos, density_opts=density_opts)
        logger.warning(
            "    Find xi with root algorithm, xi {}, obj {}".format(xi_tmp, obj)
        )

    return xi_tmp, phil, flagl


def find_new_yi(
    P, T, phil, xi, Eos, bounds=(0.01, 0.99), npoints=30, density_opts={}, **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    bounds : tuple, Optional, default=(0.01, 0.99)
        These bounds dictate the lower and upper boundary for the first component in a binary system.
    npoints : float, Optional, default=30
        Number of points to test between the bounds.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'find_new_yi' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    yi_ext = np.linspace(bounds[0], bounds[1], npoints)  # Guess for yi
    obj_ext = np.zeros(len(yi_ext))
    flag_ext = np.zeros(len(yi_ext))

    for i, yi in enumerate(yi_ext):
        yi = np.array([yi, 1 - yi])
        obj, flagv = objective_find_yi(
            yi, P, T, phil, xi, Eos, density_opts=density_opts, return_flag=True
        )
        flag_ext[i] = flagv
        obj_ext[i] = obj

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    logger.debug("    Number of valid mole fractions: {}".format(tmp))
    if tmp == 0:
        yi_tmp = np.nan
        obj_tmp = np.nan
    else:
        # Remove any NaN
        obj_tmp = obj_ext[~np.isnan(obj_ext)]
        yi_tmp = yi_ext[~np.isnan(obj_ext)]
        flag_tmp = flag_ext[~np.isnan(obj_ext)]

        # Fit spline
        spline = interpolate.Akima1DInterpolator(yi_tmp, obj_tmp)
        yi_min = spline.derivative().roots()

        if len(yi_min) > 1:

            # Remove local maxima
            yi_concav = spline.derivative(nu=2)(yi_min)
            yi_min = [yi_min[i] for i in range(len(yi_min)) if yi_concav[i] > 0.0]

            # Add end points if relevant
            if len(yi_tmp) > 1:
                if obj_tmp[0] < obj_tmp[1]:
                    yi_min.insert(0, yi_tmp[0])
                if obj_tmp[-1] < obj_tmp[-2]:
                    yi_min.append(yi_tmp[-1])
            yi_min = np.array(yi_min)

            ## Remove trivial solution
            obj_trivial = np.abs(yi_min - xi[0]) / xi[0]
            ind = np.where(obj_trivial == min(obj_trivial))[0][0]
            logger.debug(
                "    Found multiple minima: {}, discard {} as trivial solution".format(
                    yi_min, yi_min[ind]
                )
            )

            # Remove liquid roots
            yi_min = np.array([yi_min[ii] for ii in range(len(yi_min)) if ii != ind])
            if len(yi_min) > 1:
                lyi = len(yi_min)
                obj_tmp2 = np.zeros(lyi)
                flagv_tmp2 = np.zeros(lyi)
                for ii in range(lyi):
                    obj_tmp2[ii], flagv_tmp2[ii] = objective_find_yi(
                        yi_min[ii],
                        P,
                        T,
                        phil,
                        xi,
                        Eos,
                        density_opts=density_opts,
                        return_flag=True,
                    )
                yi_tmp2 = [
                    yi_min[ii] for ii in range(len(yi_min)) if flagv_tmp2[ii] != 1
                ]
                if len(yi_tmp2):
                    obj_tmp2 = [
                        obj_tmp2[ii]
                        for ii in range(len(obj_tmp2))
                        if flagv_tmp2[ii] != 1
                    ]
                    yi_min = [yi_tmp2[np.where(obj_tmp2 == min(obj_tmp2))[0][0]]]
                else:
                    yi_min = [yi_min[np.where(obj_tmp2 == min(obj_tmp2))[0][0]]]

        if not len(yi_min):
            # Choose values with lowest objective function
            ind = np.where(np.abs(obj_tmp) == min(np.abs(obj_tmp)))[0][0]
            obj_final = obj_tmp[ind]
            yi_final = yi_tmp[ind]
        else:
            yi_final = yi_min[0]
            obj_final = spline(yi_min[0])

    logger.debug("    Found new guess in yi: {}, Obj: {}".format(yi_final, obj_final))
    if not gtb.isiterable(yi_final):
        yi_final = np.array([yi_final, 1 - yi_final])

    return yi_final


def bracket_bounding_yi(
    P,
    T,
    phil,
    xi,
    Eos,
    bounds=(0.01, 0.99),
    maxiter=50,
    tol=1e-7,
    density_opts={},
    **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    bounds : tuple, Optional, default=(0.01, 0.99)
        These bounds dictate the lower and upper boundary for the first component in a binary system.
    maxiter : int, Optional, default=50
        Maximum number of iterations
    tol : float, Optional, default=1e-7
        Tolerance to quit search for yi
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    flag : int
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'calc_saturation_properties' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    if np.size(bounds) != 2:
        raise ValueError("Given bounds on y1 must be of length two.")

    bounds = np.array(bounds)
    obj_bounds = np.zeros(2)
    flag_bounds = np.zeros(2)
    obj_bounds[0], flag_bounds[0] = objective_find_yi(
        bounds[0], P, T, phil, xi, Eos, density_opts=density_opts, return_flag=True
    )
    obj_bounds[1], flag_bounds[1] = objective_find_yi(
        bounds[1], P, T, phil, xi, Eos, density_opts=density_opts, return_flag=True
    )

    if flag_bounds[0] == flag_bounds[1]:
        logger.error(
            "    Both mole fractions have flag, {}, continue seeking convergence".format(
                flag_bounds[0]
            )
        )
        y1 = bounds[1]
        flagv = flag_bounds[1]
        i = maxiter - 1

    else:
        flag_high_vapor = False
        for i in np.arange(maxiter):

            y1 = np.mean(bounds)
            obj, flagv = objective_find_yi(
                y1, P, T, phil, xi, Eos, density_opts=density_opts, return_flag=True
            )

            if not flag_high_vapor:
                ind = np.where(flag_bounds == flagv)[0][0]
                if flagv == 0 and obj > 1 / tol:
                    flag_high_vapor = True
                    bounds[0], obj_bounds[0], flag_bounds[0] = (
                        bounds[ind],
                        obj_bounds[ind],
                        flag_bounds[ind],
                    )
                    ind = 1
            else:
                if obj < obj_bounds[0]:
                    ind = 0
                else:
                    ind = 1

            bounds[ind], obj_bounds[ind], flag_bounds[ind] = y1, obj, flagv
            logger.debug(
                "    Bouncing mole fraction new bounds: {}, obj: {}, flag: {}".format(
                    bounds, obj_bounds, flag_bounds
                )
            )

            # Check convergence
            if np.abs(bounds[1] - bounds[0]) < tol:
                break

    ind_array = np.where(flag_bounds == 0)[0]
    if np.size(ind_array) == 1:
        ind = ind_array[0]
    else:
        ind = np.where(obj_bounds == np.min(obj_bounds))[0][0]

    y1, flagv = bounds[ind], flag_bounds[ind]
    if i == maxiter - 1:
        logger.debug(
            "    Bouncing mole fraction, max iterations ended with, y1={}, flagv={}".format(
                y1, flagv
            )
        )
    else:
        logger.debug(
            "    Bouncing mole fractions converged to y1={}, flagv={}".format(y1, flagv)
        )

    return np.array([y1, 1 - y1]), flagv


def objective_find_yi(yi, P, T, phil, xi, Eos, density_opts={}, return_flag=False):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    return_flag : bool, Optional, default=False
        If True, the objective value and flagv is returned, otherwise, just the objective value is returned

    Returns
    -------
    obj : numpy.ndarray
        Objective function for solving for vapor mole fractions
    flag : int, Optional
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed. Only outputted when `return_flag` is True
    """

    if type(yi) == float or np.size(yi) == 1:
        if gtb.isiterable(yi):
            yi = np.array([yi[0], 1 - yi[0]])
        else:
            yi = np.array([yi, 1 - yi])
    elif isinstance(yi, list):
        yi = np.array(yi)
    yi /= np.sum(yi)

    phiv, _, flagv = calc_vapor_fugacity_coefficient(
        P, T, yi, Eos, density_opts=density_opts
    )

    yinew = calc_new_mole_fractions(xi, phil, phiv, phase="vapor")
    yi2 = yinew / np.sum(yinew)

    if np.any(np.isnan(yi2)):
        obj = np.nan
    else:
        phiv2, _, flagv2 = calc_vapor_fugacity_coefficient(
            P, T, yi2, Eos, density_opts=density_opts
        )
        obj = np.sum(np.abs(yinew - xi * phil / phiv2))

    logger.debug(
        "    Guess yi: {}, calc yi: {}, diff={}, flagv {}".format(yi, yi2, obj, flagv)
    )

    if return_flag:
        return obj, flagv
    else:
        return obj


def find_new_xi(
    P, T, phiv, yi, Eos, density_opts={}, bounds=(0.001, 0.999), npoints=30, **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    bounds : tuple, Optional, default=(0.001, 0.999)
        These bounds dictate the lower and upper boundary for the first component in a binary system.
    npoints : float, Optional, default=30
        Number of points to test between the bounds.
        
    Returns
    -------
    xi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    """

    if len(kwargs) > 0:
        logger.debug(
            "    'find_new_xi' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    xi_ext = np.linspace(bounds[0], bounds[1], npoints)  # Guess for yi
    obj_ext = np.zeros(len(xi_ext))
    flag_ext = np.zeros(len(xi_ext))

    for i, xi in enumerate(xi_ext):
        xi = np.array([xi, 1 - xi])
        obj, flagl = objective_find_xi(
            xi, P, T, phiv, yi, Eos, density_opts=density_opts, return_flag=True
        )
        flag_ext[i] = flagl
        obj_ext[i] = obj

    tmp = np.count_nonzero(~np.isnan(obj_ext))
    logger.debug("    Number of valid mole fractions: {}".format(tmp))
    if tmp == 0:
        xi_final = np.nan
        obj_final = np.nan
    else:
        # Remove any NaN
        obj_tmp = obj_ext[~np.isnan(obj_ext)]
        xi_tmp = xi_ext[~np.isnan(obj_ext)]
        flag_tmp = flag_ext[~np.isnan(obj_ext)]

        spline = interpolate.Akima1DInterpolator(xi_tmp, obj_tmp)
        xi_min = spline.derivative().roots()

        if len(xi_min) > 1:

            # Remove local maxima
            xi_concav = spline.derivative(nu=2)(xi_min)
            xi_min = [xi_min[i] for i in range(len(xi_min)) if xi_concav[i] > 0.0]
            # Add end points if relevant
            if len(xi_tmp) > 1:
                if obj_tmp[0] < obj_tmp[1]:
                    xi_min.insert(0, xi_tmp[0])
                if obj_tmp[-1] < obj_tmp[-2]:
                    xi_min.append(xi_tmp[-1])
            xi_min = np.array(xi_min)
            # Remove trivial solution
            obj_trivial = np.abs(xi_min - yi[0]) / yi[0]
            ind = np.where(obj_trivial == min(obj_trivial))[0][0]
            logger.debug(
                "    Found multiple minima: {}, discard {} as trivial solution".format(
                    xi_min, xi_min[ind]
                )
            )
            xi_min = np.array([xi_min[ii] for ii in range(len(xi_min)) if ii != ind])

        if not len(xi_min):
            # Choose values with lowest objective function
            ind = np.where(np.abs(obj_tmp) == min(np.abs(obj_tmp)))[0][0]
            obj_final = obj_tmp[ind]
            xi_final = xi_tmp[ind]
        else:
            xi_final = xi_min[0]
            obj_final = spline(xi_min[0])

    logger.debug("    Found new guess in xi: {}, Obj: {}".format(xi_final, obj_final))
    if not gtb.isiterable(xi_final):
        xi_final = np.array([xi_final, 1 - xi_final])

    return xi_final


def objective_find_xi(xi, P, T, phiv, yi, Eos, density_opts={}, return_flag=False):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    return_flag : bool, Optional, default=False
        If True, the objective value and flagl is returned, otherwise, just the objective value is returned
        
    Returns
    -------
    obj : numpy.ndarray
        Objective function for solving for liquid mole fractions
    flag : int, Optional
        Flag identifying the fluid type. A value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means ideal gas is assumed. Only outputted when `return_flag` is True
    """

    if isinstance(xi, float) or len(xi) == 1:
        if gtb.isiterable(xi):
            xi = np.array([xi[0], 1 - xi[0]])
        else:
            xi = np.array([xi, 1 - xi])
    elif isinstance(xi, list):
        xi = np.array(xi)
    xi /= np.sum(xi)

    phil, _, flagl = calc_liquid_fugacity_coefficient(
        P, T, xi, Eos, density_opts=density_opts
    )

    xinew = calc_new_mole_fractions(yi, phil, phiv, phase="liquid")
    xi2 = xinew / np.sum(xinew)

    if np.any(np.isnan(xi2)):
        obj = np.nan
    else:
        phil2, _, flagl2 = calc_liquid_fugacity_coefficient(
            P, T, xi2, Eos, density_opts=density_opts
        )
        obj = np.sum(np.abs(xinew - xi * phiv / phil2))

    logger.debug(
        "    Guess xi: {}, calc xi: {}, diff={}, flagl {}".format(xi, xi2, obj, flagl)
    )

    if return_flag:
        return obj, flagl
    else:
        return obj


def objective_bubble_pressure(
    P, xi, T, Eos, density_opts={}, mole_fraction_options={}, **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    obj_value : float
        :math:`\sum\frac{x_{i}\{phi_l}{\phi_v}-1`
    """

    if len(kwargs) > 0:
        logger.debug(
            "'objective_bubble_pressure' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _yi_global

    if P < 0:
        return 10.0

    logger.info("P Guess: {} Pa".format(P))

    # find liquid density
    phil, rhol, flagl = calc_liquid_fugacity_coefficient(
        P, T, xi, Eos, density_opts=density_opts
    )

    yinew, phiv, flagv = calc_vapor_composition(
        _yi_global,
        xi,
        phil,
        P,
        T,
        Eos,
        density_opts=density_opts,
        **mole_fraction_options
    )
    _yi_global = yinew / np.sum(yinew)

    # given final yi recompute
    phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
        P, T, _yi_global, Eos, density_opts=density_opts
    )

    Pv_test = Eos.pressure(rhov, T, _yi_global)
    obj_value = equilibrium_objective(xi, phil, phiv, phase="vapor")
    logger.info("Obj Func: {}, Pset: {}, Pcalc: {}".format(obj_value, P, Pv_test[0]))

    return obj_value


def objective_dew_pressure(
    P, yi, T, Eos, density_opts={}, mole_fraction_options={}, **kwargs
):
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
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm

    Returns
    -------
    obj_value : list
        :math:`\sum\frac{y_{i}\{phi_v}{\phi_l}-1`
    """

    if len(kwargs) > 0:
        logger.debug(
            "'objective_dew_pressure' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _xi_global

    if P < 0:
        return 10.0

    logger.info("P Guess: {} Pa".format(P))

    # find liquid density
    phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
        P, T, yi, Eos, density_opts=density_opts
    )

    xinew, phil, flagl = calc_liquid_composition(
        _xi_global,
        yi,
        phiv,
        P,
        T,
        Eos,
        density_opts=density_opts,
        **mole_fraction_options
    )
    _xi_global = xinew / np.sum(xinew)

    # given final yi recompute
    phil, rhol, flagl = calc_liquid_fugacity_coefficient(
        P, T, _xi_global, Eos, density_opts=density_opts
    )

    Pv_test = Eos.pressure(rhol, T, _xi_global)
    obj_value = equilibrium_objective(yi, phil, phiv, phase="liquid")
    logger.info("Obj Func: {}, Pset: {}, Pcalc: {}".format(obj_value, P, Pv_test[0]))

    return obj_value


def calc_dew_pressure(
    yi,
    T,
    Eos,
    density_opts={},
    mole_fraction_options={},
    Pguess=None,
    method="bisect",
    pressure_options={},
    Psat_set=1e7,
    **kwargs
):
    r"""
    Calculate dew point mole fraction and pressure given system vapor mole fraction and temperature.
    
    Parameters
    ----------
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(yi) should equal 1.0
    T : float
        [K] Temperature of the system
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default=None
        [Pa] Guess the system pressure at the dew point. A negative value will force an estimation based on the saturation pressure of each component.
    Psat_set : float, Optional, default=1e+7
        [Pa] Set the saturation pressure if the pure component is above the critical point in these conditions
    method : str, Optional, default="bisect"
        Choose the method used to solve the dew point calculation
    pressure_options : dict, Optional, default={}
        Options used in the given method, "method", to solve the outer loop in the solving algorithm
    kwargs
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties`

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

    if len(kwargs) > 0:
        logger.debug(
            "'calc_dew_pressure' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _xi_global

    # Estimate pure component vapor pressures
    Psat = np.zeros_like(yi)
    for i in range(np.size(yi)):
        yi_tmp = np.zeros_like(yi)
        yi_tmp[i] = 1.0
        Psat[i], _, _ = calc_saturation_properties(
            T, yi_tmp, Eos, density_opts=density_opts, **kwargs
        )
        if np.isnan(Psat[i]):
            Psat[i] = Psat_set
            logger.warning(
                "Component, {}, is above its critical point. Psat is assumed to be {}.".format(
                    i + 1, Psat[i]
                )
            )

    # Estimate initial pressure
    if Pguess is None:
        P = 1.0 / np.sum(yi / Psat)
    else:
        P = Pguess

    # Estimate initial xi
    if "_xi_global" not in globals() or any(np.isnan(_xi_global)):
        _xi_global = P * (yi / Psat)
        _xi_global /= np.sum(_xi_global)
        _xi_global = copy.deepcopy(_xi_global)
        logger.info("Guess xi in calc_dew_pressure with Psat: {}".format(_xi_global))
    xi = _xi_global

    Prange, Pestimate = calc_Prange_yi(
        T,
        xi,
        yi,
        Eos,
        density_opts=density_opts,
        mole_fraction_options=mole_fraction_options,
        **kwargs
    )
    if np.any(np.isnan(Prange)):
        raise ValueError(
            "Neither a suitable pressure range, or guess in pressure could be found nor was given."
        )
    else:
        if Pguess is not None:
            if Pguess > Prange[1] or Pguess < Prange[0]:
                logger.warning(
                    "Given guess in pressure, {}, is outside of the identified pressure range, {}. Using estimated pressure, {}.".format(
                        Pguess, Prange, Pestimate
                    )
                )
                P = Pestimate
            else:
                logger.warning(
                    "Using given guess in pressure, {}, that is inside identified pressure range.".format(
                        Pguess
                    )
                )
                P = Pguess
        else:
            P = Pestimate
        P = gtb.solve_root(
            objective_dew_pressure,
            args=(yi, T, Eos, density_opts, mole_fraction_options),
            x0=P,
            method=method,
            bounds=Prange,
            options=pressure_options,
        )

    # find vapor density and fugacity
    phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
        P, T, yi, Eos, density_opts=density_opts
    )
    phil, rhol, flagl = calc_liquid_fugacity_coefficient(
        P, T, xi, Eos, density_opts=density_opts
    )
    if "tol" in mole_fraction_options:
        if mole_fraction_options["tol"] > 1e-10:
            mole_fraction_options["tol"] = 1e-10

    obj = objective_dew_pressure(
        P,
        yi,
        T,
        Eos,
        density_opts=density_opts,
        mole_fraction_options=mole_fraction_options,
    )

    logger.info(
        "Final Output: Obj {}, P {} Pa, flagl {}, xi {}".format(
            obj, P, flagl, _xi_global
        )
    )

    return P, xi, flagl, flagv, obj


def calc_bubble_pressure(
    xi,
    T,
    Eos,
    density_opts={},
    mole_fraction_options={},
    Pguess=None,
    Psat_set=1e7,
    method="bisect",
    pressure_options={},
    **kwargs
):
    r"""
    Calculate bubble point mole fraction and pressure given system liquid mole fraction and temperature.
    
    Parameters
    ----------
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        [K] Temperature of the system
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    mole_fraction_options : dict, Optional, default={}
        Options used to solve the inner loop in the solving algorithm
    Pguess : float, Optional, default=None
        [Pa] Guess the system pressure at the dew point. A value of None will force an estimation based on the saturation pressure of each component.
    Psat_set : float, Optional, default=1e+7
        [Pa] Set the saturation pressure if the pure component is above the critical point in these conditions
    method : str, Optional, default="bisect"
        Choose the method used to solve the dew point calculation
    pressure_options : dict, Optional, default={}
        Options used in the given method, "method", to solve the outer loop in the solving algorithm
    kwargs
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties`

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

    if len(kwargs) > 0:
        logger.debug(
            "'calc_bubble_pressure' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    global _yi_global

    Psat = np.zeros_like(xi)
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], _, _ = calc_saturation_properties(
            T, xi_tmp, Eos, density_opts=density_opts, **kwargs
        )
        if np.isnan(Psat[i]):
            Psat[i] = Psat_set
            logger.warning(
                "Component, {}, is above its critical point. Psat is assumed to be {}.".format(
                    i + 1, Psat[i]
                )
            )

    # Estimate initial pressure
    if Pguess == None:
        P = 1.0 / np.sum(xi / Psat)
    else:
        P = Pguess

    if "_yi_global" not in globals() or any(np.isnan(_yi_global)):
        _yi_global = xi * Psat / P
        _yi_global /= np.nansum(_yi_global)
        _yi_global = copy.deepcopy(_yi_global)
        logger.info("Guess yi in calc_bubble_pressure with Psat: {}".format(_yi_global))
    yi = _yi_global

    Prange, Pestimate = calc_Prange_xi(
        T,
        xi,
        yi,
        Eos,
        density_opts=density_opts,
        mole_fraction_options=mole_fraction_options,
        **kwargs
    )
    if np.any(np.isnan(Prange)):
        raise ValueError(
            "Neither a suitable pressure range, or guess in pressure could be found nor was given."
        )
    else:
        if Pguess != None:
            if Pguess > Prange[1] or Pguess < Prange[0]:
                logger.warning(
                    "Given guess in pressure, {}, is outside of the identified pressure range, {}. Using estimated pressure, {}.".format(
                        Pguess, Prange, Pestimate
                    )
                )
                P = Pestimate
            else:
                logger.warning(
                    "Using given guess in pressure, {}, that is inside identified pressure range.".format(
                        Pguess
                    )
                )
                P = Pguess
        else:
            P = Pestimate
        P = gtb.solve_root(
            objective_bubble_pressure,
            args=(xi, T, Eos, density_opts, mole_fraction_options),
            x0=P,
            method=method,
            bounds=Prange,
            options=pressure_options,
        )

    # find liquid density and fugacity
    phil, rhol, flagl = calc_liquid_fugacity_coefficient(
        P, T, xi, Eos, density_opts=density_opts
    )
    phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
        P, T, yi, Eos, density_opts=density_opts
    )
    if "tol" in mole_fraction_options:
        if mole_fraction_options["tol"] > 1e-10:
            mole_fraction_options["tol"] = 1e-10

    obj = objective_bubble_pressure(
        P,
        xi,
        T,
        Eos,
        density_opts=density_opts,
        mole_fraction_options=mole_fraction_options,
    )

    logger.info(
        "Final Output: Obj {}, P {} Pa, flagv {}, yi {}".format(
            obj, P, flagv, _yi_global
        )
    )

    return P, _yi_global, flagv, flagl, obj


def hildebrand_solubility(
    rhol, xi, T, Eos, dT=0.1, tol=1e-4, density_opts={}, **kwargs
):
    r"""
    Calculate the solubility parameter based on temperature and composition. 

    This function is based on the method used in Zeng, Z., Y. Xi, and Y. Li "Calculation of Solubility Parameter Using Perturbed-Chain SAFT and Cubic-Plus-Association Equations of State" Ind. Eng. Chem. Res. 2008, 47, 9663–9669.
    
    Parameters
    ----------
    rhol : float
        Liquid molar density [mol/m^3]
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    T : float
        Temperature of the system [K]
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    dT : float, Optional, default=0.1
        Change in temperature used in calculating the derivative with central difference method 
    tol : float, Optional, default=1e-4
        This cutoff value evaluates the extent to which the integrand of the calculation has decayed. If the last value if the array is greater than tol, then the remaining area is estimated as a triangle, where the intercept is estimated from an interpolation of the previous four points.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`

    Returns
    -------
    delta : float
        Solubility parameter [Pa^(1/2)], ratio of cohesive energy and molar volume
    """
    if len(kwargs) > 0:
        logger.debug(
            "'hildebrand_solubility' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    R = constants.Nav * constants.kb
    RT = T * R

    if gtb.isiterable(rhol):
        logger.info("rhol should be a float, not {}".format(rhol))

    # Find dZdT
    vlist, Plist1 = pressure_vs_volume_arrays(
        T - dT, xi, Eos, **density_opts, max_density=rhol
    )
    vlist2, Plist2 = pressure_vs_volume_arrays(
        T + dT, xi, Eos, **density_opts, max_density=rhol
    )
    vlist, Plist = pressure_vs_volume_arrays(
        T, xi, Eos, **density_opts, max_density=rhol
    )
    if any(vlist != vlist2):
        logger.error("Dependant variable vectors must be identical.")

    int_tmp = (Plist2 - Plist1) / (2 * dT) / R - Plist / (RT)
    integrand_list = gaussian_filter1d(int_tmp, sigma=0.1)

    # Calculate U_res
    integrand_spline = interpolate.InterpolatedUnivariateSpline(
        vlist, integrand_list, ext=1
    )
    U_res = -RT * integrand_spline.integral(1 / rhol, vlist[-1])

    # Check if function converged before taking integral, if not, correct area
    if integrand_list[-1] > tol:
        slope, yroot = np.polyfit(vlist[-4:], integrand_list[-4:], 1)
        xroot = -yroot / slope
        U_res += -RT * integrand_list[-1] * (xroot - vlist[-1]) / 2

    if (U_res) > 0.0:
        raise ValueError("The solubility parameter can not be imaginary")
    else:
        delta = np.sqrt(-(U_res) * rhol)
        logger.info("When T={}, xi={}, delta={}".format(T, xi, delta))

    return delta


def calc_flash(
    P,
    T,
    Eos,
    density_opts={},
    maxiter=200,
    tol=1e-9,
    max_mole_fraction0=1,
    min_mole_fraction0=0,
    Psat_set=1e7,
    **kwargs
):
    r"""
    Binary flash calculation of vapor and liquid mole fractions.
    
    Parameters
    ----------
    P : numpy.ndarray
        Pressure of the system [Pa]
    T : float
        Temperature of the system [K]
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    density_opts : dict, Optional, default={}
        Dictionary of options used in calculating pressure vs. mole in :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`
    maxiter : int, Optional, default=200
        Maximum number of iterations in updating Ki values
    tol : float, Optional, tol: 1e-9
        Tolerance to break loop. The error is defined as the absolute value of the summed difference in Ki values between iterations.
    min_mole_fraction0 : float, Optional, default=0
        Set the vapor and liquid mole fraction of component one to be greater than this number. Useful for diagrams with multiple solutions, such as those with an azeotrope.
    max_mole_fraction0 : float, Optional, default=1
        Set the vapor and liquid mole fraction of component one to be less than this number. Useful for diagrams with multiple solutions, such as those with an azeotrope.
    Psat_set : float, Optional, default=1e+7
        [Pa] Set the saturation pressure if the pure component is above the critical point in these conditions
    kwargs
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties`

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

    if len(kwargs) > 0:
        logger.debug(
            "'kwargs' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    # Initialize Variables
    if Eos.number_of_components != 2:
        raise ValueError(
            "Only binary systems are currently supported for flash calculations, {} were given.".format(
                Eos.number_of_components
            )
        )

    Psat, Ki0, xi, yi, phil, phiv = [
        np.zeros(Eos.number_of_components) for i in np.arange(6)
    ]

    # Calculate Psat and Ki
    for i in range(np.size(xi)):
        xi_tmp = np.zeros_like(xi)
        xi_tmp[i] = 1.0
        Psat[i], _, _ = calc_saturation_properties(
            T, xi_tmp, Eos, density_opts=density_opts, **kwargs
        )
        if np.isnan(Psat[i]):
            Psat[i] = Psat_set
            logger.warning(
                "Component, {}, is above its critical point. Psat is assumed to be {}.".format(
                    i + 1, Psat[i]
                )
            )
        Ki0[i] = Psat[i] / P

    Ki, _ = constrain_Ki(
        Ki0,
        min_mole_fraction0=min_mole_fraction0,
        max_mole_fraction0=max_mole_fraction0,
    )
    err = 1
    flag_critical = 0
    count_reset = 0
    for i in np.arange(maxiter):

        # Mole Fraction
        xi[0] = (1 - Ki[1]) / (Ki[0] - Ki[1])
        xi[1] = 1 - xi[0]
        if any(xi < 0.0):
            ind = np.where(xi < 0.0)[0][0]
            xi[ind] = np.sqrt(np.finfo(float).eps)
            if ind == 0:
                xi[1] = 1 - xi[0]
            elif ind == 1:
                xi[0] = 1 - xi[1]

        yi = Ki * xi
        if np.sum(yi) != 1.0:
            if np.abs(np.sum(yi) != 1.0) < np.sqrt(np.finfo(float).eps):
                raise ValueError(
                    "Vapor mole fractions do not add up to 1. Ki {}, xi {} produces {} = {}".format(
                        Ki, xi, yi, np.sum(yi)
                    )
                )
            else:
                yi /= np.sum(yi)

        # Fugacity Coefficients and New Ki values
        phil, rhol, flagl = calc_liquid_fugacity_coefficient(
            P, T, xi, Eos, density_opts=density_opts
        )
        phiv, rhov, flagv = calc_vapor_fugacity_coefficient(
            P, T, yi, Eos, density_opts=density_opts
        )
        logger.info("        xi: {}, phil: {}".format(xi, phil))
        logger.info("        yi: {}, phiv: {}".format(yi, phiv))
        Kinew = phil / phiv

        err = np.sum(np.abs(Kinew - Ki))
        logger.info(
            "  Guess {} Ki: {}, New Ki: {}, Error: {}".format(i, Ki, Kinew, err)
        )

        # Check Objective function
        Kiprev = Ki
        Ki_tmp, flag_reset = constrain_Ki(
            Kinew,
            min_mole_fraction0=min_mole_fraction0,
            max_mole_fraction0=max_mole_fraction0,
        )
        if flag_reset:
            count_reset += 1
        if not (Kinew == Ki_tmp).all():
            logger.info(
                "    Reset Ki values, {}, according to mole fraction constraint, {} to {}, to produce {}".format(
                    Kinew, min_mole_fraction0, max_mole_fraction0, Ki_tmp
                )
            )
            Ki = Ki_tmp
            if count_reset == 10:
                tmp = Ki[0]
                Ki[0] = Ki[1]
                Ki[1] = tmp
            elif count_reset == 20:
                ind = np.where(Kiprev == min(Kiprev[Kiprev > 0]))[0][0]
                err = np.abs(Ki[ind] - Kiprev[ind]) / Kiprev[ind]
                logger.warning(
                    "    Reset Ki values more than {} times. Remaining error, {}. These constraints may not be feasible".format(
                        20, err
                    )
                )
                break
        elif np.all(np.abs(Ki - 1.0) < 1e-6) and flag_critical < 2:
            eps = np.sqrt(np.finfo(float).eps)
            ind = 1 - flag_critical
            if flag_critical == 0:
                Ki[ind] = eps
                Ki[flag_critical] = 1 / eps
            else:
                Ki[ind] = 1 / eps
                Ki[flag_critical] = eps
            flag_critical += 1
            logger.info(
                "    Liquid and vapor mole fractions are equal, let search from Ki = {}".format(
                    Ki
                )
            )
        elif err < tol:
            ind = np.where(Ki == min(Ki[Ki > 0]))[0][0]
            err = np.abs(Kinew[ind] - Ki[ind]) / Ki[ind]
            logger.info("    Percent Error on smallest Ki value: {}".format(err))
            if err < tol:
                logger.info("    Found Ki")
                break
            Ki = Kinew
        else:
            Ki = Kinew

    if i == maxiter - 1:
        ind = np.where(Kiprev == min(Kiprev[Kiprev > 0]))[0][0]
        err = np.abs(Ki[ind] - Kiprev[ind]) / Kiprev[ind]
        logger.warning(
            "    More than {} iterations needed. Remaining error, {}.".format(
                maxiter, err
            )
        )

    # If needed, switch liquid and vapor mole fractions
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

    logger.info(
        "Final Output: Obj {}, xi {} flagl {}, yi {} flagv {}".format(
            err, xi, flagl, yi, flagv
        )
    )

    return xi, flagl, yi, flagv, err


def constrain_Ki(Ki0, min_mole_fraction0=0, max_mole_fraction0=1, **kwargs):
    r"""
    For a binary mixture, determine whether the K values will produce properly constrained mole fractions.

    If not, randomly choose a value of Ki[1] within the allowed range.
    
    Parameters
    ----------
    Ki : numpy.ndarray
        K values for a binary mixture
    min_mole_fraction0 : float, Optional, default=0
        Set the vapor and liquid mole fraction of component one to be greater than this number. Useful for diagrams with multiple solutions, such as those with an azeotrope.
    max_mole_fraction0 : float, Optional, default=1
        Set the vapor and liquid mole fraction of component one to be less than this number. Useful for diagrams with multiple solutions, such as those with an azeotrope.

    Returns
    -------
    Ki_new : numpy.ndarray
        New suggestion for K values for a binary mixture
    flag_reset : bool
        True or False value indicating that the K values were reset.
    """

    if len(kwargs) > 0:
        logger.debug(
            "'constrain_Ki' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    Ki = Ki0.copy()
    flag_reset = False
    eps = np.sqrt(np.finfo(float).eps)

    # Set-up
    if Ki[0] > Ki[1]:
        min0 = eps
        max0 = 1
    elif Ki[0] < Ki[1]:
        min0 = 1
        max0 = 1e8
    min_list = [min0]
    max_list = [max0]

    # flag, x0 x1 y0 y1
    flag = [False for x in range(4)]

    # Check K0
    if Ki[0] > Ki[1] and Ki[0] < 1:
        Ki[0] = 1 / eps
    elif Ki[0] < Ki[1] and (Ki[0] > 1 or Ki[0] < 0):
        Ki[0] = eps

    if min_mole_fraction0 > 0:
        bound_min_x0 = (1 - min_mole_fraction0 * Ki[0]) / (1 - min_mole_fraction0)
        bound_min_y0 = (1 - min_mole_fraction0) * Ki[0] / (Ki[0] - min_mole_fraction0)

        if Ki[0] > Ki[1]:
            max_list.extend([bound_min_y0])
            if bound_min_x0 > 0:
                max_list.extend([bound_min_x0])
            else:
                flag[0] = True

        elif Ki[0] < Ki[1]:
            min_list.extend([bound_min_x0])
            if bound_min_y0 > 0:
                min_list.extend([bound_min_y0])
            else:
                flag[1] = True

    elif min_mole_fraction0 < 0 or min_mole_fraction0 > 1:
        raise ValueError(
            "Mole fractions can only be constrained to a value between 0 and 1"
        )

    if max_mole_fraction0 < 1:
        bound_max_x0 = (1 - max_mole_fraction0 * Ki[0]) / (1 - max_mole_fraction0)
        bound_max_y0 = (1 - max_mole_fraction0) * Ki[0] / (Ki[0] - max_mole_fraction0)

        if Ki[0] > Ki[1]:
            min_list.extend([bound_max_y0])
            if bound_max_x0 > 0:
                min_list.extend([bound_max_x0])
            else:
                flag[2] = True

        elif Ki[0] < Ki[1]:
            max_list.extend([bound_max_x0])
            if bound_max_y0 > 0:
                max_list.extend([bound_max_y0])
            else:
                flag[3] = True

    elif max_mole_fraction0 < 0 or max_mole_fraction0 > 1:
        raise ValueError(
            "Mole fractions can only be constrained to a value between 0 and 1"
        )

    max0 = min(max_list)
    min0 = max(min_list)

    if np.any(Ki[1] > max_list) or np.any(Ki[1] < min_list):
        logger.debug("    Constrain K1 to between {} and {}".format(min0, max0))
        Ki[1] = (max0 - min0) * np.random.rand(1)[0] + min0
        flag_reset = True

    x0 = (1 - Ki[1]) / (Ki[0] - Ki[1])
    y0 = Ki[0] * x0

    # if flag[0]:
    #    tmp = Ki[1]*(1-
    # elif flag[1]:
    #
    # elif flag[2]:
    #
    # elif flag[3]:
    #

    if x0 < min_mole_fraction0 or y0 < min_mole_fraction0:
        raise ValueError(
            "x0: {}, y0 {}, breach lower limit {}".format(x0, y0, max_mole_fraction0)
        )

    if x0 > max_mole_fraction0 or y0 > max_mole_fraction0:
        raise ValueError(
            "x0: {}, y0 {}, breach upper limit {}".format(x0, y0, max_mole_fraction0)
        )

    return Ki, flag_reset


def fugacity_test_1(P, T, xi, rho, Eos, step_size=1e-5, **kwargs):
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
        [mol/:math:`m^3`] Density array. Length depends on values in density_opts
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    step_size : float, Optional, default=1e-5
        Step size in central difference method

    Returns
    -------
    Residual : float
        Residual from thermodynamic identity 
        
    """

    if len(kwargs) > 0:
        logger.debug(
            "'fugacity_test_1' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    if not gtb.isiterable(rho):
        rho = np.array([rho])

    Z = P / (rho * T * constants.R)
    dP = P * step_size
    log_phi_1 = np.sum(xi * np.log(Eos.fugacity_coefficient(P + dP, rho, xi, T)))
    log_phi_2 = np.sum(xi * np.log(Eos.fugacity_coefficient(P - dP, rho, xi, T)))
    residual = (log_phi_1 - log_phi_2) / (2 * dP) - (Z - 1) / P

    return residual


def fugacity_test_2(P, T, xi, rho, Eos, fractional_change=1e-1, **kwargs):
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
        [mol/:math:`m^3`] Density array. Length depends on values in density_opts
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    fractional_change : float, Optional, default=1e-1
        !!!!!!!!NoteHere!!!!!

    Returns
    -------
    Residual : float
        
    """

    if len(kwargs) > 0:
        logger.debug(
            "'fugacity_test_2' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    ncomp = len(xi)
    if not gtb.isiterable(rho):
        rho = np.array([rho])

    #    drho = rho * step_size
    #    log_phi_1 = np.log(Eos.fugacity_coefficient(P, rho+drho, xi, T))
    #    log_phi_2 = np.log(Eos.fugacity_coefficient(P, rho-drho, xi, T))
    #    residual = np.sum(xi*(log_phi_1-log_phi_2)/(2*drho))

    ind = np.where(xi > np.finfo("float").eps)[0]
    if len(ind) == 1:
        logger.error("Fugacity test two is for multicomponent systems.")
    elif len(ind) != ncomp:
        logger.info(
            "There is not a significant amount of components {} in solution".format(
                np.setdiff1d(range(ncomp), ind)
            )
        )

    #    dy = step_size
    #    dphi = np.zeros((2,ncomp))
    #    for j, delta in enumerate((dy, -dy)):
    #        y_tmp = np.copy(xi)
    #        y_tmp[ind[0]] += delta
    #        y_tmp[ind[-1]] -= delta
    #        dphi[j,:] = np.log(Eos.fugacity_coefficient(P, rho, y_tmp, T))
    #    dphidx = (dphi[0] - dphi[1]) / (2.0 * dy)

    log_phi = np.zeros((2, ncomp))
    for i, factor in enumerate([1.0, (1 - fractional_change)]):
        log_phi[i, :] = np.log(Eos.fugacity_coefficient(P * factor, rho, xi, T))
    dphidx = log_phi[0] - log_phi[1]

    # Why fractional change not here?

    residual = np.sum(xi * dphidx)

    return residual


def activity_coefficient(P, T, xi, yi, Eos, **kwargs):
    r"""

    Calculation activity coefficient given T, P, yi, and xi.
    
    Parameters
    ----------
    P : float
        [Pa] Pressure of the system
    T : float
        [K] Temperature of the system
    xi : numpy.ndarray
        Liquid mole fraction of each component, sum(xi) should equal 1.0
    yi : numpy.ndarray
        Vapor mole fraction of each component, sum(xi) should equal 1.0
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    kwargs
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties`

    Returns
    -------
    activity_coefficient : numpy.ndarray
        Activity coefficient for given composition of mixtures
    Psat : numpy.ndarray
        Saturation pressure 
    """

    if len(kwargs) > 0:
        logger.debug(
            "'activity_coefficient' does not use the following keyword arguments: {}".format(
                ", ".join(list(kwargs.keys()))
            )
        )

    ncomp = len(xi)
    Psat = np.zeros(ncomp)
    for i in range(ncomp):
        tmp = np.zeros(ncomp)
        tmp[i] = 1.0
        Psat[i], _, _ = calc_saturation_properties(T, tmp, Eos, **kwargs)

    activity_coefficient = yi * P / (Psat * xi)

    return activity_coefficient, Psat
