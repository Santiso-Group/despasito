"""
    This thermo module contains a series of wrappers to handle the inputs and outputs of these functions.

    The 'calc' module contains the thermodynamic calculations. Calculation of pressure, fugacity_coefficient, and maximum allowed density are handled by an Eos object so that these functions can be used with any EOS. None of the functions in this folder need to be handled directly, as a function factory is included in our __init__.py file. Add "from thermodynamics import thermo" and use "thermo("calc_type",Eos,input_dict)" to get started.
    
"""

import numpy as np
import logging

from despasito.utils.parallelization import MultiprocessingJob
from . import calc
from despasito import fundamental_constants as constants
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


def bubble_pressure(Eos, **sys_dict):
    r"""
    Calculates pressure and vapor compositions given liquid mole fractions and temperature.

    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    xilist : list
        List of sets of component mole fraction, where sum(xi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Tlist : list, Optional, default=298.15
        [K] Temperature of the system corresponding to composition in xilist. If one set is given, this temperature will be used for all compositions.
    Pguess : list/float, Optional
        [Pa] Guess the system pressure at the bubble point. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmin : list, Optional
        [Pa] Set the upper bound for the minimum system pressure at the bubble point. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_bubble_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmax : list, Optional
        [Pa] Set the upper bound for the maximum system pressure at the bubble point. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_bubble_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties` and :func:`~despasito.thermodynamics.calc.calc_bubble_pressure`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated arrays  

        - T: Temperature array generated from given instructions
        - xi: Given array of liquid compositions
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - P: Pressure values evaluated for given conditions
        - yi: Vapor mole fraction evaluated for given conditions
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - obj: List of objective values

    """

    if "xilist" not in sys_dict:
        raise ValueError("Mole fractions, xilist, are not specified")
    elif not gtb.isiterable(sys_dict["xilist"][0]):
        sys_dict["xilist"] = np.array([sys_dict["xilist"]], float)

    ## Extract and check input data
    thermo_keys = ["Tlist", "xilist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    thermo_defaults = [constants.standard_temperature]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, "Tlist", thermo_defaults, lx=npoints)
    )

    ## Optional values
    optional_keys = ["Pguess", "Pmin", "Pmax"]
    opts = gtb.check_length_dict(sys_dict, optional_keys, lx=npoints)
    if opts:
        logger.info(
            "Accepted user defined variables: {}".format(", ".join(list(opts.keys())))
        )

    # Delete processed keys
    for key in thermo_keys + optional_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
        del sys_dict["MultiprocessingObject"]
    else:
        flag_use_mp_object = False

    opts.update(sys_dict)  # Add unprocessed options

    ## Calculate P and yi
    per_job_var = ["Pmin", "Pmax", "Pguess"]
    inputs = []
    for i in range(npoints):
        opts_tmp = opts.copy()
        for key in per_job_var:
            if key in opts_tmp:
                opts_tmp[key] = opts_tmp[key][i]
        inputs.append(
            (thermo_dict["Tlist"][i], thermo_dict["xilist"][i], Eos, opts_tmp)
        )

    if flag_use_mp_object:
        P_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingObject.pool_job(
            _bubble_pressure_wrapper, inputs
        )
    else:
        P_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(
            _bubble_pressure_wrapper, inputs
        )

    logger.info("--- Calculation bubble_pressure Complete ---")

    return {
        "T": thermo_dict["Tlist"],
        "xi": thermo_dict["xilist"],
        "P": P_list,
        "yi": yi_list,
        "flagl": flagl_list,
        "flagv": flagv_list,
        "obj": obj_list,
    }


def _bubble_pressure_wrapper(args):
    r""" Wrapper for parallelized use of 'bubble_pressure' calculation type.
    """

    T, xi, Eos, opts = args
    logger.info("T (K), xi: {} {}, Let's Begin!".format(T, xi))

    try:
        if len(xi[xi != 0.0]) == 1:
            P, _, _ = calc.calc_saturation_properties(T, xi, Eos, **opts)
            yi, flagv, flagl, obj = xi, 0, 1, 0.0
        else:
            if "pressure_options" in opts and "method" in opts["pressure_options"]:
                opts["method"] = opts["pressure_options"]["method"]
                del opts["pressure_options"]["method"]
            P, yi, flagv, flagl, obj = calc.calc_bubble_pressure(xi, T, Eos, **opts)
    except Exception:
        logger.warning(
            "T (K), xi: {} {}, calculation did not produce a valid result.".format(
                T, xi
            )
        )
        logger.debug("Calculation Failed:", exc_info=True)
        P, yi, flagl, flagv, obj = [np.nan, np.nan * np.ones(len(xi)), 3, 3, np.nan]

    logger.info("P (Pa), yi: {} {}".format(P, yi))

    return P, yi, flagv, flagl, obj


def dew_pressure(Eos, **sys_dict):

    r"""
    Calculate pressure and liquid composition given vapor mole fractions, yi, and temperature.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    yilist : list
        List of sets of component mole fraction, where sum(yi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Tlist : list, Optional, default=298.15
        [K] Temperature of the system corresponding to composition in yilist. If one set is given, this pressure will be used for all compositions.
    Pguess : list/float, Optional
        [Pa] Guess the system pressure at the dew point. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmin : list, Optional
        [Pa] Set the upper bound for the minimum system pressure at the dew point. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_dew_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmax : list, Optional
        [Pa] Set the upper bound for the maximum system pressure at the dew point. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_dew_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties` and :func:`~despasito.thermodynamics.calc.calc_dew_pressure`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - yi: Given array of vapor compositions
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - P: Pressure values evaluated for given conditions
        - xi: Liquid mole fraction evaluated for given conditions
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - obj: List of objective values

    """

    if "yilist" not in sys_dict:
        raise ValueError("Mole fractions, yilist, are not specified")
    elif not gtb.isiterable(sys_dict["yilist"][0]):
        sys_dict["yilist"] = np.array([sys_dict["yilist"]], float)

    ## Extract and check input data
    thermo_keys = ["Tlist", "yilist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    thermo_defaults = [constants.standard_temperature]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, "Tlist", thermo_defaults, lx=npoints)
    )

    ## Optional values
    optional_keys = ["Pguess", "Pmin", "Pmax"]
    opts = gtb.check_length_dict(sys_dict, optional_keys, lx=npoints)
    if opts:
        logger.info(
            "Accepted user defined variables: {}".format(", ".join(list(opts.keys())))
        )

    # Delete processed keys
    for key in thermo_keys + optional_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts.update(sys_dict)  # Add unprocessed options

    ## Calculate P and yi
    per_job_var = ["Pguess", "Pmin", "Pmax"]
    inputs = []
    for i in range(npoints):
        opts_tmp = opts.copy()
        for key in per_job_var:
            if key in opts_tmp:
                opts_tmp[key] = opts_tmp[key][i]
        inputs.append(
            (thermo_dict["Tlist"][i], thermo_dict["yilist"][i], Eos, opts_tmp)
        )

    if flag_use_mp_object:
        P_list, xi_list, flagv_list, flagl_list, obj_list = MultiprocessingObject.pool_job(
            _dew_pressure_wrapper, inputs
        )
    else:
        P_list, xi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(
            _dew_pressure_wrapper, inputs
        )

    logger.info("--- Calculation dew_pressure Complete ---")

    return {
        "T": thermo_dict["Tlist"],
        "xi": xi_list,
        "P": P_list,
        "yi": thermo_dict["yilist"],
        "flagl": flagl_list,
        "flagv": flagv_list,
        "obj": obj_list,
    }


def _dew_pressure_wrapper(args):
    r""" Wrapper for parallelized use of 'dew_pressure' calculation type.
    """

    T, yi, Eos, opts = args
    logger.info("T (K), yi: {} {}, Let's Begin!".format(T, yi))

    try:
        if len(yi[yi != 0.0]) == 1:
            P, _, _ = calc.calc_saturation_properties(T, yi, Eos, **opts)
            xi, flagv, flagl, obj = yi, 0, 1, 0.0
        else:
            if "pressure_options" in opts and "method" in opts["pressure_options"]:
                opts["method"] = opts["pressure_options"]["method"]
                del opts["pressure_options"]["method"]
            P, xi, flagl, flagv, obj = calc.calc_dew_pressure(yi, T, Eos, **opts)
    except Exception:
        logger.warning(
            "T (K), yi: {} {}, calculation did not produce a valid result.".format(
                T, yi
            )
        )
        logger.debug("Calculation Failed:", exc_info=True)
        P, xi, flagl, flagv, obj = [np.nan, np.nan * np.ones(len(yi)), 3, 3, np.nan]

    logger.info("P (Pa), xi: {} {}".format(P, xi))

    return P, xi, flagv, flagl, obj


def activity_coefficient(Eos, **sys_dict):

    r"""
    Calculate the activity coefficient with a variety of methods.

    Obtaining the activity coefficient requires the temperature, pressure, vapor composition, and liquid composition, but if one is missing, the necessary calculation computes it.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    Tlist : list, Optional, default=298.15
        [K] Temperature of the system corresponding to composition in yilist. If one set is given, this pressure will be used for all compositions.
    xilist : list, Optional
        List of sets of component mole fraction, where sum(yi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures. If this value is not provided, a dew point calculation is inititated if the pressure is also absent, and a flash calculation if yilist is absent.
    yilist : list, Optional
        List of sets of component mole fraction, where sum(yi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures. If this value is not provided, a bubble point calculation is inititated if the pressure is also absent, and a flash calculation if xilist is absent.
    Plist : list, Optional
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures. If this value is not provided, a dew or bubble point calculation is inititated to find it.
    Pguess : list/float, Optional
        [Pa] If the pressure and either xilist or yilist is not provided, guess the system pressure at the dew/bubble point respectively. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmin : list, Optional
        [Pa] If the pressure and either xilist or yilist is not provided, set a minimum bounds in pressure for the dew/bubble point respectively. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_dew_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    Pmax : list, Optional
        [Pa] If the pressure and either xilist or yilist is not provided, set a maximum bounds in pressure for the dew/bubble point respectively. If not defined, the default in :func:`~despasito.thermodynamics.calc.calc_dew_pressure` is used. If a list of length 1 is provided, that value is used for all temperature-composition sets given.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties` and either :func:`~despasito.thermodynamics.calc.calc_dew_pressure`, :func:`~despasito.thermodynamics.calc.calc_bubble_pressure`, or :func:`~despasito.thermodynamics.calc.calc_flash`, depending on which two of xi, yi, or P is missing.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Given or calculated array of pressures
        - xi: Given or calculated array of liquid compositions
        - yi: Given or calculated array of vapor compositions
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - obj: List of objective values if a calculation was needed
        - Psat: List of calculated saturation pressures
        - gamma: List of calculated activity coefficients

    """

    if "xilist" in sys_dict and not gtb.isiterable(sys_dict["xilist"][0]):
        sys_dict["xilist"] = np.array([sys_dict["xilist"]], float)
    if "yilist" in sys_dict and not gtb.isiterable(sys_dict["yilist"][0]):
        sys_dict["yilist"] = np.array([sys_dict["yilist"]], float)

    ## Extract and check input data
    thermo_keys = ["Tlist", "yilist", "xilist", "Plist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    ## Determine Mode
    if "xilist" not in thermo_dict:
        if "yilist" not in thermo_dict:
            if "Plist" in thermo_dict:
                mode = "flash"
            else:
                mode = None
        else:
            if "Plist" in thermo_dict:
                thermodict["Pguess"] = thermo_dict["Plist"]
                logger.info("Using given pressure as initial guess")
            mode = "dew_point"
    elif "yilist" not in thermo_dict:
        if "Plist" in thermo_dict:
            thermo_dict["Pguess"] = thermo_dict["Plist"]
            logger.info("Using given pressure as initial guess")
        mode = "bubble_point"
    else:
        mode = "standard"

    if mode == None:
        raise ValueError(
            "Two of the following system properties must be provided: Tlist, Plist, xilist, or yilist"
        )
    else:
        logger.info("Activity coefficient being calculated in {} mode.".format(mode))

    ## Set Defaults
    thermo_defaults = [
        constants.standard_temperature,
        np.nan * np.ones(Eos.number_of_components),
        np.nan * np.ones(Eos.number_of_components),
        np.nan,
    ]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    ## Optional values
    optional_keys = ["Pguess", "Pmin", "Pmax"]
    opts = gtb.check_length_dict(sys_dict, optional_keys, lx=npoints)
    if opts:
        logger.info(
            "Accepted user defined variables: {}".format(", ".join(list(opts.keys())))
        )

    # Delete processed keys
    for key in thermo_keys + optional_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts.update(sys_dict)  # Add unprocessed options

    ## Calculate P and yi
    per_job_var = ["Pguess", "Pmin", "Pmax"]
    inputs = []
    for i in range(npoints):
        opts_tmp = opts.copy()
        for key in per_job_var:
            if key in opts_tmp:
                opts_tmp[key] = opts_tmp[key][i]
        inputs.append(
            (
                thermo_dict["Tlist"][i],
                thermo_dict["Plist"][i],
                thermo_dict["xilist"][i],
                thermo_dict["yilist"][i],
                Eos,
                opts_tmp,
                mode,
            )
        )

    if flag_use_mp_object:
        tmp = MultiprocessingObject.pool_job(_activity_coefficient_wrapper, inputs)
    else:
        tmp = MultiprocessingJob.serial_job(_activity_coefficient_wrapper, inputs)
    T_list, P_list, xi_list, yi_list, flagv_list, flagl_list, obj_list, Psat_list, gamma_list = (
        tmp
    )

    logger.info("--- Calculation dew_pressure Complete ---")

    return {
        "T": T_list,
        "xi": xi_list,
        "P": P_list,
        "yi": yi_list,
        "flagl": flagl_list,
        "flagv": flagv_list,
        "obj": obj_list,
        "Psat": Psat_list,
        "gamma": gamma_list,
    }


def _activity_coefficient_wrapper(args):
    r""" Wrapper for parallelized use of 'dew_pressure' calculation type.
    """

    T, P, xi, yi, Eos, opts, mode = args
    logger.info("T (K), yi: {} {}, Let's Begin!".format(T, yi))

    try:
        if "pressure_options" in opts and "method" in opts["pressure_options"]:
            opts["method"] = opts["pressure_options"]["method"]
            del opts["pressure_options"]["method"]

        if mode == "dew_point":
            P, xi, flagl, flagv, obj = calc.calc_dew_pressure(yi, T, Eos, **opts)
        elif mode == "bubble_point":
            P, yi, flagv, flagl, obj = calc.calc_bubble_pressure(xi, T, Eos, **opts)
        elif mode == "flash":
            xi, flagl, yi, flagv, obj = calc.calc_flash(P, T, Eos, **opts)
        elif mode == "standard":
            flagl, flagv, obj = [np.nan, np.nan, np.nan]

        gamma, Psat = calc.activity_coefficient(P, T, xi, yi, Eos, **opts)

    except Exception:
        logger.warning(
            "In calculating the activity coefficient, {} calculation with T, P, xi, and yi inputs: {}, {}, {}, {} did not produce a valid result.".format(
                mode, T, P, xi, yi
            )
        )
        logger.debug("Calculation Failed:", exc_info=True)

        if mode == "dew_point":
            P, xi, flagl, flagv, obj, Psat, gamma = [
                np.nan,
                np.nan * np.ones(len(yi)),
                3,
                3,
                np.nan,
                np.nan,
                np.nan,
            ]
        elif mode == "bubble_point":
            P, yi, flagl, flagv, obj, Psat, gamma = [
                np.nan,
                np.nan * np.ones(len(xi)),
                3,
                3,
                np.nan,
                np.nan,
                np.nan,
            ]
        elif mode == "flash":
            xi, yi, flagl, flagv, obj, Psat, gamma = [
                np.nan * np.ones(Eos.number_of_components),
                np.nan * np.ones(Eos.number_of_components),
                3,
                3,
                np.nan,
                np.nan,
                np.nan,
            ]
        elif mode == "standard":
            flagl, flagv, obj, Psat, gamma = [np.nan, np.nan, np.nan, np.nan, np.nan]

    logger.info("Psat (Pa), activity coefficient: {} {}".format(Psat, gamma))

    return T, P, xi, yi, flagv, flagl, obj, Psat, gamma


def flash(Eos, **sys_dict):

    r"""
    Flash calculation of vapor and liquid mole fractions. Only binary systems are currently supported

    Input and system information are assessed first. An output file is generated with T, yi, and corresponding P and xi.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    Tlist : list, Optional, default=298.15
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    Plist : list, Optional, default=101325.0
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_flash`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Pressure array generated from given instructions
        - xi: Given array of liquid compositions
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - yi: Vapor mole fraction evaluated for given conditions
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - obj: List of objective values

    """

    ## Extract and check input data
    thermo_keys = ["Tlist", "Plist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    thermo_defaults = [constants.standard_temperature, constants.standard_pressure]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    if Eos.number_of_components != 2:
        raise ValueError(
            "Only binary systems are currently supported for flash calculations, {} were given.".format(
                Eos.number_of_components
            )
        )

    inputs = [
        (thermo_dict["Tlist"][i], thermo_dict["Plist"][i], Eos, opts)
        for i in range(npoints)
    ]

    if flag_use_mp_object:
        xi_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingObject.pool_job(
            _flash_wrapper, inputs
        )
    else:
        xi_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(
            _flash_wrapper, inputs
        )

    logger.info("--- Calculation flash Complete ---")

    return {
        "T": thermo_dict["Tlist"],
        "xi": xi_list,
        "P": thermo_dict["Plist"],
        "yi": yi_list,
        "flagl": flagl_list,
        "flagv": flagv_list,
        "obj": obj_list,
    }


def _flash_wrapper(args):
    r""" Wrapper for parallelized use of 'flash' calculation type.
    """

    T, P, Eos, opts = args

    try:
        xi, flagl, yi, flagv, obj = calc.calc_flash(P, T, Eos, **opts)
    except Exception:
        logger.warning(
            "T (K), P (Pa): {} {}, calculation did not produce a valid result.".format(
                T, P
            )
        )
        logger.debug("Calculation Failed:", exc_info=True)
        xi, yi, flagl, flagv, obj = [
            np.nan * np.ones(Eos.number_of_components),
            np.nan * np.ones(Eos.number_of_components),
            3,
            3,
            np.nan,
        ]

    logger.info("xi: {}, yi: {}".format(xi, yi))

    return xi, yi, flagv, flagl, obj


def saturation_properties(Eos, **sys_dict):

    r"""
    Computes the saturated pressure, liquid, and gas density a one component phase at a temperature.

    Input and system information are assessed first.  An output file is generated with T, :math:`P^{sat}`, :math:`\rho^{sat}_{l}, :math:`\rho^{sat}_{v}
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    Tlist : list, Optional, default=298.15 
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_saturation_properties`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - Psat: Saturation pressure
        - rhol: Liquid saturation density
        - rholv: Vapor saturation density

    """

    ## Extract and check input data
    thermo_keys = ["Tlist", "xilist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    if "Tlist" not in thermo_dict:
        thermo_dict["Tlist"] = np.array([constants.standard_temperature])
        logger.info(
            "Assuming standard temperature, {}".format(constants.standard_temperature)
        )
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    thermo_defaults = [
        constants.standard_temperature,
        np.array([[1.0] for x in range(npoints)]),
    ]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    inputs = [
        (thermo_dict["Tlist"][i], thermo_dict["xilist"][i], Eos, opts)
        for i in range(npoints)
    ]
    if flag_use_mp_object:
        Psat, rholsat, rhovsat = MultiprocessingObject.pool_job(
            _saturation_properties_wrapper, inputs
        )
    else:
        Psat, rholsat, rhovsat = MultiprocessingJob.serial_job(
            _saturation_properties_wrapper, inputs
        )

    logger.info("--- Calculation saturation_properties Complete ---")

    return {"T": thermo_dict["Tlist"], "Psat": Psat, "rhol": rholsat, "rhov": rhovsat}


def _saturation_properties_wrapper(args):
    r""" Wrapper for parallelized use of 'saturation_properties' calculation type.
    """

    T, xi, Eos, opts = args

    logger.info("T (K), xi: {} {}, Let's Begin!".format(T, xi))

    try:
        Psat, rholsat, rhovsat = calc.calc_saturation_properties(T, xi, Eos, **opts)
        if np.isnan(Psat):
            logger.warning(
                "T (K), xi: {} {}, calculation did not produce a valid result.".format(
                    T, xi
                )
            )
            logger.debug("Calculation Failed:", exc_info=True)
            Psat, rholsat, rhovsat = [np.nan, np.nan, np.nan]
        else:
            logger.info("Psat {} Pa, rhol {}, rhov {}".format(Psat, rholsat, rhovsat))
    except Exception:
        logger.warning(
            "T (K), xi: {} {}, calculation did not produce a valid result.".format(
                T, xi
            )
        )
        logger.debug("Calculation Failed:", exc_info=True)
        Psat, rholsat, rhovsat = [np.nan, np.nan, np.nan]

    return Psat, rholsat, rhovsat


def liquid_properties(Eos, **sys_dict):

    r"""
    Computes the liquid density and chemical potential given a temperature, pressure, and liquid mole fractions.

    Input and system information are assessed first. An output file is generated with P, T, xi, :math:`\rho_{l}, and :math:`\phi_{l}.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    xilist : list
        List of sets of component mole fraction, where sum(xi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Tlist : list, Optional default=298.15 
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    Plist : list, Optional, default=101325.0 
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_liquid_fugacity_coefficient`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Pressure array generated from given instructions
        - xi: Composition generated from given instructions
        - rhol: Evaluated liquid density
        - phil: Evaluated liquid fugacity coefficient
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas

    """

    ## Extract and check input data
    thermo_keys = ["Tlist", "xilist", "Plist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    if "xilist" not in thermo_dict and Eos.number_of_components > 1:
        raise ValueError("Ambiguous mixture composition. Define xilist")

    thermo_defaults = [
        constants.standard_temperature,
        np.array([[1.0] for x in range(npoints)]),
        constants.standard_pressure,
    ]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    inputs = [
        (
            thermo_dict["Plist"][i],
            thermo_dict["Tlist"][i],
            thermo_dict["xilist"][i],
            Eos,
            opts,
        )
        for i in range(npoints)
    ]
    if flag_use_mp_object:
        rhol, phil, flagl = MultiprocessingObject.pool_job(
            _liquid_properties_wrapper, inputs
        )
    else:
        rhol, phil, flagl = MultiprocessingJob.serial_job(
            _liquid_properties_wrapper, inputs
        )

    logger.info("--- Calculation liquid_properties Complete ---")

    return {
        "P": thermo_dict["Plist"],
        "T": thermo_dict["Tlist"],
        "xi": thermo_dict["xilist"],
        "rhol": rhol,
        "phil": phil,
        "flagl": flagl,
    }


def _liquid_properties_wrapper(args):
    r""" Wrapper for parallelized use of 'liquid_properties' calculation type.
    """

    P, T, xi, Eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    try:
        phil, rhol, flagl = calc.calc_liquid_fugacity_coefficient(P, T, xi, Eos, **opts)
        logger.info(
            "P {} Pa, T {} K, xi {}, rhol {}, phil {}, flagl {}".format(
                P, T, xi, rhol, phil, flagl
            )
        )
    except Exception:
        logger.warning("Failed to calculate rhol at {} K and {} Pa".format(T, P))
        rhol, flagl = np.nan, 3
        phil = np.nan * np.ones(Eos.number_of_components)
        logger.debug("Calculation Failed:", exc_info=True)

    return rhol, phil, flagl


def vapor_properties(Eos, **sys_dict):

    r"""
    Computes the vapor density and chemical potential given a temperature, pressure, and vapor mole fractions.

    Input and system information are assessed first. An output file is generated with P, T, yi, :math:`\rho_{v}, and :math:`\phi_{v}.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    yilist: : list
        List of sets of component mole fraction, where sum(yi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Tlist : list, Optional, default=298.15 
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    Plist : list, Optional, default=101325.0
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_vapor_fugacity_coefficient`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Pressure array generated from given instructions
        - yi: Composition generated from given instructions
        - rhov: Evaluated vapor density
        - phiv: Evaluated vapor fugacity coefficient
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas

    """

    ## Extract and check input data
    thermo_keys = ["Tlist", "yilist", "Plist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    if "yilist" not in thermo_dict and Eos.number_of_components > 1:
        raise ValueError("Ambiguous mixture composition. Define yilist")

    thermo_defaults = [
        constants.standard_temperature,
        np.array([[1.0] for x in range(npoints)]),
        constants.standard_pressure,
    ]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    inputs = [
        (
            thermo_dict["Plist"][i],
            thermo_dict["Tlist"][i],
            thermo_dict["yilist"][i],
            Eos,
            opts,
        )
        for i in range(npoints)
    ]
    if flag_use_mp_object:
        rhov, phiv, flagv = MultiprocessingObject.pool_job(
            _vapor_properties_wrapper, inputs
        )
    else:
        rhov, phiv, flagv = MultiprocessingJob.serial_job(
            _vapor_properties_wrapper, inputs
        )

    logger.info("--- Calculation vapor_properties Complete ---")

    return {
        "P": thermo_dict["Plist"],
        "T": thermo_dict["Tlist"],
        "yi": thermo_dict["yilist"],
        "rhov": rhov,
        "phiv": phiv,
        "flagv": flagv,
    }


def _vapor_properties_wrapper(args):
    r""" Wrapper for parallelized use of 'vapor_properties' calculation type.
    """

    P, T, yi, Eos, opts = args

    logger.info("T (K), P (Pa), yi: {} {} {}, Let's Begin!".format(T, P, yi))

    try:
        phiv, rhov, flagv = calc.calc_vapor_fugacity_coefficient(P, T, yi, Eos, **opts)
        logger.info(
            "P {} Pa, T {} K, yi {}, rhov {}, phiv {}, flagv {}".format(
                P, T, yi, rhov, phiv, flagv
            )
        )
    except Exception:
        logger.warning("Failed to calculate rhov at {} K and {} Pa".format(T, P))
        rhov, flagv = np.nan, 3
        phiv = np.nan * np.ones(Eos.number_of_components)
        logger.debug("Calculation Failed:", exc_info=True)

    return rhov, phiv, flagv


def solubility_parameter(Eos, **sys_dict):

    r"""
    Calculate the Hildebrand solubility parameter based on temperature and composition. This function is based on the method used in Zeng, Z., Y. Xi, and Y. Li "Calculation of Solubility Parameter Using Perturbed-Chain SAFT and Cubic-Plus-Association Equations of State" Ind. Eng. Chem. Res. 2008, 47, 9663–9669.

    Input and system information are assessed first. An output file is generated with T, xi, :math:`\rho_{l}`, and :math:`\delta`.
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    Tlist : list, Optional, default=298.15 
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    xilist : list, Optional, default=[1.0] 
        Default assumes all of one component. List of sets of component mole fraction, where sum(xi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Plist : list, Optional, default=101325.0
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_liquid_density` and :func:`~despasito.thermodynamics.calc.calc_hildebrand_solubility`

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Pressure array generated from given instructions
        - xi: Composition generated from given instructions
        - rhol: Evaluated liquid density
        - delta: Hidebrand solubility parameter given system conditions

    """

    ## Extract and check input data
    thermo_keys = ["Tlist", "xilist", "Plist"]
    if any([x in sys_dict for x in thermo_keys]):
        thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
        npoints = len(thermo_dict[list(thermo_dict.keys())[0]])
    else:
        npoints = 1

    if "xilist" not in thermo_dict and Eos.number_of_components > 1:
        raise ValueError("Ambiguous mixture composition. Define xilist")

    thermo_defaults = [
        constants.standard_temperature,
        np.array([[1.0] for x in range(npoints)]),
        constants.standard_pressure,
    ]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, thermo_keys, thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    ## Calculate solubility parameter
    inputs = [
        (
            thermo_dict["Plist"][i],
            thermo_dict["Tlist"][i],
            thermo_dict["xilist"][i],
            Eos,
            opts,
        )
        for i in range(npoints)
    ]
    if flag_use_mp_object:
        rhol, flagl, delta = MultiprocessingObject.pool_job(
            _solubility_parameter_wrapper, inputs
        )
    else:
        rhol, flagl, delta = MultiprocessingJob.serial_job(
            _solubility_parameter_wrapper, inputs
        )

    logger.info("--- Calculation solubility_parameter Complete ---")

    return {
        "P": thermo_dict["Plist"],
        "T": thermo_dict["Tlist"],
        "xi": thermo_dict["xilist"],
        "rhol": rhol,
        "delta": delta,
    }


def _solubility_parameter_wrapper(args):
    r""" Wrapper for parallelized use of 'solubility_parameter' calculation type.
    """

    P, T, xi, Eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    try:
        rhol, flagl = calc.calc_liquid_density(P, T, xi, Eos, **opts)
        delta = calc.hildebrand_solubility(rhol, xi, T, Eos, **opts)
        logger.info(
            "P {} Pa, T {} K, xi {}, rhol {}, flagl {}, delta {}".format(
                P, T, xi, rhol, flagl, delta
            )
        )
    except Exception:
        logger.warning("Failed to calculate rhov at {} K and {} Pa".format(T, P))
        rhol, flagl, delta = np.nan, 3, np.nan
        logger.debug("Calculation Failed:", exc_info=True)

    return rhol, flagl, delta


def verify_eos(Eos, **sys_dict):

    r"""
    The following consistency checks are performed to ensure the calculated fugacity coefficients are thermodynamically consistent.

    - 1. d(log phi) / dP = (Z - 1)/P
    - 

    TODO: Finish documentation
    
    Parameters
    ----------
    Eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    Tlist : list, Optional, default=298.15 
        [K] Temperature of the system corresponding Plist. If one value is given, this temperature will be used for all temperatures.
    xilist : list, Optional
        Default array of 11 values from x1=0 to x1=1 for binary array. List of sets of component mole fraction, where sum(xi)=1.0 for each set. Each set of components corresponds to a temperature in Tlist, or if one set is given, this composition will be used for all temperatures.
    Plist : list, Optional, default=101325.0 
        [Pa] Pressure of the system corresponding to Tlist. If one value is given, this pressure will be used for all temperatures.
    MultiprocessingObject : obj, Optional
        Multiprocessing object, :class:`~despasito.utils.parallelization.MultiprocessingJob`
    kwargs, Optional
        Keyword arguments for :func:`~despasito.thermodynamics.calc.calc_vapor_density`, :func:`~despasito.thermodynamics.calc.calc_liquid_density`, :func:`~despasito.thermodynamics.calc.calc_fugacity_test_1`, and :func:`~despasito.thermodynamics.calc.calc_fugacity_test_2`,

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values

        - T: Temperature array generated from given instructions
        - P: Pressure array generated from given instructions
        - xi: Composition generated from given instructions
        - residual_v1: NoteHere
        - residual_v2: NoteHere
        - flagv: Phase flag for vapor given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - log_phivi: Log of the partial vapor fugacity coefficient for each component
        - residual_l1: NoteHere
        - residual_l2: NoteHere
        - flagl: Phase flag for liquid given temperature and calculated pressure, a value of 0 is vapor, 1 is liquid, 2 mean a critical fluid, 3 means that neither is true, 4 means we should assume ideal gas
        - log_phili: Log of the partial liquid fugacity coefficient for each component

    """

    ## Extract and check input data
    if "xilist" not in sys_dict and Eos.number_of_components == 2:
        tmp = np.linspace(0, 1, 11)
        xi_list = np.array([[x, 1.0 - x] for x in tmp])
        logger.info("Use array of mole fractions")
    else:
        raise ValueError(
            "Must have at least 2 components. With more that 2 components, the mole fractions need to be specified"
        )

    thermo_keys = ["Tlist", "xilist", "Plist"]
    thermo_dict = gtb.check_length_dict(sys_dict, thermo_keys)
    npoints = len(thermo_dict[list(thermo_dict.keys())[0]])

    thermo_defaults = [constants.standard_temperature, constants.standard_pressure]
    thermo_dict.update(
        gtb.set_defaults(thermo_dict, ["Tlist", "Plist"], thermo_defaults, lx=npoints)
    )

    # Delete processed keys
    for key in thermo_keys:
        if key in sys_dict:
            del sys_dict[key]

    if "MultiprocessingObject" in sys_dict:
        MultiprocessingObject = sys_dict["MultiprocessingObject"]
        del sys_dict["MultiprocessingObject"]
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    opts = sys_dict.copy()

    inputs = [
        (
            thermo_dict["Plist"],
            thermo_dict["Tlist"][i],
            thermo_dict["xilist"][i],
            Eos,
            opts,
        )
        for i in range(npoints)
    ]
    if flag_use_mp_object:
        residual_v1, residual_v2, flagv, log_phiv, residual_l1, residual_l2, flagl, log_phil = MultiprocessingObject.pool_job(
            _verify_eos_wrapper, inputs
        )
    else:
        residual_v1, residual_v2, flagv, log_phiv, residual_l1, residual_l2, flagl, log_phil = MultiprocessingJob.serial_job(
            _verify_eos_wrapper, inputs
        )

    logger.info("--- Calculation verify_eos Complete ---")

    return {
        "P": thermo_dict["Plist"],
        "T": thermo_dict["Tlist"],
        "xi": thermo_dict["xilist"],
        "residual_v1": residual_v1,
        "residual_v2": residual_v2,
        "flagv": flagv,
        "log_phivi": log_phiv,
        "residual_l1": residual_l1,
        "residual_l2": residual_l2,
        "flagl": flagl,
        "log_phili": log_phil,
    }


def _verify_eos_wrapper(args):
    r""" Wrapper for parallelized use of 'verify_eos' calculation type.
    """

    P, T, xi, Eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    rhov, flagv = calc.calc_vapor_density(P, T, xi, Eos, **opts)
    if np.isnan(rhov):
        logger.warning("Failed to calculate rhov at {} K and {} Pa".format(T, P))
        log_phiv, residual_v1, residual_v2 = np.nan, np.nan, np.nan
        logger.debug("Calculation Failed:", exc_info=True)
    else:
        phiv = Eos.fugacity_coefficient(P, np.array([rhov]), xi, T)
        log_phiv = np.log(phiv)
        residual_v1 = calc.fugacity_test_1(P, T, xi, rhov, Eos, **opts)
        residual_v2 = calc.fugacity_test_2(P, T, xi, rhov, Eos, **opts)
        logger.info(
            "rhov {}, flagv {}, log_phiv {}, log_phiv {}, residual1 {}, residual2 {}".format(
                rhov, flagv, np.sum(xi * log_phiv), log_phiv, residual_v1, residual_v2
            )
        )

    rhol, flagl = calc.calc_liquid_density(P, T, xi, Eos, **opts)
    if np.isnan(rhol):
        logger.warning("Failed to calculate rhol at {} K and {} Pa".format(T, P))
        log_phil, residual_l1, residual_l2 = np.nan, np.nan, np.nan
        logger.debug("Calculation Failed:", exc_info=True)
    else:
        phil = Eos.fugacity_coefficient(P, np.array([rhol]), xi, T)
        log_phil = np.log(phil)
        residual_l1 = calc.fugacity_test_1(P, T, xi, rhol, Eos, **opts)
        residual_l2 = calc.fugacity_test_2(P, T, xi, rhol, Eos, **opts)
        logger.info(
            "rhol {}, flagl {}, log_phil {}, log_phil {}, residual1 {}, residual2 {}".format(
                rhol, flagl, np.sum(xi * log_phil), log_phil, residual_l1, residual_l2
            )
        )

    return (
        residual_v1,
        residual_v2,
        flagv,
        log_phiv,
        residual_l1,
        residual_l2,
        flagl,
        log_phil,
    )
