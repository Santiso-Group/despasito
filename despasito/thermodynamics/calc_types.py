"""
    This thermo module contains a series of wrappers to handle the inputs and outputs of these functions. The `calc` module contains the thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS.
    
    None of the functions in this folder need to be handled directly, as a function factory is included in our __init__.py file. Add "from thermodynamics import thermo" and use "thermo("calc_type",eos,input_dict)" to get started.
    
"""

import numpy as np
import logging
from despasito.utils.parallelization import MultiprocessingJob

from . import calc
from despasito.utils.parallelization import batch_jobs

logger = logging.getLogger(__name__)

######################################################################
#                                                                    #
#                Phase Equilibrium given xi and T                    #
#                                                                    #
######################################################################
def phase_xiT(eos, sys_dict):

    r"""
    Calculate phase diagram given liquid mole fractions, xi, and temperature.

    Input and system information are assessed first. An output file is generated with T, xi, and corresponding P and yi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values    
    """

    #computes P and yi from xi and T

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist") 

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'],float)
        logger.info("Using xilist")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        Pguess = np.array(sys_dict['Pguess'],float)
        if np.size(T_list) != np.size(Pguess):
            if type(Pguess) not in [list, numpy.ndarray]:
                opts["Pguess"] = np.ones(len(T_list))*Pguess
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            elif len(T_list) == 1:
                opts["Pguess"] = np.ones(len(T_list))*float(Pguess[0])
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            else:
                raise ValueError("The number of provided pressure and mole fraction sets are different")
        logger.info("Using user defined initial guess has been provided")
    else:
        if 'CriticalProp' in sys_dict:
            CriticalProp = np.array(sys_dict['CriticalProp'])
            logger.info("Using critical properties to intially guess pressure")

            # Critical properties: [Tc, Pc, omega, rho_0.7, Zc, Vc, M]
            Pguess = calc.calc_CC_Pguess(xi_list, T_list, CriticalProp)
            if np.isnan(Pguess):
                logger.info("Critical properties were not used to guess an initial pressure")
            else:
                logger.info("Pguess: ", Pguess)
                opts["Pguess"] = Pguess

    # Extract desired method
    if "method" in sys_dict:
        logger.info("Accepted optimization method, {}, for solving pressure".format(sys_dict['method']))
        opts["meth"] = sys_dict['method']

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    # Extract pressure optimization dict
    if "pressure options" in sys_dict:
        logger.info("Accepted options for P optimization")
        opts["pressure_opts"] = sys_dict["pressure options"]

    # Extract pressure optimization dict
    if "mole fraction options" in sys_dict:
        logger.info("Accepted options for mole fraction optimization")
        opts["zi_opts"] = sys_dict["mole fraction options"]

    ## Calculate P and yi

    if 'Pguess' in sys_dict:
        inputs = [(T_list[i], xi_list[i], eos, opts, Pguess[i]) for i in range(len(T_list))]
    else:
        inputs = [(T_list[i], xi_list[i], eos, opts) for i in range(len(T_list))]

    if flag_use_mp_object:
        P_list, yi_list, flagv_list, flagl_list, obj_list = mpObj.pool_job(_phase_xiT_wrapper, inputs)
    else:
        P_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(_phase_xiT_wrapper, inputs)

    logger.info("--- Calculation phase_xiT Complete ---")

    return {"T":T_list,"xi":xi_list,"P":P_list,"yi":yi_list,"flagl":flagl_list,"flagv":flagv_list,"obj":obj_list}

def _phase_xiT_wrapper(args):

    if len(args) == 4:
        T, xi, eos, opts = args
    elif len(args) == 5:
        T, xi, eos, opts, Pguess = args
        opts["Pguess"] = Pguess

    logger.info("T (K), xi: {} {}, Let's Begin!".format(T, xi))
    try:
        if len(xi[xi!=0.])==1:
            if "rhodict" in opts:
                opt_tmp = {"rhodict": opts["rhodict"]}
            else:
                opt_tmp = {}
            P, _, _ = calc.calc_Psat(T, xi, eos, **opt_tmp)
            yi, flagv, flagl, obj = xi, 0, 1, 0.0
        else:
            P, yi, flagv, flagl, obj = calc.calc_xT_phase(xi, T, eos, **opts)
    except:
        logger.warning("T (K), xi: {} {}, calculation did not produce a valid result.".format(T, xi))
        logger.debug("Calculation Failed:", exc_info=True)
        P, yi, flagl, flagv, obj = [np.nan, np.nan*np.ones(len(xi)), 3, 3, np.nan]

    logger.info("P (Pa), yi: {} {}".format(P, yi)) 

    return P, yi, flagv, flagl, obj


######################################################################
#                                                                    #
#                Phase Equilibria given yi and T                     #
#                                                                    #
######################################################################
def phase_yiT(eos, sys_dict):

    r"""
    Calculate phase diagram given vapor mole fractions, yi, and temperature.

    Input and system information are assessed first. An output file is generated with T, yi, and corresponding P and xi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")

    if 'yilist' in sys_dict:
        yi_list = np.array(sys_dict['yilist'],float)
        logger.info("Using yilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["yi_list", "T_list"]]):
        raise ValueError('Tlist or yilist are not specified')

    if np.size(T_list) != np.size(yi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(yi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        Pguess = np.array(sys_dict['Pguess'],float)
        if np.size(T_list) != np.size(Pguess):
            if type(Pguess) not in [list, numpy.ndarray]:
                opts["Pguess"] = np.ones(len(T_list))*Pguess
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            elif len(T_list) == 1:
                opts["Pguess"] = np.ones(len(T_list))*float(Pguess[0])
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            else:
                raise ValueError("The number of provided pressure and mole fraction sets are different")
        logger.info("Using user defined initial guess has been provided")
    else:
        if 'CriticalProp' in sys_dict:
            CriticalProp = np.array(sys_dict['CriticalProp'])
            logger.info("Using critical properties to intially guess pressure")

            # Critical properties: [Tc, Pc, omega, rho_0.7, Zc, Vc, M]
            Pguess = calc.calc_CC_Pguess(yi_list, T_list, CriticalProp)
            if np.isnan(Pguess):
                logger.info("Critical properties were not used to guess an initial pressure")
            else:
                logger.info("Pguess: {}".format(Pguess))
                opts["Pguess"] = Pguess

    # Extract desired method
    if "method" in sys_dict:
        logger.info("Accepted optimization method, {}, for solving pressure".format(sys_dict['method']))
        opts["meth"] = sys_dict['method']

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    # Extract pressure optimization dict
    if "pressure options" in sys_dict:
        logger.info("Accepted options for P optimization")
        opts["pressure_opts"] = sys_dict["pressure options"]

    # Extract pressure optimization dict
    if "mole fraction options" in sys_dict:
        logger.info("Accepted options for mole fraction optimization")
        opts["zi_opts"] = sys_dict["mole fraction options"]

    ## Calculate P and xi
    T_list = np.array(T_list)
    if 'Pguess' in sys_dict:
        inputs = [(T_list[i], yi_list[i], eos, opts, Pguess[i]) for i in range(len(T_list))]
    else:
        inputs = [(T_list[i], yi_list[i], eos, opts) for i in range(len(T_list))]

    if flag_use_mp_object:
        P_list, xi_list, flagv_list, flagl_list, obj_list = mpObj.pool_job(_phase_yiT_wrapper, inputs)
    else:
        P_list, xi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(_phase_yiT_wrapper, inputs)
    #P_list, xi_list, flagv_list, flagl_list, obj_list = batch_jobs( _phase_yiT_wrapper, inputs, ncores=ncores, logger=logger)

    logger.info("--- Calculation phase_yiT Complete ---")

    return {"T":T_list,"xi":xi_list,"P":P_list,"yi":yi_list,"flagl":flagl_list,"flagv":flagv_list, "obj":obj_list}

def _phase_yiT_wrapper(args):

    if len(args) == 4:
        T, yi, eos, opts = args
    elif len(args) == 5:
        T, yi, eos, opts, Pguess = args
        opts["Pguess"] = Pguess

    logger.info("T (K), yi: {} {}, Let's Begin!".format(T, yi))
    try:
        if len(yi[yi!=0.])==1:
            if "rhodict" in opts:
                opt_tmp = {"rhodict": opts["rhodict"]}
            else:
                opt_tmp = {}
            P, _, _ = calc.calc_Psat(T, yi, eos, **opt_tmp)
            xi, flagv, flagl, obj = yi, 0, 1, 0.0
        else:
            P, xi, flagl, flagv, obj = calc.calc_yT_phase(yi, T, eos, **opts)
    except:
        logger.warning("T (K), yi: {} {}, calculation did not produce a valid result.".format(T, yi))
        logger.debug("Calculation Failed:", exc_info=True)
        P, xi, flagl, flagv, obj = [np.nan, np.nan*np.ones(len(xi)), 3, 3, np.nan]

    logger.info("P (Pa), xi: {} {}".format(P, xi))

    return P, xi, flagv, flagl, obj

######################################################################
#                                                                    #
#                Phase Equilibria given yi and T                     #
#                                                                    #
######################################################################
def flash(eos, sys_dict):

    r"""
    Flash calculation of vapor and liquid mole fractions. Only binary systems are currently supported

    Input and system information are assessed first. An output file is generated with T, yi, and corresponding P and xi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")
    else:
        raise ValueError('Tlist is not specified')

    if 'Plist' in sys_dict:
        P_list = np.array(sys_dict['Plist'],float)
        logger.info("Using Plist")
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    if np.size(T_list) != np.size(P_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(P_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    # Extract pressure optimization dict
    if "mole fraction options" in sys_dict:
        logger.info("Accepted options for mole fraction optimization")
        opts = sys_dict["mole fraction options"]
    else:
        opts = {}

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    # Initialize Variables
    l_c = len(eos.nui)
    if l_c != 2:
        raise ValueError("Only binary systems are currently supported for flash calculations, {} were given.".format(l_c))
    l_x = np.array(T_list).shape[0]
    T_list = np.array(T_list)
    P_list = np.array(P_list)

    inputs = [(T_list[i], P_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        xi_list, yi_list, flagv_list, flagl_list, obj_list = mpObj.pool_job(_flash_wrapper, inputs)
    else:
        xi_list, yi_list, flagv_list, flagl_list, obj_list = MultiprocessingJob.serial_job(_flash_wrapper, inputs)

    logger.info("--- Calculation flash Complete ---")

    return {"T":T_list,"xi":xi_list,"P":P_list,"yi":yi_list,"flagl":flagl_list,"flagv":flagv_list, "obj":obj_list}

def _flash_wrapper(args):

    T, P, eos, opts = args

    logger.info("T (K), P (Pa): {} {}, Let's Begin!".format(T, P))
    try:
        xi, flagl, yi, flagv, obj = calc.calc_flash(P, T, eos, **opts)
    except:
        logger.warning("T (K), P (Pa): {} {}, calculation did not produce a valid result.".format(T, P))
        logger.debug("Calculation Failed:", exc_info=True)
        xi, yi, flagl, flagv, obj = [np.nan*np.ones(len(eos.eos_dict["nui"])), np.nan*np.ones(len(eos.eos_dict["nui"])), 3, 3, np.nan]

    logger.info("xi: {}, yi: {}".format(xi, yi))

    return xi, yi, flagv, flagl, obj

######################################################################
#                                                                    #
#                Saturation calc for 1 Component                     #
#                                                                    #
######################################################################
def saturation_properties(eos, sys_dict):

    r"""
    Computes the saturated pressure, liquid, and gas density a one component phase at a temperature.

    Input and system information are assessed first.  An output file is generated with T, :math:`P^{sat}`, :math:`\rho^{sat}_{l}, :math:`\rho^{sat}_{v}
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")
    else:
        raise ValueError('Tlist is not specified')

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'],float)
        logger.info("Using xilist")
    else:
            xi_list = np.array([[1.0] for x in range(len(T_list))])

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        elif len(xi_list) == 1:
            xi_list = [xi_list[0] for i in T_list]
            logger.info("The same mole fraction set, {}, was used for all temperature values".format(xi_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        logger.info("Guess in Psat has been provided, but is unused for this function")

    if 'CriticalProp' in sys_dict:
        logger.info("Critial properties have been provided, but are unused for this function")

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Calculate saturation properties
    T_list = np.array(T_list)

    inputs = [(T_list[i], xi_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        Psat, rholsat, rhovsat = mpObj.pool_job(_saturation_properties_wrapper, inputs)
    else:
        Psat, rholsat, rhovsat = MultiprocessingJob.serial_job(_saturation_properties_wrapper, inputs)

    logger.info("--- Calculation saturation_properties Complete ---")

    return {"T":T_list,"Psat":Psat,"rhol":rholsat,"rhov":rhovsat}

def _saturation_properties_wrapper(args):

    T, xi, eos, opts = args

    logger.info("T (K), xi: {} {}, Let's Begin!".format(T, xi))

    try:
        Psat, rholsat, rhovsat = calc.calc_Psat(T, xi, eos, **opts)
        if np.isnan(Psat):
            logger.warning("T (K), xi: {} {}, calculation did not produce a valid result.".format(T, xi))
            logger.debug("Calculation Failed:", exc_info=True)
            Psat, rholsat, rhovsat = [np.nan, np.nan, np.nan]
        else:
            logger.info("Psat {} Pa, rhol {}, rhov {}".format(Psat,rholsat,rhovsat))
    except:
        logger.warning("T (K), xi: {} {}, calculation did not produce a valid result.".format(T, xi))
        logger.debug("Calculation Failed:", exc_info=True)
        Psat, rholsat, rhovsat = [np.nan, np.nan, np.nan]

    return Psat, rholsat, rhovsat

######################################################################
#                                                                    #
#                Liquid density given xi, T, and P                   #
#                                                                    #
######################################################################
def liquid_properties(eos, sys_dict):

    r"""
    Computes the liquid density and chemical potential given a temperature, pressure, and liquid mole fractions.

    Input and system information are assessed first. An output file is generated with P, T, xi, :math:`\rho_{l}, and :math:`\phi_{l}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'],float)
        logger.info("Using xilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        elif len(xi_list) == 1:
            xi_list = [xi_list[0] for i in T_list]
            logger.info("The same mole fraction set, {}, was used for all temperature values".format(xi_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if "Plist" in sys_dict:
        P_list = np.array(sys_dict['Plist'])
        if np.size(T_list) != np.size(P_list, axis=0):
            if len(P_list)==1:
                P_list = P_list[0] * np.ones_like(T_list)
            else:
                raise ValueError("The number of provided temperatures and pressure sets are different")
        logger.info("Using Plist")
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        logger.info("Guess in pressure has been provided, but is unused for this function")

    if 'CriticalProp' in sys_dict:
        logger.info("Critical properties have been provided, but are unused for this function")

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate liquid density
    l_x = len(T_list)
    T_list = np.array(T_list)

    inputs = [(P_list[i], T_list[i], xi_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        rhol, phil, flagl = mpObj.pool_job(_liquid_properties_wrapper, inputs)
    else:
        rhol, phil, flagl = MultiprocessingJob.serial_job(_liquid_properties_wrapper, inputs)

    logger.info("--- Calculation liquid_properties Complete ---")

    return {"P":P_list,"T":T_list,"xi":xi_list,"rhol":rhol,"phil":phil,"flagl":flagl}

def _liquid_properties_wrapper(args):

    P, T, xi, eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    try:
        rhol, flagl = calc.calc_rhol(P, T, xi, eos, **opts)
        phil = eos.fugacity_coefficient(P, np.array([rhol]), xi, T)
        logger.info("P {} Pa, T {} K, xi {}, rhol {}, phil {}, flagl {}".format(P, T, xi, rhol, phil, flagl))
    except:
        logger.warning('Failed to calculate rhol at {} K and {} Pa'.format(T,P))
        phil = np.nan*np.ones(len(eos.eos_dict["nui"]))

    return rhol, phil, flagl

######################################################################
#                                                                    #
#                Vapor density given yi, T, and P                    #
#                                                                    #
######################################################################
def vapor_properties(eos, sys_dict):

    r"""
    Computes the vapor density and chemical potential given a temperature, pressure, and vapor mole fractions.

    Input and system information are assessed first. An output file is generated with P, T, yi, :math:`\rho_{v}, and :math:`\phi_{v}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")

    if 'yilist' in sys_dict:
        yi_list = np.array(sys_dict['yilist'],float)
        logger.info("Using yilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["yi_list", "T_list"]]):
        raise ValueError('Tlist or yilist are not specified')

    if np.size(T_list) != np.size(yi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(yi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        elif len(yi_list) == 1:
            yi_list = [yi_list[0] for i in T_list]
            logger.info("The same mole fraction set, {}, was used for all temperature values".format(yi_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if "Plist" in sys_dict:
        P_list = np.array(sys_dict['Plist'])
        if np.size(T_list) != np.size(P_list, axis=0):
            if len(P_list)==1:
                P_list = P_list[0] * np.ones_like(T_list)
            else:
                raise ValueError("The number of provided temperatures and pressure sets are different")
        logger.info("Using Plist")
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        logger.info("Guess in pressure has been provided, but is unused for this function")

    if 'CriticalProp' in sys_dict:
        logger.info("Critical properties have been provided, but are unused for this function")

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate vapor density
    T_list = np.array(T_list)

    inputs = [(P_list[i], T_list[i], yi_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        rhov, phiv, flagv = mpObj.pool_job(_vapor_properties_wrapper, inputs)
    else:
        rhov, phiv, flagv = MultiprocessingJob.serial_job(_vapor_properties_wrapper, inputs)

    logger.info("--- Calculation vapor_properties Complete ---")

    return {"P":P_list,"T":T_list,"yi":yi_list,"rhov":rhov,"phiv":phiv,"flagv":flagv}

def _vapor_properties_wrapper(args):

    P, T, yi, eos, opts = args

    logger.info("T (K), P (Pa), yi: {} {} {}, Let's Begin!".format(T, P, yi))

    try:
        rhov, flagv = calc.calc_rhov(P, T, yi, eos, **opts)
        phiv = eos.fugacity_coefficient(P, np.array([rhov]), yi, T)
        logger.info("P {} Pa, T {} K, yi {}, rhov {}, phiv {}, flagv {}".format(P, T, yi, rhov, phiv, flagv))
    except:
        logger.warning('Failed to calculate rhov at {} K and {} Pa'.format(T,P))
        phiv = np.nan*np.ones(len(eos.eos_dict["nui"]))

    return rhov, phiv, flagv

######################################################################
#                                                                    #
#               Solubility Parameter given xi and T                  #
#                                                                    #
######################################################################

def solubility_parameter(eos, sys_dict):

    r"""
    Calculate the Hildebrand solubility parameter based on temperature and composition. This function is based on the method used in Zeng, Z., Y. Xi, and Y. Li "Calculation of Solubility Parameter Using Perturbed-Chain SAFT and Cubic-Plus-Association Equations of State" Ind. Eng. Chem. Res. 2008, 47, 9663–9669.

    Input and system information are assessed first. An output file is generated with T, xi, :math:`\rho_{l}, and :math:`\detla.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")
        del sys_dict['Tlist']

    variables = list(locals().keys())
    if all([key not in variables for key in ["T_list"]]):
        raise ValueError('Tlist are not specified')

    if "Plist" in sys_dict:
        P_list = np.array(sys_dict['Plist'])
        logger.info("Using Plist")
        del sys_dict['Plist']
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    if "xilist" in sys_dict:
        xi_list = np.array(sys_dict['xilist'])
        logger.info("Using xilist")
        del sys_dict['xilist']
    else:
        xi_list = np.array([[1.0] for x in range(len(T_list))])
        logger.info("Single mole fraction of one.")

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        elif len(xi_list) == 1:
            xi_list = np.array([xi_list[0] for x in range(len(T_list))])
            logger.info("The same mole fraction values, {}, were used for all temperature values".format(xi_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if np.size(T_list) != np.size(P_list, axis=0):
        if len(P_list) == 1:
            P_list = np.ones(len(T_list))*P_list[0]
            logger.info("The same pressure, {}, was used for all temperature values".format(P_list[0]))
        else:
            raise ValueError("The number of provided temperatures and pressure sets are different")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Optional values
    opts = {}
    for key, val in sys_dict.items():
        if key in ['dT', 'tol']:
            opts[key] = val
            del sys_dict[key]

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate solubility parameter
    inputs = [(P_list[i], T_list[i], xi_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        rhol, flagl, delta = mpObj.pool_job(_solubility_parameter_wrapper, inputs)
    else:
        rhol, flagl, delta = MultiprocessingJob.serial_job(_solubility_parameter_wrapper, inputs)

    logger.info("--- Calculation solubility_parameter Complete ---")

    return {"P":P_list,"T":T_list,"xi":xi_list,"rhol":rhol,"delta":delta}

def _solubility_parameter_wrapper(args):

    P, T, xi, eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    rhol, flagl = calc.calc_rhol(P, T, xi, eos, **opts)

    if flagl not in [1,2]:
        logger.warning('Failed to calculate rhov at {} K and {} Pa'.format(T,P))
        delta = np.nan
    else:
        delta = calc.hildebrand_solubility(rhol, xi, T, eos, **opts)
        logger.info("P {} Pa, T {} K, xi {}, rhol {}, flagl {}, delta {}".format(P, T, xi, rhol, flagl, delta))

    return rhol, flagl, delta

######################################################################
#                                                                    #
#               Solubility Parameter given xi and T                  #
#                                                                    #
######################################################################

def verify_eos(eos, sys_dict):

    r"""
    The following consistency checks are performed to ensure the calculated fugacity coefficients are thermodynamically consistent.

    - 1. d(log phi) / dP = (Z - 1)/P
    - 
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object (e.g. options for density array :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    ## Extract and check input data

    ncomp = len(eos.nui)
    if "xilist" in sys_dict:
        xi_list = np.array(sys_dict['xilist'])
        logger.info("Using xilist")
        del sys_dict['xilist']
    elif ncomp == 2:
        tmp = np.linspace(0,1,11)
        xi_list = np.array([[x, 1.0-x] for x in tmp])
        logger.info("Use array of mole fractions")
    else:
        raise ValueError("With more that 2 components, the mole fractions need to be specified")

    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'],float)
        logger.info("Using Tlist")
        del sys_dict['Tlist']
    else:
        T_list = 298.15*np.array(len(xi_list))
        logger.info("Assume 298.15 K")

    if "Plist" in sys_dict:
        P_list = np.array(sys_dict['Plist'])
        logger.info("Using Plist")
        del sys_dict['Plist']
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        elif len(xi_list) == 1:
            xi_list = np.array([xi_list[0] for x in range(len(T_list))])
            logger.info("The same mole fraction values, {}, were used for all temperature values".format(xi_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")

    if np.size(T_list) != np.size(P_list, axis=0):
        if len(P_list) == 1:
            P_list = np.ones(len(T_list))*P_list[0]
            logger.info("The same pressure, {}, was used for all temperature values".format(P_list[0]))
        else:
            raise ValueError("The number of provided temperatures and pressure sets are different")

    if 'mpObj' in sys_dict:
        mpObj = sys_dict['mpObj']
        flag_use_mp_object = True
    else:
        flag_use_mp_object = False

    ## Optional values
    opts = {}

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate solubility parameter
    inputs = [(P_list[i], T_list[i], xi_list[i], eos, opts) for i in range(len(T_list))]
    if flag_use_mp_object:
        residual_v1, residual_v2, flagv, log_phiv, residual_l1, residual_l2, flagl, log_phil = mpObj.pool_job(_verify_eos_wrapper, inputs)
    else:
        residual_v1, residual_v2, flagv, log_phiv, residual_l1, residual_l2, flagl, log_phil = MultiprocessingJob.serial_job(_verify_eos_wrapper, inputs)

    logger.info("--- Calculation verify_eos Complete ---")

    return {"P":P_list, "T":T_list, "xi":xi_list, "residual_v1":residual_v1, "residual_v2":residual_v2, "flagv": flagv, "log_phivi":log_phiv, "residual_l1":residual_l1, "residual_l2":residual_l2, "flagl": flagl, "log_phili":log_phil}

def _verify_eos_wrapper(args):

    P, T, xi, eos, opts = args

    logger.info("T (K), P (Pa), xi: {} {} {}, Let's Begin!".format(T, P, xi))

    rhov, flagv = calc.calc_rhov(P, T, xi, eos, **opts)
    if np.isnan(rhov):
        logger.warning('Failed to calculate rhov at {} K and {} Pa'.format(T,P))
        log_phiv, residual_v1, residual_v2 = np.nan, np.nan, np.nan
    else:
        phiv = eos.fugacity_coefficient(P, np.array([rhov]), xi, T)
        log_phiv = np.log(phiv)
        residual_v1 = calc.fugacity_test_1(P, T, xi, rhov, eos, **opts)
        residual_v2 = calc.fugacity_test_2(P, T, xi, rhov, eos, **opts)
        logger.info("rhov {}, flagv {}, log_phiv {}, log_phiv {}, residual1 {}, residual2 {}".format(rhov, flagv, np.sum(xi*log_phiv), log_phiv, residual_v1, residual_v2))

    rhol, flagl = calc.calc_rhol(P, T, xi, eos, **opts)
    if np.isnan(rhol):
        logger.warning('Failed to calculate rhol at {} K and {} Pa'.format(T,P))
        log_phil, residual_l1, residual_l2 = np.nan, np.nan, np.nan
    else:
        phil = eos.fugacity_coefficient(P, np.array([rhol]), xi, T)
        log_phil = np.log(phil)
        residual_l1 = calc.fugacity_test_1(P, T, xi, rhol, eos, **opts)
        residual_l2 = calc.fugacity_test_2(P, T, xi, rhol, eos, **opts)
        logger.info("rhol {}, flagl {}, log_phil {}, log_phil {}, residual1 {}, residual2 {}".format(rhol, flagl, np.sum(xi*log_phil), log_phil, residual_l1, residual_l2))

    return residual_v1, residual_v2, flagv, log_phiv, residual_l1, residual_l2, flagl, log_phil

