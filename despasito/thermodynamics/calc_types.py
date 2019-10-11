"""
    This thermo module contains a series of wrappers to handle the inputs and outputs of these functions. The calc module contains the thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS.
    
    None of the functions in this folder need to be handled directly, as a function factory is included in our __init__.py file. Add "from thermodynamics import thermo" and use "thermo("calc_type",eos,input_dict)" to get started.
    
"""

import logging
import numpy as np
import os
import sys
import logging

from . import calc

"""
.. todo::
    phase_xiT: add like to rhodict options 

"""

######################################################################
#                                                                    #
#                Phase Equilibrium given xi and T                     #
#                                                                    #
######################################################################
def phase_xiT(eos, sys_dict):

    r"""
    Assess input and system information and calculate phase diagram given liquid mole fractions, xi, and temperature.

    An output file is generated with T, xi, and corresponding P and yi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values    
    """

    logger = logging.getLogger(__name__)

    #computes P and yi from xi and T

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'])
        logger.info("Using Tlist") 

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'])
        logger.info("Using xilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')
        logger.error('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")
            logger.error("The number of provided temperatures and mole fraction sets are different")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        Pguess = sys_dict['Pguess']
        if np.size(T_list) != np.size(Pguess):
            if type(Pguess) not in [list, numpy.ndarray]:
                opts["Pguess"] = np.ones(len(T_list))*Pguess
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            elif len(T_list) == 1:
                opts["Pguess"] = np.ones(len(T_list))*Pguess[0]
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            else:
                raise ValueError("The number of provided pressure and mole fraction sets are different")
                logger.error("The number of provided pressure and mole fraction sets are different")
        logger.info("Using user defined inital guess has been provided")
    else:
        if 'CriticalProp' in sys_dict:
            CriticalProp = np.array(sys_dict['CriticalProp'])
            logger.info("Using critical properties to intially guess pressure")

            # Critical properties: [Tc, Pc, omega, rho_0.7, Zc, Vc, M]
            Pguess = calc.calc_CC_Pguess(xi_list, T_list, CriticalProp)
            if np.isnan(Pguess):
                logger.info("Critical properties were not used to guess an intial pressure")
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
    T_list = np.array(T_list)
    P_list = np.zeros_like(T_list)
    flagv_list = np.zeros_like(T_list)
    flagl_list = np.zeros_like(T_list)
    yi_list = np.zeros_like(xi_list)
    obj_list = np.zeros(len(T_list))
    for i in range(np.size(T_list)):
        optsi = opts
        if "Pguess" in opts:
            optsi["Pguess"] = optsi["Pguess"][i]

        logger.info("T (K), xi: {} {}, Let's Begin!".format(str(T_list[i]), str(xi_list[i])))
        try:
            P_list[i], yi_list[i], flagv_list[i], flagl_list[i], obj_list[i] = calc.calc_xT_phase(xi_list[i], T_list[i], eos, **optsi)
        except:
            logger.warning("T (K), xi: {} {}, calculation did not produce a valid result.".format(T_list[i], xi_list[i]))
            logger.debug("Calculation Failed:", exc_info=True)
            P_list[i], yi_list[i] = [np.nan, np.nan]
            flagl_list[i], flagv_list[i], obj_list[i] = [3, 3, np.nan]
            continue
        logger.info("P (Pa), yi: {} {}".format(P_list[i], yi_list[i]))

    logger.info("--- Calculation phase_xiT Complete ---")

    return {"T":T_list,"xi":xi_list,"P":P_list,"yi":yi_list,"flagl":flagl_list,"flagv":flagv_list,"obj":obj_list}


######################################################################
#                                                                    #
#                Phase Equilibria given yi and T                     #
#                                                                    #
######################################################################
def phase_yiT(eos, sys_dict):

    r"""
    Assess input and system information and calculate phase diagram given vapor mole fractions, yi, and temperature.

    An output file is generated with T, yi, and corresponding P and xi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    logger = logging.getLogger(__name__)

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'])
        logger.info("Using Tlist")

    if 'yilist' in sys_dict:
        yi_list = np.array(sys_dict['yilist'])
        logger.info("Using yilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["yi_list", "T_list"]]):
        raise ValueError('Tlist or yilist are not specified')
        logger.error('Tlist or yilist are not specified')

    if np.size(T_list) != np.size(yi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(yi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")
            logger.error("The number of provided temperatures and mole fraction sets are different")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        Pguess = sys_dict['Pguess']
        if np.size(T_list) != np.size(Pguess):
            if type(Pguess) not in [list, numpy.ndarray]:
                opts["Pguess"] = np.ones(len(T_list))*Pguess
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            elif len(T_list) == 1:
                opts["Pguess"] = np.ones(len(T_list))*Pguess[0]
                logger.info("The same pressure, {}, was used for all mole fraction values".format(Pguess))
            else:
                raise ValueError("The number of provided pressure and mole fraction sets are different")
                logger.error("The number of provided pressure and mole fraction sets are different")
        logger.info("Using user defined inital guess has been provided")
    else:
        if 'CriticalProp' in sys_dict:
            CriticalProp = np.array(sys_dict['CriticalProp'])
            logger.info("Using critical properties to intially guess pressure")

            # Critical properties: [Tc, Pc, omega, rho_0.7, Zc, Vc, M]
            Pguess = calc.calc_CC_Pguess(yi_list, T_list, CriticalProp)
            if np.isnan(Pguess):
                logger.info("Critical properties were not used to guess an intial pressure")
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
    P_list = np.zeros_like(T_list)
    flagv_list = np.zeros(len(T_list))
    flagl_list = np.zeros(len(T_list))
    xi_list = np.zeros(len(yi_list))
    obj_list = np.zeros(len(T_list))
    for i in range(np.size(T_list)):
        optsi = opts
        if "Pguess" in opts:
            optsi["Pguess"] = optsi["Pguess"][i]
        logger.info("T (K), yi: {} {}, Let's Begin!".format(str(T_list[i]), str(yi_list[i])))
        try:
            P_list[i], xi_list[i], flagl_list[i], flagv_list[i], obj_list[i]  = calc.calc_yT_phase(yi_list[i], T_list[i], eos, **optsi)
        except:
            logger.warning("T (K), yi: {} {}, calculation did not produce a valid result.".format(str(T_list[i]), str(yi_list[i])))
            logger.debug("Calculation Failed:", exc_info=True)
            P_list[i], xi_list[i] = [np.nan, np.nan]
            flagl_list[i], flagv_list[i], obj_list[i] = [3, 3, np.nan]
            continue
        logger.info("P (Pa), xi: {} {}".format(str(P_list[i]), str(xi_list[i])))

    logger.info("--- Calculation phase_yiT Complete ---")

    return {"T":T_list,"yi":yi_list,"P":P_list,"xi":xi_list,"flagl":flagl_list,"flagv":flagv_list, "obj":obj_list}

######################################################################
#                                                                    #
#                Saturation calc for 1 Component               #
#                                                                    #
######################################################################
def sat_props(eos, sys_dict):

    r"""
    Assess input and system information and computes the saturated pressure, liquid, and gas density a one component phase at a temperature.

    An output file is generated with T, :math:`P^{sat}`, :math:`\rho^{sat}_{l}, :math:`\rho^{sat}_{v}
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    logger = logging.getLogger(__name__)

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'])
        logger.info("Using Tlist")

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'])
        logger.info("Using xilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')
        logger.error('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")
            logger.error("The number of provided temperatures and mole fraction sets are different")

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

    ## Calculate saturation properties
    T_list = np.array(T_list)
    Psat = np.zeros(len(T_list))
    rholsat = np.zeros(len(T_list))
    rhovsat = np.zeros(len(T_list))

    for i in range(np.size(T_list)):

        logger.info("T (K), xi: {} {}, Let's Begin!".format(str(T_list[i]), str(xi_list[i])))
        try:
            Psat[i], rholsat[i], rhovsat[i] = calc.calc_Psat(T_list[i], xi_list[i], eos, **opts)
        except:
            logger.warning("T (K), xi: {} {}, calculation did not produce a valid result.".format(str(T_list[i]), str(xi_list[i])))
            logger.debug("Calculation Failed:", exc_info=True)
            Psat[i], rholsat[i], rhovsat[i] = [np.nan, np.nan, np.nan]
            continue
        logger.info("Psat {} Pa, rhol {}, rhov {}".format(Psat[i],rholsat[i],rhovsat[i]))

    logger.info("--- Calculation sat_props Complete ---")

    return {"T":T_list,"Psat":Psat,"rhol":rholsat,"rhov":rhovsat}


######################################################################
#                                                                    #
#                Liquid density given xi, T, and P                   #
#                                                                    #
######################################################################
def liquid_properties(eos, sys_dict):

    r"""
    Assess input and system information and computes the liquid density and chemical potential given a temperature, pressure, and liquid mole fractions.

    An output file is generated with P, T, xi, :math:`\rho_{l}, and :math:`\phi_{l}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    logger = logging.getLogger(__name__)

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'])
        logger.info("Using Tlist")

    if 'xilist' in sys_dict:
        xi_list = np.array(sys_dict['xilist'])
        logger.info("Using xilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["xi_list", "T_list"]]):
        raise ValueError('Tlist or xilist are not specified')
        logger.error('Tlist or xilist are not specified')

    if np.size(T_list) != np.size(xi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(xi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".format(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")
            logger.error("The number of provided temperatures and mole fraction sets are different")

    if "Plist" not in sys_dict:
        logger.info("Using Plist")
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        logger.info("Guess in pressure has been provided, but is unused for this function")

    if 'CriticalProp' in sys_dict:
        logger.info("Critial properties have been provided, but are unused for this function")

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate liquid density
    T_list = np.array(T_list)
    rhol = np.zeros(len(T_list))
    phil = []
    for i in range(np.size(T_list)):
        rhol[i], flagl = calc.calc_rhol(P_list[i], T_list[i], xi_list[i], eos, **opts)

        if np.isnan(rhol[i]):
            logger.warning('Failed to calculate rhol at {}'.format(T_list[i]))
            phil[i] = np.nan
        else:
            logger.info("P (Pa), T (K), xi, rhol: {} {} {} {}".format(P_list[i],T_list[i],xi_list[i],rhol[i]))
            muil_tmp = eos.chemicalpotential(P_list[i], np.array([rhol[i]]), xi_list[i], T_list[i])
            phil.append(np.exp(muil_tmp))

    logger.info("--- Calculation liquid_density Complete ---")

    return {"P":P_list,"T":T_list,"xi":xi_list,"rhol":rhol,"phil":phil}

######################################################################
#                                                                    #
#                Vapor density given yi, T, and P                    #
#                                                                    #
######################################################################
def vapor_properties(eos, sys_dict):

    r"""
    Assess input and system information and computes the vapor density and chemical potential given a temperature, pressure, and vapor mole fractions.

    An output file is generated with P, T, yi, :math:`\rho_{v}, and :math:`\phi_{v}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.

    Returns
    -------
    output_dict : dict
        Output of dictionary containing given and calculated values
    """

    logger = logging.getLogger(__name__)

    ## Extract and check input data
    if 'Tlist' in sys_dict:
        T_list = np.array(sys_dict['Tlist'])
        logger.info("Using Tlist")

    if 'yilist' in sys_dict:
        yi_list = np.array(sys_dict['yilist'])
        logger.info("Using yilist")

    variables = list(locals().keys())
    if all([key not in variables for key in ["yi_list", "T_list"]]):
        raise ValueError('Tlist or yilist are not specified')
        logger.error('Tlist or yilist are not specified')

    if np.size(T_list) != np.size(yi_list, axis=0):
        if len(T_list) == 1:
            T_list = np.ones(len(yi_list))*T_list[0]
            logger.info("The same temperature, {}, was used for all mole fraction values".join(T_list[0]))
        else:
            raise ValueError("The number of provided temperatures and mole fraction sets are different")
            logger.error("The number of provided temperatures and mole fraction sets are different")

    if "Plist" not in sys_dict:
        logger.info("Using Plist")
    else:
        P_list = 101325.0 * np.ones_like(T_list)
        logger.info("Assuming atmospheric pressure.")

    ## Optional values
    opts = {}

    # Process initial guess in pressure
    if 'Pguess' in sys_dict:
        logger.info("Guess in pressure has been provided, but is unused for this function")

    if 'CriticalProp' in sys_dict:
        logger.info("Critial properties have been provided, but are unused for this function")

    # Extract rho dict
    if "rhodict" in sys_dict:
        logger.info("Accepted options for P vs. density curve")
        opts["rhodict"] = sys_dict["rhodict"]

    ## Calculate vapor density
    T_list = np.array(T_list)
    rhov = np.zeros(len(T_list))
    phiv = []
    for i in range(np.size(T_list)):
        rhov[i], flagv = calc.calc_rhov(P_list[i], T_list[i], yi_list[i], eos, **opts)
        if np.isnan(rhov[i]):
            logger.warning('Failed to calculate rhov at {}'.format(T_list[i]))
            phiv[i] = np.nan
        else:
            logger.info("P (Pa), T (K), yi, rhov: {} {} {} {}".format(P_list[i],T_list[i],yi_list[i],rhov[i]))
            muiv_tmp = eos.chemicalpotential(P_list[i], np.array([rhov[i]]), yi_list[i], T_list[i])
            phiv.append(np.exp(muiv_tmp))

    logger.info("--- Calculation vapor_density Complete ---")

    return {"P":P_list,"T":T_list,"yi":yi_list,"rhov":rhov,"phiv":phiv}

