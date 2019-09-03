"""
    This thermo module contains a series of wrappers to handle the inputs and outputs of these functions. The calc module contains the thermodynamic calculations. Calculation of pressure, chemical potential, and max density are handled by an eos object so that these functions can be used with any EOS.
    
    None of the functions in this folder need to be handled directly, as a function factory is included in our __init__.py file. Add "from thermodynamics import thermo" and use "thermo("calc_type",eos,input_dict)" to get started.
    
"""

import numpy as np
import os
import sys

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
def phase_xiT(eos, sys_dict, rhodict={}, output_file="phase_xiT_output.txt"):

    r"""
    Assess input and system information and calculate phase diagram given liquid mole fractions, xi, and temperature.

    An output file is generated with T, xi, and corresponding P and yi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole fraction curves.
    output_file : str, Optional, default: "phase_xiT_output.txt"
        Name of file in which the temperature, pressure, and vapor/liquid mole fractions are saved.
    """

    #computes P and yi from xi and T

    ## Extract and check input data
    try:
        T_list = np.array(sys_dict['Tlist'])
        xi_list = np.array(sys_dict['xilist'])
        print("Using xilist and Tlist")
    except:
        raise Exception('Tlist or xilist are not specified')
    assert np.size(T_list) == np.size(xi_list, axis=0), "The number of Temperatures and xi are differnt"

    try:
        Pguess = float(input_dict['Pguess'])
        flag_gss = 1
        print("Using user defined inital guess for P1")
    except:
        flag_gss = 0
        try:
            CriticalProp = np.array(input_dict['CriticalProp'])
            flag_gss = 2
            print("Using intial guess for P using critical properties")
        except:
            flag_gss = 0
            print("Critical properties aren't specified for an initial guess")

    if flag_gss == 2:
        # Critical properties: [Tc, Pc, omega, rho_0.7, Zc, Vc, M]
        Pguess = calc.calc_CC_Pguess(xi_list, T_list, CriticalProp)
        if type(Pguess) == float:
            if np.isnan(Pguess):
                flag_gss = 0
        else:
            print("Pguess: ", Pguess)
        #if all(-0.847 < x < 0.2387 for x in CriticalProp[2]):
        #    Pguess = calc.calc_CC_Pguess(xi_list,T_list,CriticalProp)
        #    print "Pguess: ", Pguess
        #else:
        #    flag_gss = 0
        #    print "Omega values are out of range for the specified correlation"

    ## Generate Output
    with open(output_file, 'w') as f:
        f.write('Temperature (K), xi, Pressure (Pa), yi\n')

    ## Calculate P and yi
    P_list = np.zeros_like(T_list)
    yi_list = np.zeros_like(xi_list)
    for i in range(np.size(T_list)):
        print("T (K), xi", T_list[i], xi_list[i], "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        if flag_gss == 0:
            P_list[i], yi_list[i] = calc.calc_xT_phase(xi_list[i], T_list[i], eos, rhodict=rhodict)
            Pguess = np.zeros(len(T_list))
        else:
            P_list[i], yi_list[i] = calc.calc_xT_phase(xi_list[i], T_list[i], eos, rhodict=rhodict, Pguess=Pguess)
            Pguess = P_list[i]
        print("P (Pa), yi", P_list[i], yi_list[i][0], yi_list[i][1])

        ## Generate Output
        with open(output_file, 'a') as f:
            tmp = [T_list[i], xi_list[i], P_list[i], yi_list[i]]
            f.write(", ".join([str(x) for x in tmp]) + '\n')

    print("--- Calculation phase_xiT Complete ---")


######################################################################
#                                                                    #
#                Phase Equilibria given yi and T                     #
#                                                                    #
######################################################################
def phase_yiT(eos, sys_dict, rhodict={}, output_file="phase_yiT_output.txt"):

    r"""
    Assess input and system information and calculate phase diagram given vapor mole fractions, yi, and temperature.

    An output file is generated with T, yi, and corresponding P and xi.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole fraction curves.
    output_file : str, Optional, default: "phase_yiT_output.txt"
        Name of file in which the temperature, pressure, and vapor/liquid mole fractions are saved.
    """

    ## Extract and check input data
    try:
        T_list = np.array(sys_dict['Tlist'])
        yi_list = np.array(sys_dict['yilist'])
        print("Using yilist and Tlist")
    except:
        raise Exception('Tlist or yilist are not specified')
    assert np.size(T_list) == np.size(yi_list, axis=0), "The number of Temperatures and yi are differnt"

    ## Generate Output
    with open(output_file, 'w') as f:
        f.write('Temperature (K), yi, Pressure (Pa), xi\n')

    ## Calculate P and xi
    P_list = np.zeros_like(T_list)
    xi_list = np.zeros_like(yi_list)
    for i in range(np.size(T_list)):
        print("T (K), yi", T_list[i], yi_list[i], "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        P_list[i], xi_list[i] = calc.calc_yT_phase(yi_list[i], T_list[i], eos, rhodict=rhodict)
        print("Final P, xi:",P_list[i], xi_list[i])

        ## Generate Output
        with open(output_file, 'a') as f:
            tmp = [T_list[i], yi_list[i], P_list[i], xi_list[i]]
            f.write(", ".join([str(x) for x in tmp]) + '\n')

    print("--- Calculation phase_yiT Complete ---")


######################################################################
#                                                                    #
#                Saturation calc for 1 Component               #
#                                                                    #
######################################################################
def sat_props(eos, sys_dict, rhodict={}, output_file="saturation_output.txt"):

    r"""
    Assess input and system information and computes the saturated pressure, liquid, and gas density a one component phase at a temperature.

    An output file is generated with T, :math:`P^{sat}`, :math:`\rho^{sat}_{l}, :math:`\rho^{sat}_{v}
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole fraction curves.
    output_file : str, Optional, default: "saturation_output.txt"
        Name of file in which the temperature, saturation pressure, and vapor/liquid densities.
    """

    ## Exctract and check input data
    try:
        T_list = np.array(sys_dict['Tlist'])
        print("Using Tlist")
    except:
        raise Exception('Temperature must be specified')

    ## Generate Output
    with open(output_file, 'w') as f:
        f.write('Temperature (K), Saturated Pressure (Pa), liquid denisty (mol/m^3), vapor density (mol/m^3)\n')

    ## Calculate saturation properties
    Psat = np.zeros_like(T_list)
    rholsat = np.zeros_like(T_list)
    rhovsat = np.zeros_like(T_list)

    for i in range(np.size(T_list)):
        try:
            Psat[i], rholsat[i], rhovsat[i] = calc.calc_Psat(T_list[i], np.array([1.0]), eos, rhodict=rhodict)
        except:
            Psat[i] = np.nan
            rholsat[i] = np.nan
            rhovsat[i] = np.nan
            print('Failed to calculate Psat, rholsat, rhovsat at', T_list[i])

        ## Generate Output
        with open(output_file, 'a') as f:
            tmp = [T_list[i], Psat[i], rholsat[i], rhovsat[i]]
            f.write(", ".join([str(x) for x in tmp]) + '\n')

    print("--- Calculation sat_props Complete ---")


######################################################################
#                                                                    #
#                Liquid density given xi, T, and P                   #
#                                                                    #
######################################################################
def liquid_properties(eos, sys_dict, rhodict={}, output_file="liquid_properties_output.txt"):

    r"""
    Assess input and system information and computes the liquid density and chemical potential given a temperature, pressure, and liquid mole fractions.

    An output file is generated with P, T, xi, :math:`\rho_{l}, and :math:`\phi_{l}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole fraction curves.
    output_file : str, Optional, default: "liquid_properties_output.txt"
        Name of file in which the pressure, temperature, and liquid mole fraction, density, and chemical potential.
    """

    ## Extract and check input data
    try:
        T_list = np.array(sys_dict['Tlist'])
        xi_list = np.array(sys_dict['xilist'])
        try:
            P_list = np.array(sys_dict['Plist'])
            print("Using xilist, Tlist, and Plist")
        except:
            P_list = 101325.0 * np.ones_like(T_list)
            print("Assuming atmospheric pressure.")
            print("Using xilist and Tlist")
    except:
        raise Exception('Tlist or xilist are not specified')
    assert np.size(T_list) == np.size(xi_list, axis=0), "The number of Temperatures and xi are differnt"

    ## Generate Output
    with open(output_file, 'w') as f:
        f.write('Pressure (Pa), Temperature (K), xi, liquid denisty (mol/m^3), fugacity coeff.\n')

    ## Calculate liquid density
    rhol = np.zeros_like(T_list)
    phil = []
    for i in range(np.size(T_list)):
        rhol[i], flagl = calc.calc_rhol(P_list[i], T_list[i], xi_list[i], eos, rhodict=rhodict)

        if np.isnan(rhol[i]):
            print('Failed to calculate rhol at', T_list[i])
            phil[i] = np.nan
        else:
            print(T_list[i], rhol[i])
            muil_tmp = eos.chemicalpotential(P_list[i], np.array([rhol[i]]), xi_list[i], T_list[i])
            phil.append(np.exp(muil_tmp))

        ## Generate Output
        with open(output_file, 'a') as f:
            tmp = [P_list[i], T_list[i], xi_list[i], rhol[i], phil[i]]
            f.write(", ".join([str(x) for x in tmp]) + '\n')

    print("--- Calculation liquid_density Complete ---")


######################################################################
#                                                                    #
#                Vapor density given yi, T, and P                    #
#                                                                    #
######################################################################
def vapor_properties(eos, sys_dict, rhodict={}, output_file="vapor_properties_output.txt"):

    r"""
    Assess input and system information and computes the vapor density and chemical potential given a temperature, pressure, and vapor mole fractions.

    An output file is generated with P, T, yi, :math:`\rho_{v}, and :math:`\phi_{v}.
    
    Parameters
    ----------
    eos : obj
        An instance of the defined EOS class to be used in thermodynamic computations.
    sys_dict: dict
        A dictionary of all information given in the input .json file that wasn't used to create the EOS object.
    rhodict : dict, Optional, default: {}
        Dictionary of options used in calculating pressure vs. mole fraction curves.
    output_file : str, Optional, default: "vapor_properties_output.txt"
        Name of file in which the pressure, temperature, and vapor mole fraction, density, and chemical potential.
    """

    ## Extract and check input data
    try:
        T_list = np.array(sys_dict['Tlist'])
        yi_list = np.array(sys_dict['yilist'])
        try:
            P_list = np.array(sys_dict['Plist'])
            print("Using yilist, Tlist, and Plist")
        except:
            P_list = 101325.0 * np.ones_like(T_list)
            print("Assuming atmospheric pressure.")
            print("Using yilist and Tlist")
    except:
        raise Exception('Tlist or yilist are not specified')
    assert np.size(T_list) == np.size(yi_list, axis=0), "The number of Temperatures and xi are differnt"

    ## Generate Output
    with open(output_file, 'w') as f:
        f.write('Pressure (Pa), Temperature (K), yi, vapor denisty (mol/m^3), fugacity coeff.\n')

    ## Calculate vapor density
    rhov = np.zeros_like(T_list)
    phiv = []
    for i in range(np.size(T_list)):
        rhov[i], flagv = calc.calc_rhov(P_list[i], T_list[i], yi_list[i], eos, rhodict=rhodict)
        if np.isnan(rhov[i]):
            print('Failed to calculate rhov at', T_list[i])
            phiv[i] = np.nan
        else:
            print(T_list[i], rhov[i])
            muiv_tmp = eos.chemicalpotential(P_list[i], np.array([rhov[i]]), yi_list[i], T_list[i])
            phiv.append(np.exp(muiv_tmp))

        ## Generate Output
        with open(output_file, 'a') as f:
            tmp = [P_list[i], T_list[i], yi_list[i], rhov[i],phiv[i]]
            f.write(", ".join([str(x) for x in tmp]) + '\n')

    print("--- Calculation vapor_density Complete ---")

