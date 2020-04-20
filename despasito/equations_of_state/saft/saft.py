# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""
import sys
import numpy as np
import logging

from despasito.equations_of_state import constants
import despasito.utils.general_toolbox as gtb
import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state.interface import EOStemplate

logger = logging.getLogger(__name__)

def saft_type(name):
    
    if name == "gamma_mie":
        from despasito.equations_of_state.saft.helmholtz.Aideal import Aideal
        from despasito.equations_of_state.saft.helmholtz.Amonomer_mie import Amonomer
        from despasito.equations_of_state.saft.helmholtz.Achain_mie import Achain
        from despasito.equations_of_state.saft.helmholtz.Aassoc import Aassoc
    
    return Aideal, Amonomer, Achain, Aassoc

class saft(EOStemplate):

    r"""
    Initialize EOS object for SAFT-:math:`\gamma`-Mie.
    
    All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        :math:`\\nu_{i,k}/k_B`. Array of number of components by number of bead types. Defines the number of each type of group in each component.
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
        - mass: Bead mass [kg/mol]
        - l_r: :math:`\lambda^{r}_{k,k}`, Exponent of repulsive term between groups of type k
        - l_a: :math:`\lambda^{a}_{k,k}`, Exponent of attractive term between groups of type k\
        - Vks: :math:`V_{k,s}`, Number of groups, k, in component
        - Sk: Optional, :math:`S_{k}`, Shape parameter of group k
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - epsilon: :math:`\epsilon_{k,l}/k_B`, Energy parameter scaled by Boltzmann Constant
        - l_r: :math:`\lambda^{r}_{k,l}`, Exponent of repulsive term between groups of type k and l
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.

    sitenames : list[str], Optional, default: ["H", "e1", "e2"]
        List of unique association sites used among components
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):

        needed_attributes = ['nui','beads','beadlibrary']

        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                setattr(self, key, kwargs[key])

        Aideal, Amonomer, Achain, Aassoc = saft_type(kwargs["saft_name"])
        
        self.Aideal = Aideal(kwargs)
        self.Amonomer = Amonomer(kwargs)
        self.Achain = Achain(kwargs)
        self.Aassoc = Aassoc(kwargs)

    def residual_helmholtz_energy(self, rho, T, xi):
        r"""
        Return a vector of residual Helmholtz energy.
        
        :math:`\frac{A^{res}}{N k_{B} T}`
        
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
        Ares : numpy.ndarray
            Residual helmholtz energy for each density value given.
        """
    
        if len(xi) != len(self.Amonomer.eos_dict['nui']):
            raise ValueError("Number of components in mole fraction list doesn't match components in nui. Check bead_config.")
    
        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        if any(np.array(xi) < 0.):
            raise ValueError("Mole fractions cannot be less than zero.")

        Ares = self.Amonomer.Amonomer(rho, T, xi) + self.Achain.Achain(rho, T, xi)

        if self.Aassoc.flag_assoc:
            Ares += self.Aassoc.Aassoc(rho, T, xi)

        return Ares

    def helmholtz_energy(self, rho, T, xi):
        r"""
        Return a vector of Helmholtz energy.
        
        :math:`\frac{A}{N k_{B} T}`
        
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
        A : numpy.ndarray
            Total helmholtz energy for each density value given.
        """

#        np.savetxt("new_method.csv",np.transpose(np.array([rho, rho*constants.Nav])),delimiter=",")
#        sys.exit("stop")

        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        A = self.residual_helmholtz_energy(rho, T, xi) + self.Aideal.Aideal(rho, T, xi)

      #  tmp = np.transpose(np.array([rho*constants.Nav, self.Aideal.Aideal(rho, T, xi), self.Amonomer.Ahard_sphere(rho, T, xi), self.Amonomer.Afirst_order(rho, T, xi), self.Amonomer.Asecond_order(rho, T, xi), self.Amonomer.Athird_order(rho, T, xi), self.Achain.Achain(rho, T, xi), self.Aassoc.Aassoc(rho, T, xi)]))
      #  np.savetxt("new_method.csv",tmp,delimiter=",")
      #  sys.exit("stop")

        return A

    def pressure(self, rho, T, xi, step_factor=1E+4):
        """
        Compute pressure given system information.
       
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : list[float]
            Mole fraction of each component
       
        Returns
        -------
        P : numpy.ndarray
            Array of pressure values [Pa] associated with each density and so equal in length
        """

        P_tmp = gtb.central_difference(rho, self.helmholtz_energy, args=(T, xi), step_factor=step_factor)
        pressure = P_tmp*T*(constants.kb*constants.Nav)

        return pressure

    def fugacity_coefficient(self, P, rho, xi, T, dy=1e-2):

        """
        Compute fugacity coefficient.
      
        Parameters
        ----------
        P : float
            Pressure of the system [Pa]
        rho : float
            Molar density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : list[float]
            Mole fraction of each component
    
        Returns
        -------
        mui : numpy.ndarray
            Array of chemical potential values for each component
        """

        logZ = np.log(P / (rho * T * (constants.Nav * constants.kb)))
        Ares = self.residual_helmholtz_energy(rho, T, xi)
        dAresdrho = tb.partial_density_central_difference(xi, rho, T, self.residual_helmholtz_energy, step_size=dy)
        phi = np.exp(Ares + rho*dAresdrho - logZ)

        return phi

    def density_max(self, xi, T, maxpack=0.65):

        """
        Estimate the maximum density based on the hard sphere packing fraction.
        
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]
        maxpack : float, Optional, default: 0.65
            Maximum packing fraction
        
        Returns
        -------
        maxrho : float
            Maximum molar density [mol/m^3]
        """

        maxrho = self.Amonomer.density_max(xi, T, maxpack=maxpack)


        return maxrho

    def param_guess(self, param_name, bead_names):
        """
        Generate initial guesses for the parameters to be fit.

        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS mentation for supported parameter names.
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.

        Returns
        -------
        param_initial_guess : numpy.ndarray, 
            An initial guess for parameter, it will be optimized throughout the process.
        """

        param_types = ["epsilon", "sigma", "l_r", "l_a", "Sk", "K"]

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))

        # Non bonded parameters
        if (param_name in ["epsilon", "sigma", "l_r", "l_a", "Sk"]):
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.eos_dict['beadlibrary']:
                    param_value = self.eos_dict['beadlibrary'][bead_names[0]][param_name]
                else:
                    param_value = self.check_bounds(param_name, bead_names, np.empty(2))[1]/2
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self._crosslibrary and bead_names[0] in self._crosslibrary[bead_names[1]]:
                    param_value = self._crosslibrary[bead_names[1]][bead_names[0]][param_name]
                elif bead_names[0] in self._crosslibrary and bead_names[1] in self._crosslibrary[bead_names[0]]:
                    param_value = self._crosslibrary[bead_names[0]][bead_names[1]][param_name]
                else:
                    param_value = self.check_bounds(bead_names[0], param_name, np.empty(2))[1]/2

        # Association Sites
        elif any([param_name.startswith('epsilon'), param_name.startswith('K')]):
            tmp = [param_name.startswith('epsilon'), param_name.startswith('K')]
            # Ensure sitenames are valid and on list
            if tmp[0] == True:
                tmp_name_full = param_name.replace("epsilon","")
            elif tmp[1] == True:
                tmp_name_full = param_name.replace("K","")
                         
            flag = 0

            for site1 in self._sitenames:
                if tmp_name_full.startswith(site1):
                    tmp_name = tmp_name_full.replace(site1,"")
                    for site2 in self._sitenames:
                        if tmp_name == site2:
                            flag = 1
                            break
                    if flag == 1:
                        break
            if flag == 0:
                raise ValueError("site_names should be two different sites in the list: {}. You gave: {}".format(tmp_name_full,", ".join(sitenames=self._sitenames)))

            tmp_nm = param_name+"".join(site_names)
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.eos_dict['beadlibrary'] and tmp_nm in self.eos_dict['beadlibrary'][bead_names[0]]:
                    param_value = self.eos_dict['beadlibrary'][bead_names[0]][tmp_nm]
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self._crosslibrary and bead_names[0] in self._crosslibrary[bead_names[1]]:
                    param_value = self._crosslibrary[bead_names[1]][bead_names[0]][tmp_nm]
                elif bead_names[0] in self._crosslibrary and bead_names[1] in self._crosslibrary[bead_names[0]]:
                    param_value = self._crosslibrary[bead_names[0]][bead_names[1]][tmp_nm]
                else:
                    param_value = self.check_bounds(param_name, bead_names, np.empty(2))[1]/2

        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(param_types)))

        return param_value

    def check_bounds(self, fit_bead, param_name, bounds):
        """
        Generate initial guesses for the parameters to be fit.
        
        Parameters
        ----------
        fit_bead : str
            Name of bead being fit
        param_name : str
            Parameter to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        param_value : float
            Value of parameter
        
        Returns
        -------
        bounds : list
            A screened and possibly corrected low and a high value for the parameter, param_name
        """
        
        param_bound_extreme = {"epsilon":[0.,1000.], "sigma":[0.,9e-9], "l_r":[0.,100.], "l_a":[0.,100.], "Sk":[0.,1.], "epsilon-a":[0.,5000.], "K":[0.,10000.]}

        bead_names = [fit_bead]

        fit_params_list = param_name.split("_")
        param_name = fit_params_list[0]
        if len(fit_params_list) > 1:
            if fit_params_list[0] == "l":
                if fit_params_list[1] in ["r","a"]:
                    param_name = "_".join([fit_params_list[0],fit_params_list[1]])
                    fit_params_list.remove(fit_params_list[1])

            if len(fit_params_list) > 1:
                bead_names.append(fit_params_list[1])
        
        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))
        
        bounds_new = np.zeros(2)
        # Non bonded parameters
        if (param_name in param_bound_extreme):
            # Self interaction parameter
            if len(bead_names) == 1:
                if bounds[0] < param_bound_extreme[param_name][0]:
                    logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],param_bound_extreme[param_name][0]))
                    bounds_new[0] = param_bound_extreme[param_name][0]
                else:
                    bounds_new[0] = bounds[0]
        
                if (bounds[1] > param_bound_extreme[param_name][1] or bounds[1] < 1e-32):
                    logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],param_bound_extreme[param_name][1]))
                    bounds_new[1] = param_bound_extreme[param_name][1]
                else:
                    bounds_new[1] = bounds[1]
                        
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bounds[0] < param_bound_extreme[param_name][0]:
                    logger.debug("Given {}_{} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bead_names[1],bounds[0],param_bound_extreme[param_name][0]))
                    bounds_new[0] = param_bound_extreme[param_name][0]
                else:
                    bounds_new[0] = bounds[0]

                if (bounds[1] > param_bound_extreme[param_name][1] or bounds[1] < 1e-32):
                    logger.debug("Given {}_{} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bead_names[1],bounds[1],param_bound_extreme[param_name][1]))
                    bounds_new[1] = param_bound_extreme[param_name][1]
                else:
                    bounds_new[1] = bounds[1]
        
        # Association Sites
        elif any([param_name.startswith('epsilon'), param_name.startswith('K')]):
            tmp = [param_name.startswith('epsilon'), param_name.startswith('K')]
            # Ensure sitenames are valid and on list
            if tmp[0] == True:
                param_name_tmp = "epsilon-a"
            elif tmp[1] == True:
                param_name_tmp = "K"

            if bounds[0] < param_bound_extreme[param_name_tmp][0]:
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],param_bound_extreme[param_name_tmp][0]))
                bounds_new[0] = param_bound_extreme[param_name][0]
            else:
                bounds_new[0] = bounds[0]

            if (bounds[1] > param_bound_extreme[param_name_tmp][1] or bounds[1] < 1e-32):
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],param_bound_extreme[param_name_tmp][1]))
                bounds_new[1] = param_bound_extreme[param_name][1]
            else:
                bounds_new[1] = bounds[1]
                
        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(param_types)))
        
        return bounds_new
    
    def update_parameters(self, fit_bead, param_name, param_value):
        r"""
        Update a single parameter value during parameter fitting process.

        To refresh those parameters that are dependent on to _beadlibrary or _crosslibrary, use method "parameter refresh".
        
        Parameters
        ----------
        fit_bead : str
            Name of bead being fit
        param_name : str
            Parameter to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
        param_value : float
            Value of parameter
        """

        param_types = ["epsilon", "sigma", "l_r", "l_a", "Sk", "K"]

        bead_names = [fit_bead]

        fit_params_list = param_name.split("_")
        param_name = fit_params_list[0]
        if len(fit_params_list) > 1:
            if fit_params_list[0] == "l":
                if fit_params_list[1] in ["r","a"]:
                    param_name = "_".join([fit_params_list[0],fit_params_list[1]])
                    fit_params_list.remove(fit_params_list[1])

            if len(fit_params_list) > 1:
                bead_names.append(fit_params_list[1])

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))

        # Non bonded parameters
        if (param_name in ["epsilon", "sigma", "l_r", "l_a", "Sk"]):
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.eos_dict['beadlibrary']:
                    self.eos_dict['beadlibrary'][bead_names[0]][param_name] = param_value
                else:
                    self.eos_dict['beadlibrary'][bead_names[0]] = {param_name: param_value}
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self._crosslibrary and bead_names[0] in self._crosslibrary[bead_names[1]]:
                    self._crosslibrary[bead_names[1]][bead_names[0]][param_name] = param_value
                elif bead_names[0] in self._crosslibrary:
                    if bead_names[1] in self._crosslibrary[bead_names[0]]:
                        self._crosslibrary[bead_names[0]][bead_names[1]][param_name] = param_value
                    else:
                        self._crosslibrary[bead_names[0]][bead_names[1]] = {param_name: param_value} 
                else:
                    self._crosslibrary[bead_names[0]] = {bead_names[1]: {param_name: param_value}}

        # Association Sites
        elif any([param_name.startswith('epsilon'), param_name.startswith('K')]):
            tmp = [param_name.startswith('epsilon'), param_name.startswith('K')]
            # Ensure sitenames are valid and on list
            if tmp[0] == True: tmp_name_full = param_name.replace("epsilon","")
            elif tmp[1] == True: tmp_name_full = param_name.replace("K","")
            flag = 0

            for site1 in self._sitenames:
                if tmp_name_full.startswith(site1):
                    tmp_name = tmp_name_full.replace(site1,"")
                    for site2 in self._sitenames:
                        if tmp_name == site2:
                            flag = 1
                            break
                    if flag == 1:
                        break
            if flag == 0:
                raise ValueError("site_names should be two different sites in the list: {}. You gave: {}".format(tmp_name_full,", ".join(sitenames=self._sitenames)))

            tmp_nm = param_name+"".join(site_names)
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.eos_dict['beadlibrary'] and tmp_nm in self.eos_dict['beadlibrary'][bead_names[0]]:
                    self.eos_dict['beadlibrary'][bead_names[0]][tmp_nm] = param_value
                else:
                    self.eos_dict['beadlibrary'][bead_names[0]] = {tmp_nm: param_value}
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self._crosslibrary and bead_names[0] in self._crosslibrary[bead_names[1]]:
                    self._crosslibrary[bead_names[1]][bead_names[0]][tmp_nm] = param_value
                elif bead_names[0] in self._crosslibrary:
                    if bead_names[1] in self._crosslibrary[bead_names[0]]:
                        self._crosslibrary[bead_names[0]][bead_names[1]][tmp_nm] = param_value
                    else:
                        self._crosslibrary[bead_names[0]][bead_names[1]] = {tmp_nm: param_value}
                else:
                    self._crosslibrary[bead_names[0]] = {bead_names[1]: {tmp_nm: param_value}}

        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(param_types)))

    def parameter_refresh(self):
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on _beadlibrary and _crosslibrary attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        # Update Non bonded matrices
        self.eos_dict['epsilonkl'], self.eos_dict['sigmakl'], self.eos_dict['l_akl'], self.eos_dict['l_rkl'], self.eos_dict['Ckl'] = funcs.calc_interaction_matrices(self.eos_dict['beads'], self.eos_dict['beadlibrary'], crosslibrary=self._crosslibrary)

        # Update Association site matrices
        self.eos_dict['epsilonHB'], self.eos_dict['Kklab'], self.eos_dict['nk'] = funcs.calc_assoc_matrices(self.eos_dict['beads'],self.eos_dict['beadlibrary'],sitenames=self._sitenames,crosslibrary=self._crosslibrary)

        # Update temperature dependent variables
        if np.isnan(self.T) == False:
            self._calc_temperature_dependent_variables(T)


    def __str__(self):

        string = "Beads:" + str(self.eos_dict['beads']) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string


