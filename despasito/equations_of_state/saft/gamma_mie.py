# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""

import numpy as np
import logging

from . import constants
from . import gamma_mie_funcs as funcs
# Later this line will be in an abstract class file in this directory, and all versions of SAFT will reference it
from despasito.equations_of_state.interface import EOStemplate

logger = logging.getLogger(__name__)

# ________________ Saft Family ______________
# NoteHere: Insert SAFT family abstract class in this directory to clean up


class saft_gamma_mie(EOStemplate):

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

        self.eos_dict = {}
        # Self interaction parameters
        self.eos_dict['nui'] = kwargs['nui']
        self.eos_dict['beads'] = kwargs['beads']
        self.eos_dict['beadlibrary'] = kwargs['beadlibrary']

        if "num_rings" in kwargs:
            self.eos_dict['num_rings'] = kwargs['num_rings']
            logger.info("Accepted component ring structure: {}".format(kwargs["num_rings"]))
        else:
            self.eos_dict['num_rings'] = np.zeros(len(self.eos_dict['nui']))

        massi = np.zeros(len(self.eos_dict['nui']))
        for i in range(len(self.eos_dict['nui'])):
            for k in range(np.size(self.eos_dict['beads'])):
                massi[i] += self.eos_dict['nui'][i, k] * self.eos_dict['beadlibrary'][self.eos_dict['beads'][k]]["mass"]
        self.eos_dict['massi'] = massi

        # Cross interaction parameters
        if 'crosslibrary' in kwargs:
            crosslibrary = kwargs['crosslibrary']
        else:
            crosslibrary = {}

        epsilonkl, sigmakl, l_akl, l_rkl, Ckl = funcs.calc_interaction_matrices(self.eos_dict['beads'], self.eos_dict['beadlibrary'], crosslibrary=crosslibrary)

        self._crosslibrary = crosslibrary
        self.eos_dict['epsilonkl'] = epsilonkl
        self.eos_dict['sigmakl'] = sigmakl
        self.eos_dict['l_akl'] = l_akl
        self.eos_dict['l_rkl'] = l_rkl
        self.eos_dict['Ckl'] = Ckl

        # Association sites
        if 'sitenames' in kwargs:
            self._sitenames = kwargs['sitenames']
        else:
            self._sitenames = ["H", "e1", "e2"]

        epsilonHB, Kklab, nk = funcs.calc_assoc_matrices(self.eos_dict['beads'], self.eos_dict['beadlibrary'], sitenames=self._sitenames, crosslibrary=self._crosslibrary)

        self.eos_dict['epsilonHB'] = epsilonHB
        self.eos_dict['Kklab'] = Kklab
        self.eos_dict['nk'] = nk

        # Initialize temperature attribute
        self.T = np.nan

    def _temp_dependent_variables(self, T):

        """
        Temperature dependent variables are initialized or updated.
    
        Parameters
        ----------
        T : float, default: numpy.nan
            Temperature of the system [K]
    
        Attributes
        ----------
        T : float, default: numpy.nan
            Temperature of the system
        """

        dkk, dkl, x0kl = funcs.calc_hard_sphere_matricies(self.eos_dict['beads'], self.eos_dict['beadlibrary'], self.eos_dict['sigmakl'], T)
        self.T = T
        self.eos_dict['dkk'] = dkk
        self.eos_dict['dkl'] = dkl
        self.eos_dict['x0kl'] = x0kl

    def _xi_dependent_variables(self, xi):

        """
        Variables dependent on composition are initialized or updated.
    
        Parameters
        ----------
        xi : list[float]
            Mole fraction of component

        """

        Cmol2seg, xsk, xskl = funcs.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beads'], self.eos_dict['beadlibrary'])
        self.eos_dict['Cmol2seg'] = Cmol2seg
        self.eos_dict['xsk'] = xsk
        self.eos_dict['xskl'] = xskl

    def pressure(self, rho, T, xi):
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

        if len(xi) != len(self.eos_dict['nui']):
            raise ValueError("Number of components in mole fraction list doesn't match components in nui. Check bead_config.")

        if T != self.T:
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)
        
        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        if np.all(rho > self.density_max(xi, T)):
            logger.error("Density value, {}, should not all be greater than {}, or calc_Amono will fail in log calculation.".format(rho, self.density_max(xi, T)))

        rho = rho*constants.Nav

        step = np.sqrt(np.finfo(float).eps) *rho * 10000.0
        # Decreasing step size by 2 orders of magnitude didn't reduce noise in P values
        nrho = np.size(rho)

        # computer rho+step and rho-step for better a bit better performance
        A = funcs.calc_A(rho=np.append(rho + step, rho - step), xi=xi, T=T, **self.eos_dict)

        P_tmp = (A[:nrho]-A[nrho:])*((constants.kb*T)/(2.0*step))*rho**2

        return P_tmp

    def fugacity_coefficient(self, P, rho, xi, T, dy=1e-4):

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

        if len(xi) != len(self.eos_dict['nui']):
            raise ValueError("Number of components in mole fraction list doesn't match components in nui. Check bead_config.")

        if len(rho.shape) > 1:
            rho = rho[0]

        if np.all(rho > self.density_max(xi, T)):
            raise ValueError("Density value, {}, should not all be greater than {}, or calc_Amono will fail in log calculation.".format(rho, self.density_max(xi, T)))

        if T != self.T:
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)

        Z = P / (rho * T * constants.Nav * constants.kb)
        phi_tmp = np.zeros(len(xi))

#        #### Traditional Central Difference Method
#        # Set step size in finite difference method
#        exp = np.floor(np.log10(rho))-3 # Make sure step size is three orders of magnitude lower
#        drho = 10**exp
#        logger.debug("    Compute phi for density, {}, with step size {}.".format(rho,drho))
#
#        # compute phi
#        Ares = funcs.calc_Ares(rho=rho *  constants.Nav, xi=xi, T=T, **self.eos_dict)
#        rhoi = rho*np.array(xi,float)
#        for i in range(np.size(phi_tmp)):
#            dAres = np.zeros(2)
#            for j, delta in enumerate((drho, -drho)):
#                rhoi_temp = np.copy(rhoi)
#                if rhoi_temp[i] != 0.:
#                    rhoi_temp[i] += delta
#                dAres[j] = self._calc_dAres_drhoi_wrap(T, rhoi_temp)
#            phi_tmp[i] = Ares + rho*(dAres[0] - dAres[1]) / (2.0 * drho) - np.log(Z) 
#            #with open("OldPhi.csv","a") as f:
#            #    f.write("{}, {}, {}, {}, {}, {}, {}\n".format(i,xi[i],phi_tmp[i], Ares, rho, dAres, drho))

        #### Transform y=log(rhoi) Central Difference Method without worrying about negative mole fractions 
        # Set step size in finite difference method
        y = np.log(rho*np.array(xi,float))
        #dy = 0.05

        # compute phi
        Ares = funcs.calc_Ares(rho=rho *  constants.Nav, xi=xi, T=T, **self.eos_dict)
        for i in range(np.size(phi_tmp)):
            if xi[i] != 0.0:
                dAres = np.zeros(2)
                for j, delta in enumerate((dy, -dy)):
                    y_temp = np.copy(y)
                    y_temp[i] += delta
                    dAres[j] = self._calc_dAres_drhoi_wrap(T, np.exp(y_temp))
                phi_tmp[i] = np.exp(Ares + rho/np.exp(y[i])*(dAres[0] - dAres[1]) / (2.0 * dy) - np.log(Z))
            else:
                phi_tmp[i] = 1e-32 # This should be zero, but to prevent the thermo calculation from complaining about diving by zero we give it a value, the mole fraction is zero though, so it'll go away.
        ##########################################

        # Reset composition dependent variables
        self._xi_dependent_variables(xi)

        return phi_tmp

    def _calc_dAres_drhoi_wrap(self, T, rhoi):
        """
        Compute derivative of Helmholtz energy wrt to density.
      
        Parameters
        ----------
        T : float
            Temperature of the system [K]
        rhoi : float
            Molar density of each component, add up to the total density [mol/m^3]
    
        Returns
        -------
        Ares : float
            Helmholtz energy give number of moles, length of array rho
        """

        if T != self.T:
            self._temp_dependent_variables(T)

        # Calculate new xi values
        rho = np.array([np.sum(rhoi)])
        xi = rhoi/rho

        self._xi_dependent_variables(xi)

        Cmol2seg, xsk, xskl = funcs.calc_composition_dependent_variables(xi, self.eos_dict['nui'], self.eos_dict['beads'], self.eos_dict['beadlibrary'])

        Ares = funcs.calc_Ares(rho=rho * constants.Nav, xi=xi, T=T, **self.eos_dict)

        return Ares

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

        if T != self.T:
            self._temp_dependent_variables(T)

        self._xi_dependent_variables(xi)

        # estimate the maximum density based on the hard sphere packing fraction
        # etax, assuming a maximum packing fraction specified by maxpack
        maxrho = maxpack * 6.0 / (self.eos_dict['Cmol2seg'] * np.pi * np.sum(self.eos_dict['xskl'] * (self.eos_dict['dkl']**3))) / constants.Nav

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
            if len(bead_names) == 2:
                name = "_".join([param_name,bead_names[1]])
            else:
                name = param_name

            if bounds[0] < param_bound_extreme[param_name][0]:
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(name,bounds[0],param_bound_extreme[param_name][0]))
                bounds_new[0] = param_bound_extreme[param_name][0]
                if bounds_new[0] >= bounds[1]:
                    logger.debug("New {} lower boundary, {}, is greater than given upper boundary, {}. Using value of {}.".format(name,bounds_new[0],bounds[1],param_bound_extreme[param_name][1]))
                    bounds[1] = param_bound_extreme[param_name][1]
            else:
                bounds_new[0] = bounds[0]
        
            if bounds[1] > param_bound_extreme[param_name][1]:
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(name,bounds[1],param_bound_extreme[param_name][1]))
                bounds_new[1] = param_bound_extreme[param_name][1]
                if bounds_new[0] >= bounds_new[1]:
                    logger.debug("New {} upper boundary, {}, is greater than lower boundary, {}. Using lower boundary of {}.".format(name,bounds_new[1],bounds_new[0],param_bound_extreme[param_name][0]))
                    bounds_new[0] = param_bound_extreme[param_name][0]
            elif bounds[1] < bounds_new[0]:
                logger.debug("Given {} upper boundary, {}, is less than the given lower bound. Using value of {}.".format(name,bounds[1],param_bound_extreme[param_name][1]))
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
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(name,bounds[0],param_bound_extreme[param_name_tmp][0]))
                bounds_new[0] = param_bound_extreme[param_name_tmp][0]
                if bounds_new[0] >= bounds[1]:
                    logger.debug("New {} lower boundary, {}, is greater than given upper boundary, {}. Using value of {}.".format(name,bounds_new[0],bounds[1],param_bound_extreme[param_name_tmp][1]))
                    bounds[1] = param_bound_extreme[param_name_tmp][1]
            else:
                bounds_new[0] = bounds[0]
        
            if bounds[1] > param_bound_extreme[param_name_tmp][1]:
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(name,bounds[1],param_bound_extreme[param_name_tmp][1]))
                bounds_new[1] = param_bound_extreme[param_name_tmp][1]
                if bounds_new[0] >= bounds_new[1]:
                    logger.debug("New {} upper boundary, {}, is greater than lower boundary, {}. Using lower boundary of {}.".format(name,bounds_new[1],bounds_new[0],param_bound_extreme[param_name_tmp][0]))
                    bounds_new[0] = param_bound_extreme[param_name_tmp][0]
            elif bounds[1] < bounds_new[0]:
                logger.debug("Given {} upper boundary, {}, is less than the given lower bound. Using value of {}.".format(name,bounds[1],param_bound_extreme[param_name_tmp][1]))
                bounds_new[1] = param_bound_extreme[param_name_tmp][1]
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
            self._temp_dependent_variables(self.T)


    def __str__(self):

        string = "Beads:" + str(self.eos_dict['beads']) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string


