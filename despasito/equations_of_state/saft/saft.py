# -- coding: utf8 --

r"""
    
    EOS object for SAFT-:math:`\gamma`-Mie
    
    Equations referenced in this code are from V. Papaioannou et al J. Chem. Phys. 140 054107 2014
    
"""
import sys
import numpy as np
import logging
#np.set_printoptions(threshold=sys.maxsize)

from despasito.equations_of_state import constants
import despasito.utils.general_toolbox as gtb
import despasito.equations_of_state.toolbox as tb
from despasito.equations_of_state.interface import EOStemplate

from despasito.equations_of_state.saft.helmholtz import Aideal
#from despasito.equations_of_state.saft.helmholtz import Aassoc

logger = logging.getLogger(__name__)

def saft_type(name):
    
    if name == "gamma_mie":
###################
        from despasito.equations_of_state.saft.helmholtz import gamma_mie as saft_source
        from despasito.equations_of_state.saft.helmholtz.Amonomer_mie import Amonomer
        from despasito.equations_of_state.saft.helmholtz.Achain_mie import Achain
        from despasito.equations_of_state.saft.helmholtz.Aassoc import Aassoc
    
    return saft_source, Amonomer, Achain, Aassoc
###################

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

        - mass: Bead mass [kg/mol]
        - epsilon: :math:`\epsilon_{k,k}/k_B`, Energy well depth scaled by Boltzmann constant
        - sigma: :math:`\sigma_{k,k}`, Size parameter [m]
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
    eos_dict : dict, default: keys = ['beadlibrary', 'beads', 'nui', 'massi']
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):

###################
        saft_source, Amonomer, Achain, Aassoc = saft_type(kwargs["saft_name"])
        
        self.Amonomer = Amonomer(kwargs)
        self.Achain = Achain(kwargs)
        self.Aassoc = Aassoc(kwargs)
###################

        if not hasattr(self, 'eos_dict'):
            self.eos_dict = {}

        # Extract needed variables from saft type file (e.g. gamma_mie)
        saft_attributes = ["Aideal_method", "parameter_types", "parameter_bound_extreme"]
        for key in saft_attributes:
            try:
                self.eos_dict[key] = getattr(saft_source,key)
            except:
                raise ValueError("SAFT type, {}, is missing the variable {}.".format(kwargs["saft_name"],key))

        # Extract needed values from kwargs
        needed_attributes = ['nui','beads','beadlibrary']
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                self.eos_dict[key] = kwargs[key]

        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.eos_dict['nui'],self.eos_dict['beadlibrary'],self.eos_dict['beads'])

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
    
        if len(xi) != len(self.eos_dict['nui']):
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

        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        A = self.residual_helmholtz_energy(rho, T, xi) + self.Aideal(rho, T, xi, method=self.eos_dict["Aideal_method"])

        return A

    def Aideal(self, rho, T, xi, method="Abroglie"):
        r"""
        Return a vector of ideal contribution of Helmholtz energy.
    
        :math:`\frac{A^{ideal}}{N k_{B} T}`
    
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        T : float
            Temperature of the system [K]
        xi : numpy.ndarray
            Mole fraction of each component, sum(xi) should equal 1.0
        massi : numpy.ndarray
            Vector of component masses that correspond to the mole fractions in xi [kg/mol]
        method : str, Optional, default: Abroglie
            The function name of the method to calculate the ideal contribution of the helmholtz energy. To add a new one, add a function to: despasito.equations_of_state,helholtz.Aideal.py
    
        Returns
        -------
        Aideal : numpy.ndarray
            Helmholtz energy of ideal gas for each density given.
        """

        self._check_density(rho)

        return Aideal.Aideal_contribution(rho, T, xi, self.eos_dict["massi"], method=method)

    def pressure(self, rho, T, xi, step_size=1E-6):
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

        P_tmp = gtb.central_difference(rho, self.helmholtz_energy, args=(T, xi), step_size=step_size)
        pressure = P_tmp*T*constants.R*rho**2

        return pressure

    def fugacity_coefficient(self, P, rho, xi, T, dy=1e-5, log_method=True):

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
        log_method : bool, Optional, default: False
            Choose to use a log transform in central difference method. This allows easier calulations for very small numbers.
    
        Returns
        -------
        mui : numpy.ndarray
            Array of chemical potential values for each component
        """

        logZ = np.log(P / (rho * T * constants.R))
        Ares = self.residual_helmholtz_energy(rho, T, xi)
        dAresdrho = tb.partial_density_central_difference(xi, rho, T, self.residual_helmholtz_energy, step_size=dy, log_method=True)

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

    def param_guess(self, parameter, bead_names):
        """
        Generate initial guesses for the parameters to be fit.

        Parameters
        ----------
        parameter : str
            Parameter to be fit. See EOS documentation for supported parameter names.
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.

        Returns
        -------
        param_initial_guess : numpy.ndarray, 
            An initial guess for parameter, it will be optimized throughout the process.
        """

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))

        param_name = parameter.split("-")[0]

        if param_name not in self.eos_dict["parameter_types"]:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.eos_dict["parameter_types"])))

        param_value = None
        # Self interaction parameter
        if len(bead_names) == 1:
            if bead_names[0] in self.eos_dict['beadlibrary']:
                param_value = self.eos_dict['beadlibrary'][bead_names[0]][parameter]
        # Cross interaction parameter
        elif len(bead_names) == 2:
            if bead_names[1] in self.eos_dict['crosslibrary'] and bead_names[0] in self.eos_dict['crosslibrary'][bead_names[1]]:
                param_value = self.eos_dict['crosslibrary'][bead_names[1]][bead_names[0]][parameter]
            elif bead_names[0] in self.eos_dict['crosslibrary'] and bead_names[1] in self.eos_dict['crosslibrary'][bead_names[0]]:
                param_value = self.eos_dict['crosslibrary'][bead_names[0]][bead_names[1]][parameter]

        if param_value is None:
            bounds = self.check_bounds(bead_names[0], parameter, np.empty(2))
            param_value = (bounds[1]-bounds[0])/2 + bounds[0]

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

        parameter = param_name.split("-")[0]
        
        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))
        
        bounds_new = np.zeros(2)
        # Non bonded parameters
        if (parameter in self.eos_dict["parameter_bound_extreme"]):
            if len(bead_names) == 2:
                param_name = "{}_{}".format(param_name,bead_names[1])

            if bounds[0] < self.eos_dict["parameter_bound_extreme"][parameter][0]:
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],self.eos_dict["parameter_bound_extreme"][parameter][0]))
                bounds_new[0] = self.eos_dict["parameter_bound_extreme"][parameter][0]
            else:
                bounds_new[0] = bounds[0]
        
            if (bounds[1] > self.eos_dict["parameter_bound_extreme"][parameter][1] or bounds[1] < np.finfo("float").eps):
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],self.eos_dict["parameter_bound_extreme"][parameter][1]))
                bounds_new[1] = self.eos_dict["parameter_bound_extreme"][parameter][1]
            else:
                bounds_new[1] = bounds[1]
                        
        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.eos_dict["parameter_types"])))
        
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

        parameter_list = param_name.split("-")[0]
        parameter = parameter_list[0]

        if len(parameter_list) > 1 and len(parameter_list[1:]) != 2 or parameter_list[1]==parameter_list[2]:
            raise ValueError("sitenames should be two different sites in the list: {}. You gave: {}".format(self.eos_dict["sitenames"],", ",join(parameter_list[1:])))

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.eos_dict['beads']):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.eos_dict['beads'])))

        if parameter in self.eos_dict["parameter_types"]:
            # Self interaction parameter
            if len(bead_names) == 1:
                if bead_names[0] in self.eos_dict['beadlibrary']:
                    self.eos_dict['beadlibrary'][bead_names[0]][param_name] = param_value
                else:
                    self.eos_dict['beadlibrary'][bead_names[0]] = {param_name: param_value}
            # Cross interaction parameter
            elif len(bead_names) == 2:
                if bead_names[1] in self.eos_dict['crosslibrary'] and bead_names[0] in self.eos_dict['crosslibrary'][bead_names[1]]:
                    self.eos_dict['crosslibrary'][bead_names[1]][bead_names[0]][param_name] = param_value
                elif bead_names[0] in self.eos_dict['crosslibrary']:
                    if bead_names[1] in self.eos_dict['crosslibrary'][bead_names[0]]:
                        self.eos_dict['crosslibrary'][bead_names[0]][bead_names[1]][param_name] = param_value
                    else:
                        self.eos_dict['crosslibrary'][bead_names[0]][bead_names[1]] = {param_name: param_value} 
                else:
                    self.eos_dict['crosslibrary'][bead_names[0]] = {bead_names[1]: {param_name: param_value}}

        else:
            raise ValueError("The parameter name {} is not found in the allowed parameter types: {}".format(param_name,", ".join(self.eos_dict["parameter_types"])))

        # NoteHere: update with cross library and beadlibrary

    def parameter_refresh(self):
        r""" 
        To refresh dependent parameters
        
        Those parameters that are dependent on _beadlibrary and _crosslibrary attributes **must** be updated by running this function after all parameters from update_parameters method have been changed.
        """

        # Update Non bonded matrices
        self.eos_dict['epsilonkl'], self.eos_dict['sigmakl'], self.eos_dict['l_akl'], self.eos_dict['l_rkl'], self.eos_dict['Ckl'] = funcs.calc_interaction_matrices(self.eos_dict['beads'], self.eos_dict['beadlibrary'], crosslibrary=self.eos_dict['crosslibrary'])

        # Update Association site matrices
        self.eos_dict['epsilonHB'], self.eos_dict['Kklab'], self.eos_dict['nk'] = funcs.calc_assoc_matrices(self.eos_dict['beads'],self.eos_dict['beadlibrary'],sitenames=self.eos_dict['sitenames'],crosslibrary=self.eos_dict['crosslibrary'])

        # Update temperature dependent variables
        if np.isnan(self.T) == False:
            self._calc_temperature_dependent_variables(T)

    @staticmethod
    def _check_density(rho):
        r"""
        This function checks the attritutes of
        
        Parameters
        ----------
        rho : numpy.ndarray
        Number density of system [mol/m^3]
        T : float
        Temperature of the system [K]
        xi : numpy.ndarray
        Mole fraction of each component, sum(xi) should equal 1.0
        """
        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
            raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

    def __str__(self):

        string = "Beads: {},\nMasses: {} kg/mol\n".format(self.eos_dict['beads'],self.eos_dict['massi'])
        string += "T:" + str(self.T) + "\n"
        return string


