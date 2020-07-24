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

from despasito.equations_of_state.saft import Aideal
from despasito.equations_of_state.saft import Aassoc

logger = logging.getLogger(__name__)

def saft_type(name):
    
    if name == "gamma_mie":
        from despasito.equations_of_state.saft.gamma_mie import gamma_mie as saft_source
    elif name == "gamma_sw":
        from despasito.equations_of_state.saft.gamma_sw import gamma_sw as saft_source
    else:
        raise ValueError("SAFT type, {}, is not supported. Be sure the class is added to the factory function 'saft_type'".format(name))

    return saft_source

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
        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.
        - Nk*: Optional, The number of sites of from list sitenames. Asterisk represents string from sitenames.

    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - epsilon*: Optional, Interaction energy between each bead and association site. Asterisk represents string from sitenames.
        - K**: Optional, Bonding volume between each association site. Asterisk represents two strings from sitenames.

    Attributes
    ----------
    eos_dict : dict, default: keys = ['beadlibrary', 'beads', 'nui', 'massi']
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):

        saft_source = saft_type(kwargs["saft_name"])
        self.saft_source = saft_source(kwargs)

        if not hasattr(self, 'eos_dict'):
            self.eos_dict = {}
# NoteHere
        # Extract needed variables from saft type file (e.g. gamma_mie)
        saft_attributes = ["Aideal_method", "parameter_types", "parameter_bound_extreme","residual_helmholtz_contributions"]
        for key in saft_attributes:
            try:
                self.eos_dict[key] = getattr(self.saft_source,key)
            except:
                raise ValueError("SAFT type, {}, is missing the variable {}.".format(kwargs["saft_name"],key))

        for res in self.eos_dict["residual_helmholtz_contributions"]:
            setattr( self, res, getattr(self.saft_source, res))

        if "Aideal_method" in kwargs:
            logger.info("Switching Aideal method from {} to {}.".format(self.eos_dict["Aideal_method"],kwargs["Aideal_method"]))
            self.eos_dict["Aideal_method"] = kwargs["Aideal_method"]

        # Extract needed values from kwargs
        needed_attributes = ['beadlibrary',"nui","beads"]
        for key in needed_attributes:
            if key not in kwargs:
                raise ValueError("The one of the following inputs is missing: {}".format(", ".join(tmp)))
            elif not hasattr(self, key):
                self.eos_dict[key] = kwargs[key]
        self.nui = self.eos_dict["nui"]
        self.beads = self.eos_dict["beads"]

        if "mixing_rules" in kwargs:
            for key, value in kwargs["mixing_rules"].items():
                self.saft_source.mixing_rules[key] = value

        if 'crosslibrary' not in kwargs:
            self.eos_dict['crosslibrary'] = {}
        else:
            self.eos_dict['crosslibrary'] = kwargs['crosslibrary']

        if not hasattr(self, 'massi'):
            self.eos_dict['massi'] = tb.calc_massi(self.nui,self.eos_dict['beadlibrary'],self.beads)

        if "reduction_ratio" in kwargs:
            self.eos_dict['reduction_ratio'] = kwargs["reduction_ratio"]
            

        # Initiate association site terms
        self.eos_dict['sitenames'], self.eos_dict['nk'], self.eos_dict['flag_assoc'] = Aassoc.initiate_assoc_matrices(self.beads,self.eos_dict['beadlibrary'],self.nui)
        assoc_output = Aassoc.calc_assoc_matrices(self.beads,self.eos_dict['beadlibrary'],self.nui,sitenames=self.eos_dict['sitenames'],crosslibrary=self.eos_dict['crosslibrary'],nk=self.eos_dict['nk'])
        self.eos_dict.update(assoc_output)
        if np.size(np.where(self.eos_dict['epsilonHB']!=0.0))==0:
            self.eos_dict['flag_assoc'] = False

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

        if len(xi) != len(self.nui):
            raise ValueError("Number of components in mole fraction list doesn't match components in nui. Check bead_config.")
    
        rho = self._check_density(rho)

        if any(np.array(xi) < 0.):
            raise ValueError("Mole fractions cannot be less than zero.")

        Ares = np.zeros(len(rho))
        for res in self.eos_dict["residual_helmholtz_contributions"]:
            Ares += getattr(self.saft_source, res)(rho, T, xi)

        if self.eos_dict["flag_assoc"]:
            Ares += self.Aassoc(rho, T, xi)

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
        rho = self._check_density(rho)

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

        rho = self._check_density(rho)

        return Aideal.Aideal_contribution(rho, T, xi, self.eos_dict["massi"], method=method)

    def Aassoc(self, rho, T, xi):
        r"""
        Return a vector of association site contribution of Helmholtz energy.
    
        :math:`\frac{A^{association}}{N k_{B} T}`
    
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
        Aassoc : numpy.ndarray
            Helmholtz energy of ideal gas for each density given.
        """
        rho = self._check_density(rho)

        # compute F_klab    
        Fklab = np.exp(self.eos_dict['epsilonHB'] / T) - 1.0
        if 'rc_klab' in self.eos_dict:
            if 'rd_klab' in self.eos_dict:
                opts = {"rd_klab": self.eos_dict["rd_klab"]}
            elif "reduction_ratio" in self.eos_dict:
                opts = {"reduction_ratio": self.eos_dict['reduction_ratio']}
            else:
                opts = {}
            Kklab = self.saft_source.calc_Kijklab(T, self.eos_dict["rc_klab"], **opts)
            Ktype = "ijklab"
        else:
            Kklab = self.eos_dict['Kklab']
            Ktype = "klab"

        gr_assoc = self.saft_source.calc_gr_assoc(rho, T, xi, Ktype=Ktype)

        # Compute Xika: with python with numba  {BottleNeck}
        indices = Aassoc.assoc_site_indices(self.eos_dict['nk'], self.nui, xi=xi)
        Xika = Aassoc.calc_Xika_wrap(indices, rho, xi, self.nui, self.eos_dict['nk'], Fklab, Kklab, gr_assoc)

        # Compute A_assoc
        Assoc_contribution = np.zeros(np.size(rho)) 
        for ind, (i, k, a) in enumerate(indices):
            if self.eos_dict['nk'][k, a] != 0.0:
                #tmp = (np.log(Xika[:, i, k, a]) + ((1.0 - Xika[:, i, k, a]) / 2.0))
                tmp = (np.log(Xika[:,ind]) + ((1.0 - Xika[:,ind]) / 2.0))
                Assoc_contribution += xi[i] * self.nui[i, k] * self.eos_dict['nk'][k, a] * tmp

        return Assoc_contribution

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
        # derivative of Aideal_broglie here wrt to rho is 1/rho
        rho = self._check_density(rho)
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
        rho = self._check_density(rho)
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

        maxrho = self.saft_source.density_max(xi, T, maxpack=maxpack)


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
        if not set(bead_names).issubset(self.beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.beads)))

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
        if not set(bead_names).issubset(self.beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.beads)))
        
        bounds_new = np.zeros(2)
        # Non bonded parameters
        if (parameter in self.eos_dict["parameter_bound_extreme"]):
            if len(bead_names) == 2:
                param_name = "{}_{}".format(param_name,bead_names[1])

            if bounds[0] < self.eos_dict["parameter_bound_extreme"][parameter][0]:
                logger.debug("Given {} lower boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],self.eos_dict["parameter_bound_extreme"][parameter][0]))
                bounds_new[0] = self.eos_dict["parameter_bound_extreme"][parameter][0]
            elif bounds[0] > self.eos_dict["parameter_bound_extreme"][parameter][1]:
                logger.debug("Given {} lower boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[0],self.eos_dict["parameter_bound_extreme"][parameter][0]))
                bounds_new[0] = self.eos_dict["parameter_bound_extreme"][parameter][0]
            else:
                bounds_new[0] = bounds[0]
        
            if (bounds[1] > self.eos_dict["parameter_bound_extreme"][parameter][1]):
                logger.debug("Given {} upper boundary, {}, is greater than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],self.eos_dict["parameter_bound_extreme"][parameter][1]))
                bounds_new[1] = self.eos_dict["parameter_bound_extreme"][parameter][1]
            elif (bounds[1] < self.eos_dict["parameter_bound_extreme"][parameter][0]):
                logger.debug("Given {} upper boundary, {}, is less than what is recommended by eos object. Using value of {}.".format(param_name,bounds[1],self.eos_dict["parameter_bound_extreme"][parameter][1]))
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

        parameter_list = param_name.split("-")
        parameter = parameter_list[0]

        if (len(parameter_list) > 1 and len(parameter_list[1:]) != 2):
            raise ValueError("Sitenames should be two different sites in the list: {}. You gave: {}".format(self.eos_dict["sitenames"],", ".join(parameter_list[1:])))

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self.beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self.beads)))

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

        self.saft_source.parameter_refresh(self.eos_dict['beadlibrary'],self.eos_dict['crosslibrary'])

        # Update Association site matrices
        if self.eos_dict["flag_assoc"]: 
            self.eos_dict['epsilonHB'], self.eos_dict['Kklab'] = Aassoc.calc_assoc_matrices(self.beads,self.eos_dict['beadlibrary'],self.nui,sitenames=self.eos_dict['sitenames'],crosslibrary=self.eos_dict['crosslibrary'], nk=self.eos_dict['nk'])

    def _check_density(self,rho):
        r"""
        This function checks the attritutes of the density array
        
        Parameters
        ----------
        rho : numpy.ndarray
            Number density of system [mol/m^3]
        """

        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)
        if len(np.shape(rho)) == 2:
            rho = rho[0]

        if any(np.isnan(rho)):
            raise ValueError("NaN was given as a value of density, rho")
        elif rho.size == 0:
                raise ValueError("No value of density, rho, was given")
        elif any(rho < 0.):
            raise ValueError("Density values cannot be negative.")

        return rho

    def __str__(self):

        string = "Beads: {},\nMasses: {} kg/mol\nSitenames: {}".format(self.beads,self.eos_dict['massi'],self.eos_dict['sitenames'])
        return string


