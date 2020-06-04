# -- coding: utf8 --

"""
    EOS object for Peng-Robinson
    
"""

import numpy as np
import logging

from despasito.equations_of_state.interface import EOStemplate

logger = logging.getLogger(__name__)

class cubic_peng_robinson(EOStemplate):

    """
    EOS object for the Peng-Robinson EOS. 

    All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    xi : list[float]
        Mole fraction of component, only relevant for parameter fitting
    beads : list[str]
        List of unique component names
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - Tc: :math:`T_{C}`, Critical temperature [K]
        - Pc: :math:`P_{C}`, Critical pressure [Pa]
        - omega: :math:`\omega`, Acentric factor 

    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - kij: :math:`k_{ij}`, binary interaction parameter
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):


        # Self interaction parameters
        self._beads = kwargs['beads']
        self._beadlibrary = kwargs['beadlibrary']
        self.nui = np.identity(len(self._beads))

        self._Tc = np.zeros(len(self._beads))
        self._Pc = np.zeros(len(self._beads))
        self._omega = np.zeros(len(self._beads))
        self._kappa = np.zeros(len(self._beads))
        self.alpha = np.empty(len(self._beads))
        self.ai = np.zeros(len(self._beads))
        self.bi = np.zeros(len(self._beads))

        self._R = 8.31446261815324 # [J/mol*K]

        for bead in self._beads:
            if bead in self._beadlibrary:
                ind = self._beads.index(bead)
                try:
                    self._Tc[ind] = self._beadlibrary[bead]["Tc"]
                    self._Pc[ind] = self._beadlibrary[bead]["Pc"]
                    if "omega" in self._beadlibrary[bead]:
                        self._omega[ind] = self._beadlibrary[bead]["omega"]     
                        self._kappa[ind] = 0.37464 + 1.54226*self._omega[ind] - 0.26992*self._omega[ind]**2

                    self.ai[ind] = 0.45723553*(self._R*self._Tc[ind])**2/self._Pc[ind]
                    self.bi[ind] = 0.07779607*(self._R*self._Tc[ind]/self._Pc[ind])

                except:
                    raise ValueError("Either 'Tc' or 'Pc' was not provided for component: {}".format(bead))
            else:
                raise ValueError("Parameters weren't provided for component: {}".format(bead))

        # Cross interaction parameters
        self._kij = np.zeros((len(self._beads),len(self._beads)))
        if 'crosslibrary' in list(kwargs.keys()):
            crosslibrary = kwargs['crosslibrary']
            for key, value in crosslibrary.items():
                if key in self._beads:
                    ind = self._beads.index(key)
                    for key2, value2 in value.items():
                        if key2 in self._beads:
                            jnd = self._beads.index(key2)
                            if "kij" in value2:
                                self._kij[ind,jnd] = value2["kij"]
                                self._kij[jnd,ind] = value2["kij"]
                                logger.info("Parameter 'kij' accepted for interactions between {} and {}".format(key,key2))

        # Initialize temperature attribute
        self.T = np.nan
        self.aij = np.nan
        self.bij = np.nan

    def _calc_temp_dependent_parameters(self,T):
        """
        Compute ai and alpha given temperature
       
        Parameters
        ----------
        T : float
            Temperature of the system [K]

        Attributes
        ----------
        ai : numpy.ndarray
            Peng-Robinson parameter a [m^6/mol^2]
        alpha : numpy.ndarray
            Peng-Robinson parameter b [m^3/mol]
        """
        for bead in self._beads: 
            if bead in self._beadlibrary:
                ind = self._beads.index(bead)
                if self._kappa[ind]:
                    self.alpha[ind] = (1+self._kappa[ind]*(1-np.sqrt(T/self._Tc[ind])))**2
                else:
                    self.alpha[ind] = 1.0

    def _calc_mixed_parameters(self,xi,T):

        """
        Compute mixing aij and bij given composition
       
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]

        Attributes
        ----------
        aij : float
            Peng-Robinson parameter a [m^6/mol^2]
        bij : float
            Peng-Robinson parameter b [m^3/mol]
        """

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        aij = 0
        index = range(len(xi))
        for i in index:
            for j in index:
                aij += xi[i]*xi[j]*np.sqrt(self.ai[i]*self.alpha[i]*self.ai[j]*self.alpha[j])*(1.-self._kij[i][j])

        self.aij = aij
        self.bij = np.sum(xi*self.bi)

    def pressure(self, rho, T, xi):
        """
        Compute pressure given system information
       
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

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi,T)
        
        if np.isscalar(rho):
            rho = np.array([rho])
        elif type(rho) != np.ndarray:
            rho = np.array(rho)

        P = self._R*self.T * rho / (1-self.bij*rho) - rho**2*self.aij/((1+self.bij*rho)+rho*self.bij*(1-self.bij*rho))

        return P

    def fugacity_coefficient(self, P, rho, xi, T):

        """
        Compute fugacity coefficient
      
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
            :math:`\mu_i`, Array of chemical potential values for each component
        """

        if T != self.T:
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi,T)

        Z = P/(T*self._R*rho)
        Ai = self.ai*self.alpha*P/(self._R*T)**2
        Bi = self.bi*P/(self._R*T)
        B = self.bij*P/(self._R*T)
        A = self.aij*P/(self._R*T)**2

        sqrt2 = np.sqrt(2.0)
        tmp1 = A/(2.0*sqrt2*B)*np.log((Z+(1+sqrt2)*B)/(Z+(1-sqrt2)*B))
        tmp3 = Bi*(Z-1)/B-np.log(Z-B)
        tmp2 = np.zeros(len(xi))

        index = range(len(xi))
        for i in index:
            Aij = np.zeros(len(xi))
            for j in index:
                Aij[j] = np.sqrt(Ai[i]*Ai[j])*(1.-self._kij[i][j])
            tmp2[i] = Bi[i]/B - 2*np.sum(xi*Aij)/A
        phi = np.exp(tmp1*tmp2+tmp3)

        return phi

    def density_max(self, xi, T, maxpack=0.9):

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
            self.T = T
            self._calc_temp_dependent_parameters(T)

        self._calc_mixed_parameters(xi,T)

        maxrho = maxpack /self.bij

        return maxrho

    def param_guess(self, param_name, bead_names):
        r"""
        Update a single parameter value to _beadlibrary or _crosslibrary attributes during parameter fitting process.
            
        To refresh those parameters that are dependent on these libraries, use method "parameter refresh".
            
        Parameters
        ----------
        param_name : str
            name of parameters being updated
        bead_names : list
            List of bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
            
        Returns
        -------
        param_initial_guess : numpy.ndarray,
            An initial guess for parameter, it will be optimized throughout the process.
        """
        
        param_types = ["ai", "bi", "kij"]
        
        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self._beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self._beads)))
        
        # Non bonded parameters
        if (param_name in param_types):
            # Parameter kij
            if "kij" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    if bead_names[1] in self._beads:
                        jnd = self._beads.index(bead_names[1])
                        param_value = self._kij[ind,jnd]
            elif "ai" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    param_value = self.ai[ind]
            elif "bi" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    param_value = self.bi[ind]
    
        else:
            raise ValueError("The parameter name %s is not found in the allowed parameter types: %s" % (param_name,", ".join(param_types)))
                
        return param_value
            
    def check_bounds(self, param_name, bead_names, bounds):
        """
        Generate initial guesses for the parameters to be fit.
        
        Parameters
        ----------
        param_name : str
            Parameter to be fit. See EOS mentation for supported parameter names.
        bead_names : list
            Bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        bounds : list
            A low and a high value for the parameter, param_name
        
        Returns
        -------
        param_initial_guess : numpy.ndarray,
            An initial guess for parameter, it will be optimized throughout the process.
        bounds : list
            A screened and possibly corrected low and a high value for the parameter, param_name
        """
        
        param_types = {"ai":[0., 50.], "bi":[0., 1e-3], "kij":[-1.,1.]}
        
        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self._beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self._beads)))
        
        # Non bonded parameters
        if (param_name in param_types):
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

        else:
            raise ValueError("The parameter name %s is not found in the allowed parameter types: %s" % (param_name,", ".join(param_types)))
        
        return bounds_new

    def update_parameters(self, param_name, bead_names, param_value):
        r"""
        Update a single parameter value to _beadlibrary or _crosslibrary attributes during parameter fitting process. 

        To refresh those parameters that are dependent on these libraries, use method "parameter refresh".
        
        Parameters
        ----------
        param_name : str
            name of parameters being updated
        bead_names : list
            List of bead names to be changed. For a self interaction parameter, the length will be 1, for a cross interaction parameter, the length will be two.
        param_value : float
            Value of parameter
        """

        param_types = ["ai", "bi", "kij"]

        if len(bead_names) > 2:
            raise ValueError("The bead names {} were given, but only a maximum of 2 are permitted.".format(", ".join(bead_names)))
        if not set(bead_names).issubset(self._beads):
            raise ValueError("The bead names {} were given, but they are not in the allowed list: {}".format(", ".join(bead_names),", ".join(self._beads)))

        # Non bonded parameters
        if (param_name in param_types):
            # Parameter kij
            if "kij" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    if bead_names[1] in self._beads:
                        jnd = self._beads.index(bead_names[1])
                        self._kij[ind,jnd] = param_value
                        self._kij[jnd,ind] = param_value
            elif "ai" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    self.ai[ind] = param_value
            elif "bi" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    self.bi[ind] = param_value

        else:
            raise ValueError("The parameter name %s is not found in the allowed parameter types: %s" % (param_name,", ".join(param_types)))

    def __str__(self):

        string = "Beads:" + str(self._beads) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string


