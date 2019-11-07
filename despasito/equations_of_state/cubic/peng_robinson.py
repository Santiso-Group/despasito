# -- coding: utf8 --

r"""
    
    EOS object for Peng-Robinson
    
"""

import sys
import numpy as np
import logging

from despasito.equations_of_state.interface import EOStemplate



class cubic_peng_robinson(EOStemplate):

    r"""
    Initialize EOS with system component parameters so that methods may be used by thermodynamic calculations. All input and calculated parameters are defined as hidden attributes.
    
    Parameters
    ----------
    xi : list[float]
        Mole fraction of component, only relevant for parameter fitting
    beads : list[str]
        List of unique component names
    beadlibrary : dict
        A dictionary where bead names are the keys to access EOS self interaction parameters:

        - T_{C}: Critical temperature [K]
        - P_{C}: Ctritical pressure [Pa]

    crosslibrary : dict, Optional, default: {}
        Optional library of bead cross interaction parameters. As many or as few of the desired parameters may be defined for whichever group combinations are desired.

        - kij: binary interaction parameter
        
    Attributes
    ----------
    T : float, default: numpy.nan
        Temperature value is initially defined as NaN for a placeholder until temperature dependent attributes are initialized by using a method of this class.
    
    """

    def __init__(self, kwargs):

        logger = logging.getLogger(__name__)

        # Self interaction parameters
        self._beads = kwargs['beads']
        self._beadlibrary = kwargs['beadlibrary']

        self._Tc = np.zeros(len(self._beads))
        self._Pc = np.zeros(len(self._beads))
        self._omega = np.zeros(len(self._beads))
        self._m = np.zeros(len(self._beads))
        self._ai = np.zeros(len(self._beads))
        self._bi = np.zeros(len(self._beads))

        self._R = 8.31446261815324 # [J/mol*K]

        for bead in self._beads:
            if bead in self._beadlibrary:
                ind = self._beads.index(bead)
                try:
                    self._Tc[ind] = self._beadlibrary[bead]["Tc"]
                    self._Pc[ind] = self._beadlibrary[bead]["Pc"]
                    if "omega" in self._beadlibrary[bead]:
                        self._omega[ind] = self._beadlibrary[bead]["omega"]     
                        self._m[ind] = 0.37464 + 1.54226*self._omega[ind] - 0.26992*self._omega[ind]**2

                    self._ai[ind] = 0.45723553*(self._R*self._Tc[ind])**2/self._Pc[ind]
                    self._bi[ind] = 0.07779607*(self._R*self._Tc[ind]/self._Pc[ind])

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

    def _calc_mixed_parameters(self,xi,T):

        """
        Compute mixing aij and bij given compositon
       
        Parameters
        ----------
        xi : list[float]
            Mole fraction of each component
        T : float
            Temperature of the system [K]

        Attributes
        ----------
        aij : float
            Peng-Robnson parameter a [m^6/mol^2]
        bij : float
            Peng-Robnson parameter b [m^3/mol]
        """

        aij = 0
        index = range(len(xi))
        for i in index:
            for j in index:
                aii = self._ai[i]*np.sqrt(1+self._m[i]*(1-np.sqrt(T/self._Tc[i])))
                ajj = self._ai[j]*np.sqrt(1+self._m[j]*(1-np.sqrt(T/self._Tc[j])))
                aij += xi[i]*xi[j]*np.sqrt(aii*ajj)*(1.-self._kij[i][j])

        self.aij = aij
        self.bij = np.sum(xi*self._bi)

        print("aij",self.aij)


    def P(self, rho, T, xi):
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

        logger = logging.getLogger(__name__)

        if T != self.T:
            self.T = T

        self._calc_mixed_parameters(xi,T)
        
        if type(rho) != np.ndarray:
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
            Array of chemical potential values for each component
        """

        logger = logging.getLogger(__name__)

        if T != self.T:
            self.T = T

        self._calc_mixed_parameters(xi,T)

        phi = np.exp( -np.log(1-self.bij*rho) - self.aij/(self.bij*self._R*T*np.sqrt(8))*np.log((1+(1+np.sqrt(2))*self.bij*rho)/(1+(1-np.sqrt(2))*self.bij*rho)))

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

        logger = logging.getLogger(__name__)

        self._calc_mixed_parameters(xi,T)

        maxrho = maxpack /self.bij

        return maxrho

    def param_guess(self,fit_params):
        """
        Generate initial guesses for the parameters to be fit.

        Parameters
        ----------
        fit_params : list[str]
        A list of parameters to be fit. See EOS mentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).

        Returns
        -------
        param_initial_guesses : numpy.ndarray, 
            An array of initial guesses for parameters, these will be optimized throughout the process.
    """

        logger = logging.getLogger(__name__)

        l_fitparams = len(fit_params)
        if l_fitparams == 1:
            param_initial_guesses = np.array([l_fitparams])
        else:
            param_initial_guesses = np.zeros(1,l_fitparams)

        for i,param in enumerate(fit_params):
            if param == "ai":
                param_initial_guesses[i] = 1e+7
            elif param == "bi":
                param_initial_guesses[i] = 1e+11
            else:
                raise ValueError("The parameter name %s does not fall under any of the catagories, ai or bi") 

        return param_initial_guesses

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

        logger = logging.getLogger(__name__)

        param_types = ["ai", "bi", "kij"]

        if len(bead_names) > 2:
            raise ValueError("The bead names %s were given, but only a maximum of 2 are permitted." % (", ".join(bead_names)))
        if not set(bead_names).issubset(self._beads):
            raise ValueError("The bead names %s were given, but they are not in the allowed list: " % (", ".join(bead_names),", ".join(self._beads)))

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
                    self._ai[ind] = param_value
            elif "bi" == param_name:
                if bead_names[0] in self._beads:
                    ind = self._beads.index(bead_names[0])
                    self._bi[ind] = param_value

        else:
            raise ValueError("The parameter name %s is not found in the allowed parameter types: %s" % (param_name,", ".join(param_types)))

    def __str__(self):

        string = "Beads:" + str(self._beads) + "\n"
        string += "T:" + str(self.T) + "\n"
        return string


