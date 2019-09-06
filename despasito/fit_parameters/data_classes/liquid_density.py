r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import sys
import numpy as np
from multiprocessing import Pool

from despasito.fit_parameters import fit_funcs as ff
from despasito.fit_parameters.interface import ExpDataTemplate

##################################################################
#                                                                #
#                       Liquid Density                           #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for liquid density data. This data is evaluated with "liquid_properties". 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of type rhol.
        * name : str, data type, in this case RhoL
        * calctype : str, Optional, default: 'liquid_properties'
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in liquid_properties calculations
        * weights : list/float, Either a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case rhol
    calctype : str, Optional, default: 'liquid_properties'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        try:
            self.calctype = data_dict["calctype"]
        except:
            self.calctype = "liquid_properties"

        try:
            self.xi = data_dict["xi"]
            self.T = data_dict["T"]
        except:
            raise ImportError("Given liquid property data, values for T and xi should have been provided.")
        try:
            self.rhol = data_dict["rhol"]
        except:
            pass

        try:
            self.P = data_dict["P"]
            if (type(P) == float or len(P)==1):
                P = np.ones(self.T)*P
        except:
            P = np.ones(self.T)*101325.0

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = 1.0

        try:
            self._rhodict = data_dict["rhodict"]
        except:
            self._rhodict = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

    def _thermo_wrapper(self, P, T, xi,eos,rhodict={}):

        """
        Generate thermodynamic predictions from eos object

        Parameters
        ----------
        T : float
            Temperature of system [K]
        xi : list
            Liquid mole fractions, must add up to one
        eos : obj
            EOS object with updated parameters
        rhodict : dict, Optional, default: {}
            Dictionary of options used in calculating pressure vs. mole fraction curves.

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        try:
            rhol, flagl = calc.calc_rhol(P, T, xi, eos, rhodict=rhodict)
        except:
            raise ValueError("Calculation of calc_rhol failed for xi:%s, T:%g P:%g" %(str(xi), T, P))
        return rhol


    def objective(self, eos, threads=1):

        """
        Generate objective function value from this dataset

        Parameters
        ----------
        eos : obj
            EOS object with updated parameters

        Returns
        -------
        obj_val : float
            A value for the objective function
        """

        # Generate input_list to distribute to multiple threads.
        input_list = []
        for i,T in enumerate(self.T):
             input_list.append((self.P[i], T, self.xi[i], eos, self._rhodict))

        # Submit data points to be evaluated
        p = Pool(threads)
        phase_list = p.map(self._thermo_wrapper, input_list, 1)
        p.close()
        p.join()

        # Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list) 

        # objective function
        obj_value = np.sum((((phase_list[0] - self.rhol) / self.rhol)**2)*self.weights)

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self.T)}
        return string