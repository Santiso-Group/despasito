r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.fit_parameters import fit_funcs as ff
from despasito.fit_parameters.interface import ExpDataTemplate

logger = logging.getLogger(__name__)

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
        * calculation_type : str, Optional, default: 'liquid_properties'
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in liquid_properties calculations
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * density_dict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case rhol
    calculation_type : str, Optional, default: 'liquid_properties'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}
        if "calculation_type" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calculation_type"]
            self.calculation_type = data_dict["calculation_type"]
        else:
            self.calculation_type = "liquid_properties"
            self._thermodict["calculation_type"] = "liquid_properties"

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = {}

        if "eos_obj" in data_dict:
            self.eos = data_dict["eos_obj"]
        else:
            raise ValueError("An eos object should have been included")

        if "xi" in data_dict:
            self._thermodict["xilist"] = data_dict["xi"]
        if "T" in data_dict:
            self._thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            key = "rhol"
            self._thermodict["rhol"] = data_dict["rhol"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        tmp = ["Tlist","rhol"]
        if not all([x in self._thermodict.keys() for x in tmp]):
            raise ImportError("Given liquid property data, values for T, xi, and rhol should have been provided.")

        if "P" in data_dict:
            if (type(data_dict["P"]) == float or len(data_dict["P"])==1):
                self._thermodict["Plist"] = np.ones(len(self._thermodict["Tlist"]))*data_dict["P"]
            else:
                self._thermodict["Plist"] = data_dict["P"]
        else:
            self._thermodict["Plist"] = np.ones(len(self._thermodict["Tlist"]))*101325.0
            logger.info("Assume atmospheric pressure")

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != 'calculation_type':
                    self.weights[key] = 1.0

        logger.info("Data type 'liquid_properties' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.calculation_type,", ".join(self._thermodict.keys()),self.weights))

        if 'density_dict' in data_dict:
            self._thermodict["density_dict"] = data_dict["density_dict"]
        else:
            self._thermodict["density_dict"] = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

        if "mpObj" in data_dict:
            self._thermodict["mpObj"] = data_dict["mpObj"]

    def _thermo_wrapper(self):

        """
        Generate thermodynamic predictions from eos object

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        # Check bead type
        if 'xilist' not in self._thermodict:
            if len(self.eos.nui) > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self._thermodict['xilist'] = np.array([[1.0] for x in range(len(self._thermodict['Tlist']))])

        try:
            output_dict = thermo(self.eos, self._thermodict)
            output = [output_dict["rhol"]]
        except:
            raise ValueError("Calculation of calc_rhol failed")
        return output


    def objective(self):

        """
        Generate objective function value from this dataset

        Returns
        -------
        obj_val : float
            A value for the objective function
        """

        #phase_list = self._thermo_wrapper()

        ## Reformat array of results
        #phase_list, len_list = ff.reformat_ouput(phase_list) 
        #phase_list = np.transpose(np.array(phase_list))

        ## objective function
        #obj_value = np.nansum(((np.abs(phase_list[0] - self._thermodict["rhol"]) / self._thermodict["rhol"])**2)*self.weights['rhol'])

        rand = np.random.randint(2)
        if rand == 1:
            obj_value = np.inf
        else:
            obj_value = 1
        logger.info("Objective: {}".format(obj_value))

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalculation_type:%s\nNdatapts:%g" % {self.name, self.calculation_type, len(self._thermodict["Tlist"])}
        return string
