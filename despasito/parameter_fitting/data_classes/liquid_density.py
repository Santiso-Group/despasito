r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.parameter_fitting import fit_funcs as ff
from despasito.parameter_fitting.interface import ExpDataTemplate

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
        * objective_method : str, The 'method' keyword in function despasito.parameter_fitting.fit_funcs.obj_function_form.
        * density_dict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case rhol
    weights : dict, Optional, deafault: {"some_property": 1.0 ...}
        Dicitonary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default: liquid_properties
        - density_dict (dict) default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)
        
        tmp = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        if 'density_dict' in self.thermodict:
            tmp.update(self.thermodict["density_dict"])
        self.thermodict["density_dict"] = tmp
        
        if self.thermodict["calculation_type"] == None:
            self.thermodict["calculation_type"] = "liquid_properties"

        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            self.thermodict["rhol"] = data_dict["rhol"]
            if "rhol" in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        if "Tlist" not in self.thermodict and "rhol" not in self.thermodict:
            raise ImportError("Given liquid property data, values for T, xi, and rhol should have been provided.")

        if "P" in data_dict:
            if (type(data_dict["P"]) == float or len(data_dict["P"])==1):
                self.thermodict["Plist"] = np.ones(len(self.thermodict["Tlist"]))*data_dict["P"]
            else:
                self.thermodict["Plist"] = data_dict["P"]
        else:
            self.thermodict["Plist"] = np.ones(len(self.thermodict["Tlist"]))*101325.0
            logger.info("Assume atmospheric pressure")

        self.npoints = len(self.thermodict["Tlist"])
        thermo_keys = ["Plist", "rhol"]
        for key in thermo_keys:
            if key in self.thermodict and len(self.thermodict[key]) != self.npoints:
                raise ValueError("T, P, and rhol are not all the same length.")

        for key in self.thermodict.keys():
            if key not in self.weights:
                if key not in ['calculation_type',"density_dict","mpObj"]:
                    self.weights[key] = 1.0

        logger.info("Data type 'liquid_properties' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.thermodict["calculation_type"],", ".join(self.thermodict.keys()),self.weights))

    def _thermo_wrapper(self):

        """
        Generate thermodynamic predictions from eos object

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        # Check bead type
        if 'xilist' not in self.thermodict:
            if self.eos.number_of_components > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self.thermodict['xilist'] = np.array([[1.0] for x in range(len(self.thermodict['Tlist']))])

        try:
            output_dict = thermo(self.eos, **self.thermodict)
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

        phase_list = self._thermo_wrapper()

        # Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list) 
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = ff.obj_function_form(phase_list, self.thermodict['rhol'], weights=self.weights['rhol'], **self.obj_opts)

        if (np.isnan(obj_value) or obj_value==0.0):
            obj_value = np.inf

        return obj_value

