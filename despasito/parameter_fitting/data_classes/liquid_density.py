r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito import fundamental_constants as constants
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
        Dictionary of exp data of type liquid density.

        * calculation_type : str, Optional, default: 'liquid_properties'
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in liquid_properties calculations
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * objective_method : str, The 'method' keyword in function despasito.parameter_fitting.fit_funcs.obj_function_form.
        * density_opts : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case liquid_density
    weights : dict, Optional, deafault: {"some_property": 1.0 ...}
        Dicitonary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default: liquid_properties
        - density_opts (dict) default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)
        
        self.name = "liquid_density"
        tmp = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        if 'density_opts' in self.thermodict:
            tmp.update(self.thermodict["density_opts"])
        self.thermodict["density_opts"] = tmp
        
        if self.thermodict["calculation_type"] == None:
            self.thermodict["calculation_type"] = "liquid_properties"

        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
            del data_dict["xi"]
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            del data_dict["T"]
        if "P" in data_dict:
            self.thermodict["Plist"] = data_dict["P"]
            del data_dict["P"]

        thermo_keys = ["Plist", "xilist", "Tlist"]
        lx = max([len(self.thermodict[key]) for key in thermo_keys if key in self.thermodict])
        for key in thermo_keys:
            if key in self.thermodict and len(self.thermodict[key]) == 1:
                self.thermodict[key] = np.array([self.thermodict[key][0] for x in range(lx)])

        if "Tlist" not in self.thermodict:
            self.thermodict["Tlist"] = np.ones(lx)*constants.standard_temperature
            logger.info("Assume {}K".format(constants.standard_temperature))
        if "Plist" not in self.thermodict:
            self.thermodict["Plist"] = np.ones(lx)*constants.standard_pressure
            logger.info("Assume atmospheric pressure")
        if 'xilist' not in self.thermodict:
            if self.Eos.number_of_components > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self.thermodict['xilist'] = np.array([[1.0] for x in range(lx)])

        self.result_keys = ["rhol", "phil"]
        for key in self.result_keys:
            if key in data_dict:
                self.thermodict[key] = data_dict[key]
                del data_dict[key]
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
                else:
                    self.weights[key] = 1.0

        if "Tlist" not in self.thermodict and "rhol" not in self.thermodict:
            raise ImportError("Given liquid property data, values for T, xi, and rhol should have been provided.")

        self.npoints = len(self.thermodict["rhol"])
        thermo_keys = ["Plist", "rhol", "xilist", "Tlist", "phil"]
        for key in thermo_keys:
            if key in self.thermodict and len(self.thermodict[key]) != self.npoints:
                raise ValueError("T, P, xi, and rhol are not all the same length.")

        self.thermodict.update(data_dict)

        logger.info("Data type 'liquid_properties' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.thermodict["calculation_type"],", ".join(self.result_keys),self.weights))

    def _thermo_wrapper(self):

        """
        Generate thermodynamic predictions from Eos object

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        # Remove results
        opts = self.thermodict.copy()
        tmp = self.result_keys + ["name", "beadparams0"]
        for key in tmp:
            if key in opts:
                del opts[key]

        try:
            output_dict = thermo(self.Eos, **opts)
            output = [output_dict["rhol"]]
        except:
            raise ValueError("Calculation of calc_liquid_density failed")
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

