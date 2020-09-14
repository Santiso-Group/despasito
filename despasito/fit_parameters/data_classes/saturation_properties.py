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
#                       Saturation Props                         #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for saturation data. This data is evaluated with "saturation_properties". 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of saturation properties.

        * name : str, data type, in this case SatProps
        * calculation_type : str, Optional, default: 'saturation_properties
        * T : list, List of temperature values for calculation
        * xi : list, (or yi) List of liquid mole fractions used in saturation properties calculations, should be 1 for the molecule of focus and 0 for the rest.
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * objective_method : str, The 'method' keyword in function despasito.fit_parameters.fit_funcs.obj_function_form.
        * density_dict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case SatProps
    weights : dict, Optional, deafault: {"some_property": 1.0 ...}
        Dicitonary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default: saturation_properties
        - density_dict (dict) default: {"minrhofrac":(1.0 / 80000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)
        
        # If required items weren't defined, set defaults
        if self.thermodict["calculation_type"] == None:
            self.thermodict["calculation_type"] = "saturation_properties"

        tmp = {"minrhofrac":(1.0 / 80000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        if 'density_dict' in self.thermodict:
            tmp.update(self.thermodict["density_dict"])
        self.thermodict["density_dict"] = tmp

        # Extract system data
        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
        if "yi" in data_dict:
            self.thermodict["xilist"] = data_dict["yi"]
            logger.info("Vapor mole fraction recorded as 'xi'")
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            key = 'rhol'
            self.thermodict["rhol"] = data_dict["rhol"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "rhov" in data_dict:
            key = "rhov"
            self.thermodict["rhov"] = data_dict["rhov"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "P" in data_dict:
            self.thermodict["Psat"] = data_dict["P"]
            if 'P' in self.weights:
                self.weights['Psat'] = self.weights.pop('P')
        if "Psat" in data_dict:
            self.thermodict["Psat"] = data_dict["Psat"]
            if 'Psat' in self.weights:
                self.weights['Psat'] = self.weights.pop('Psat')

        key = "Psat"
        if key in self.weights:
            if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        tmp = ["Tlist"]
        if not all([x in self.thermodict.keys() for x in tmp]):
            raise ImportError("Given saturation data, value(s) for T should have been provided.")

        tmp = ["Psat","rhol","rhov"]
        if not any([x in self.thermodict.keys() for x in tmp]):
            raise ImportError("Given saturation data, values for Psat, rhol, and/or rhov should have been provided.")

        for key in self.thermodict:
            if key not in self.weights:
                if key not in ['calculation_type',"density_dict","mpObj"]:
                    self.weights[key] = 1.0

        self.npoints = len(self.thermodict["Tlist"])

        logger.info("Data type 'saturation_properties' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.thermodict["calculation_type"],", ".join(self.thermodict.keys()),self.weights))

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
            if len(self.eos.nui) > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self.thermodict['xilist'] = np.array([[1.0] for x in range(len(self.thermodict['Tlist']))])
 
        # Run thermo calculations
        try:
            output_dict = thermo(self.eos, self.thermodict)
            output = [output_dict["Psat"],output_dict["rhol"],output_dict["rhov"]]
        except:
            raise ValueError("Calculation of calc_Psat failed")

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

        ## Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = np.zeros(3)
        if "Psat" in self.thermodict:
            obj_value[0] = ff.obj_function_form(phase_list[0], self.thermodict['Psat'], weights=self.weights['Psat'], **self.obj_opts)
        if "rhol" in self.thermodict:
            obj_value[1] = ff.obj_function_form(phase_list[1], self.thermodict['rhol'], weights=self.weights['rhol'], **self.obj_opts)
        if "rhov" in self.thermodict:
            obj_value[2] = ff.obj_function_form(phase_list[2], self.thermodict['rhov'], weights=self.weights['rhov'], **self.obj_opts)

        logger.debug("Obj. breakdown for {}: Psat {}, rhol {}, rhov {}".format(self.name,obj_value[0],obj_value[1],obj_value[2]))

        if all([(np.isnan(x) or x==0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

        
