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
    Object for saturation data. This data is evaluated with "sat_props". 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of saturation properties.

        * name : str, data type, in this case SatProps
        * calctype : str, Optional, default: 'sat_props
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in saturation properties calculations, should be 1 for the molecule of focus and 0 for the rest.
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case SatProps
    calctype : str, Optional, default: 'sat_props'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, only one should be equal to 1.
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}
        if "calctype" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calctype"]
            self.calctype = data_dict["calctype"]
        else:
            self.calctype = "sat_props"
            self._thermodict["calculation_type"] = "sat_props"

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
        if "yi" in data_dict:
            self._thermodict["xilist"] = data_dict["yi"]
            logger.info("Vapor mole fraction recorded as 'xi'")
        if "T" in data_dict:
            self._thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            key = 'rhol'
            self._thermodict["rhol"] = data_dict["rhol"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "rhov" in data_dict:
            key = "rhov"
            self._thermodict["rhov"] = data_dict["rhov"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                    raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "P" in data_dict:
            self._thermodict["Psat"] = data_dict["P"]
            if 'P' in self.weights:
                self.weights['Psat'] = self.weights.pop('P')
        if "Psat" in data_dict:
            self._thermodict["Psat"] = data_dict["Psat"]

        key = "Psat"
        if key in self.weights:
            if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        tmp = ["Tlist"]
        if not all([x in self._thermodict.keys() for x in tmp]):
            raise ImportError("Given saturation data, value(s) for T should have been provided.")

        tmp = ["Psat","rhol","rhov"]
        if not any([x in self._thermodict.keys() for x in tmp]):
            raise ImportError("Given saturation data, values for Psat, rhol, and/or rhov should have been provided.")

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != 'calculation_type':
                    self.weights[key] = 1.0

        logger.info("Data type 'sat_props' initiated with calctype, {}, and data types: {}.\nWeight data by: {}".format(self.calctype,", ".join(self._thermodict.keys()),self.weights))

        if 'rhodict' in data_dict:
            self._thermodict["rhodict"] = data_dict["rhodict"]
        else:
            self._thermodict["rhodict"] = {"minrhofrac":(1.0 / 80000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

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
            if len(self.eos.eos_dict['nui']) > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self._thermodict['xilist'] = np.array([[1.0] for x in range(len(self._thermodict['Tlist']))])
 
        # Run thermo calculations
        try:
            output_dict = thermo(self.eos, self._thermodict)
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
        if "Psat" in self._thermodict:
            obj_value[0] = np.nansum((((phase_list[0] - self._thermodict['Psat']) / self._thermodict['Psat'])**2)*self.weights['Psat'])
        if "rhol" in self._thermodict:
            obj_value[1] = np.nansum((((phase_list[1] - self._thermodict['rhol']) / self._thermodict['rhol'])**2)*self.weights['rhol'])
        if "rhov" in self._thermodict:
            obj_value[2] = np.nansum((((phase_list[2] - self._thermodict['rhov']) / self._thermodict['rhov'])**2)*self.weights['rhov'])

        logger.debug("Obj. breakdown for {}: Psat {}, rhol {}, rhov {}".format(self.name,obj_value[0],obj_value[1],obj_value[2]))

        if all(np.isnan(obj_value)):
            obj_total = np.nan
        else:
            obj_total = np.nansum(obj_value)
      
        return obj_total

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self._thermodict['T'])}
        return string
        
