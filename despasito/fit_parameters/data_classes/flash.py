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
#                              TLVE                              #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for flash calculation. 

    This data could be evaluated with flash. Most entries in the exp. dictionary are converted to attributes. 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of type TLVE.

        * name : str, data type, in this case TLVE
        * calculation_type : str, Optional, default: 'phase_xiT', 'phase_yiT' is also acceptable
        * T : list, List of temperature values for calculation
        * P : list, List of pressure values for calculation
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * density_dict : dict, Optional, default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case TLVE
    weights : dict, Optional, deafault: {"some_property": 1.0 ...}
        Dicitonary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default: flash
        - density_dict (dict) default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
    
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)

        if self.thermodict["calculation_type"] == None:
            logger.warning("No calculation type has been provided.")
            self.thermodict["calculation_type"] = "flash"
    
        tmp = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
        if 'density_dict' in self.thermodict:
            tmp.update(self.thermodict["density_dict"])
        self.thermodict["density_dict"] = tmp
        
        if "xi" in data_dict: 
            self.thermodict["xilist"] = data_dict["xi"]
            if 'xi' in self.weights:
                self.weights['xilist'] = self.weights.pop('xi')
                key = 'xilist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            if 'T' in self.weights:
                self.weights['Tlist'] = self.weights.pop('T')
        if "yi" in data_dict:
            self.thermodict["yilist"] = data_dict["yi"]
            if 'yi' in self.weights:
                self.weights['yilist'] = self.weights.pop('yi')
                key = 'yilist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "P" in data_dict: 
            self.thermodict["Plist"] = data_dict["P"]
            self.thermodict["Pguess"] = data_dict["P"]
            if 'P' in self.weights:
                self.weights['Plist'] = self.weights.pop('P')
                key = 'Plist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self.thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        if 'Plist' not in self.thermodict and 'Tlist' not in self.thermodict:
            raise ImportError("Given flash data, values for P and T should have been provided.")

        if 'xilist' not in self.thermodict or 'yilist' not in self.thermodict:
            raise ImportError("Given flash data, mole fractions should have been provided.")

        self.npoints = len(self.thermodict["Tlist"])
        thermo_keys = ["Plist", 'xilist', 'yilist']
        for key in thermo_keys:
            if key in self.thermodict and len(self.thermodict[key]) != self.npoints:
                raise ValueError("T, P, yi, and xi are not all the same length.")

        for key in self.thermodict.keys():
            if key not in self.weights:
                if key not in ['calculation_type',"density_dict","mpObj"]:
                    self.weights[key] = 1.0

        logger.info("Data type 'flash' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.thermodict["calculation_type"],", ".join(self.thermodict.keys()),self.weights))

    def _thermo_wrapper(self):

        """
        Generate thermodynamic predictions from eos object

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        try:
            output_dict = thermo(self.eos, self.thermodict)
            output = [output_dict["yi"],output_dict['xi']]
        except:
            raise ValueError("Calculation of flash failed")

        return output

    def objective(self):

        """
        Generate objective function value from this dataset

        Returns
        -------
        obj_val : float
            A value for the objective function
        """

        # objective function
        phase_list = self._thermo_wrapper()
        phase_list, len_cluster = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        ncomp = np.shape(self.eos.nui)[0]
   
        obj_value = np.zeros(2)

        if "yilist" in self.thermodict:
            yi = np.transpose(self.thermodict["yilist"])
            obj_value[0] = 0
            for i in range(len(yi)):
                obj_value[0] += ff.obj_function_form(phase_list[i], yi[i], weights=self.weights['yilist'], **self.obj_opts)

        if "xilist" in self.thermodict:
            xi = np.transpose(self.thermodict["xilist"])
            obj_value[1] = 0
            for i in range(len(xi)):
                obj_value[1] += ff.obj_function_form(phase_list[ncomp+i], xi[i], weights=self.weights['xilist'], **self.obj_opts)

        logger.debug("Obj. breakdown for {}: xi {}, yi {}".format(self.name,obj_value[0],obj_value[1]))

        if all([(np.isnan(x) or x==0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

