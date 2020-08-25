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
    calculation_type : str, Optional, default: 'phase_xiT'
        Thermodynamic calculation type, 'phase_yiT' is another acceptable option for this data type
    T : list
        List of temperature values for calculation
    P : list
        List of pressure values for calculation
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
    yi : list
        List of vapor mole fractions, sum(yi) should equal 1
    
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}

        if "weights" in data_dict:
            self.weights = data_dict["weights"]
        else:
            self.weights = {}

        self.obj_opts = {}
        if "objective_method" in data_dict:
            self.obj_opts["method"] = data_dict["objective_method"]
        fitting_opts = ["nan_number", "nan_ratio"]
        for key in fitting_opts:
            if key in data_dict:
                self.obj_opts[key] = data_dict[key]
        logger.info("Objective function options: {}".format(self.obj_opts))

        if "eos_obj" in data_dict:
            self.eos = data_dict["eos_obj"]
        else:
            raise ValueError("An eos object should have been included")

        if "xi" in data_dict: 
            self._thermodict["xilist"] = data_dict["xi"]
            if 'xi' in self.weights:
                self.weights['xilist'] = self.weights.pop('xi')
                key = 'xilist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "T" in data_dict:
            self._thermodict["Tlist"] = data_dict["T"]
            if 'T' in self.weights:
                self.weights['Tlist'] = self.weights.pop('T')
        if "yi" in data_dict:
            self._thermodict["yilist"] = data_dict["yi"]
            if 'yi' in self.weights:
                self.weights['yilist'] = self.weights.pop('yi')
                key = 'yilist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
        if "P" in data_dict: 
            self._thermodict["Plist"] = data_dict["P"]
            self._thermodict["Pguess"] = data_dict["P"]
            if 'P' in self.weights:
                self.weights['Plist'] = self.weights.pop('P')
                key = 'Plist'
                if key in self.weights:
                    if type(self.weights[key]) != float and len(self.weights[key]) != len(self._thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))

        if not any([x in self._thermodict.keys() for x in ['Plist', 'Tlist']]):
            raise ImportError("Given flash data, values for P and T should have been provided.")

        if not all([x in self._thermodict.keys() for x in ['xilist', 'yilist']]):
            raise ImportError("Given flash data, mole fractions should have been provided.")

        if not any(np.array([len(x) for key,x in self._thermodict.items()]) == len(self._thermodict['xilist'])):
            raise ValueError("T, P, yi, and xi are not all the same length.")

        if "calculation_type" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calculation_type"]
            self.calculation_type = data_dict["calculation_type"]
        else:
            logger.warning("No calculation type has been provided.")
            self._thermodict["calculation_type"] = "flash"

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != 'calculation_type':
                    self.weights[key] = 1.0

        logger.info("Data type 'flash' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.calculation_type,", ".join(self._thermodict.keys()),self.weights))

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

        try:
            output_dict = thermo(self.eos, self._thermodict)
            output = [output_dict['xi'],output_dict["yi"]]
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

        if "yilist" in self._thermodict:
            yi = np.transpose(self._thermodict["yilist"])
            obj_value[0] = 0
            print(np.shape(yi), np.shape(phase_list))
            for i in range(len(yi)):
                obj_value[0] += ff.obj_function_form(phase_list[i], yi[i], weights=self.weights['yilist'], **self.obj_opts)

        if "xilist" in self._thermodict:
            xi = np.transpose(self._thermodict["xilist"])
            obj_value[1] = 0
            print(np.shape(xi), np.shape(phase_list))
            for i in range(len(xi)):
                obj_value[1] += ff.obj_function_form(phase_list[ncomp+i], xi[i], weights=self.weights['xilist'], **self.obj_opts)

        logger.debug("Obj. breakdown for {}: xi {}, yi {}".format(self.name,obj_value[0],obj_value[1]))

        if all([(np.isnan(x) or x==0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalculation_type:%s\nNdatapts:%g" % {self.name, self.calculation_type, len(self._thermodict["Tlist"])}
        return string
        
