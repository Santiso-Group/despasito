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
    Object for Temperature dependent VLE data. 

    This data could be evaluated with phase_xiT or phase_yiT. Most entries in the exp. dictionary are converted to attributes. 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of type TLVE.

        * name : str, data type, in this case TLVE
        * calculation_type : str, Optional, default: 'phase_xiT', 'phase_yiT' is also acceptable
        * T : list, List of temperature values for calculation
        * xi(yi) : list, List of liquid (or vapor) mole fractions used in phase_xiT (or phase_yiT) calculation.
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
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
    yi : list
        List of vapor mole fractions, sum(yi) should equal 1
    
    """

    def __init__(self, data_dict):

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}

        if "eos_obj" in data_dict:
            self.eos = data_dict["eos_obj"]
        else:
            raise ValueError("An eos object should have been included")

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = {}

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
            raise ImportError("Given TLVE data, values for P and T should have been provided.")

        if not all([x in self._thermodict.keys() for x in ['xilist', 'yilist']]):
            raise ImportError("Given TLVE data, mole fractions should have been provided.")

        if not any(np.array([len(x) for key,x in self._thermodict.items()]) == len(self._thermodict['xilist'])):
            raise ValueError("T, P, yi, and xi are not all the same length.")

        if "calculation_type" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calculation_type"]
            self.calculation_type = data_dict["calculation_type"]
        else:
            logger.warning("No calculation type has been provided.")
            if self.xi:
                self.calculation_type = "phase_xiT"
                self._thermodict["calculation_type"] = "phasexiT"
                logger.warning("Assume a calculation type of phase_xiT")
            elif self.yi:
                self.calculation_type = "phase_yiT"
                self._thermodict["calculation_type"] = "phaseyiT"
                logger.warning("Assume a calculation type of phase_yiT")
            else:
                raise ValueError("Unknown calculation instructions")

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != 'calculation_type':
                    self.weights[key] = 1.0

        logger.info("Data type 'TLVE' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.calculation_type,", ".join(self._thermodict.keys()),self.weights))

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

        if self.calculation_type == "phase_xiT":
            try:
                output_dict = thermo(self.eos, self._thermodict)
                output = [output_dict['P'],output_dict["yi"]]
            except:
                raise ValueError("Calculation of calc_xT_phase failed")

        elif self.calculation_type == "phase_yiT":
            try:
                output_dict = thermo(self.eos, self._thermodict)
                output = [output_dict['P'],output_dict["xi"]]
            except:
                raise ValueError("Calculation of calc_yT_phase failed")

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
   
        obj_value = np.zeros(2)

        if "Plist" in self._thermodict:
            obj_value[0] = np.nansum((((phase_list[0] - self._thermodict["Plist"]) / self._thermodict["Plist"])**2)*self.weights['Plist'])

        if self.calculation_type == "phase_xiT":
            if "yilist" in self._thermodict:
                yi = np.transpose(self._thermodict["yilist"])
                obj_value[1] = np.nansum((((phase_list[1:] - yi)/yi)**2)*self.weights['yilist'])
        elif self.calculation_type == "phase_yiT":
            if "xilist" in self._thermodict:
                xi = np.transpose(self._thermodict["xilist"])
                obj_value[1] = np.nansum((((phase_list[1:] - xi)/xi)**2)*self.weights['xilist'])

        logger.debug("Obj. breakdown for {}: P {}, zi {}".format(self.name,obj_value[0],obj_value[1]))

        if all(np.isnan(obj_value)):
            obj_total = np.nan
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalculation_type:%s\nNdatapts:%g" % {self.name, self.calculation_type, len(self._thermodict["Tlist"])}
        return string
        
