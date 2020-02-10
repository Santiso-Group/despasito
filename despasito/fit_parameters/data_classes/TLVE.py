r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.fit_parameters import fit_funcs as ff
from despasito.fit_parameters.interface import ExpDataTemplate

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
        * calctype : str, Optional, default: 'phase_xiT', 'phase_yiT' is also acceptable
        * T : list, List of temperature values for calculation
        * xi(yi) : list, List of liquid (or vapor) mole fractions used in phase_xiT (or phase_yiT) calculation.
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case TLVE
    calctype : str, Optional, default: 'phase_xiT'
        Thermodynamic calculation type, 'phase_yiT' is another acceptable option for this data type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
    yi : list
        List of vapor mole fractions, sum(yi) should equal 1
    
    """

    def __init__(self, data_dict):

        logger = logging.getLogger(__name__)

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}

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

        if "calctype" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calctype"]
            self.calctype = data_dict["calctype"]
        else:
            logger.warning("No calculation type has been provided.")
            if self.xi:
                self.calctype = "phase_xiT"
                self._thermodict["calculation_type"] = "phasexiT"
                logger.warning("Assume a calculation type of phase_xiT")
            elif self.yi:
                self.calctype = "phase_yiT"
                self._thermodict["calculation_type"] = "phaseyiT"
                logger.warning("Assume a calculation type of phase_yiT")
            else:
                raise ValueError("Unknown calculation instructions")

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != 'calculation_type':
                    self.weights[key] = 1.0

        logger.info("Data type 'TLVE' initiated with calctype, {}, and data types: {}.\nWeight data by: {}".format(self.calctype,", ".join(self._thermodict.keys()),self.weights))

        if 'rhodict' in data_dict:
            self._thermodict["rhodict"] = data_dict["rhodict"]
        else:
            self._thermodict["rhodict"] = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

    def _thermo_wrapper(self, eos):

        """
        Generate thermodynamic predictions from eos object

        Parameters
        ----------
        eos : obj
            EOS object with updated parameters

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        if self.calctype == "phase_xiT":
            try:
                output_dict = thermo(eos, self._thermodict)
                output = [output_dict['P'],output_dict["yi"]]
            except:
                raise ValueError("Calculation of calc_xT_phase failed")

        elif self.calctype == "phase_yiT":
            try:
                output_dict = thermo(eos, self._thermodict)
                output = [output_dict['P'],output_dict["xi"]]
            except:
                raise ValueError("Calculation of calc_yT_phase failed")

        return output

    def objective(self, eos):

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

        logger = logging.getLogger(__name__)

        # objective function
        phase_list = self._thermo_wrapper(eos)
        phase_list, len_cluster = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))
   
        obj_value = np.zeros(2)

        if "Plist" in self._thermodict:
            obj_value[0] = np.sum((((phase_list[0] - self._thermodict["Plist"]) / self._thermodict["Plist"])**2)*self.weights['Plist'])

        if self.calctype == "phase_xiT":
            if "yilist" in self._thermodict:
                yi = np.transpose(self._thermodict["yilist"])
                obj_value[1] = np.sum((((phase_list[1:] - yi)/yi)**2)*self.weights['yilist'])
        elif self.calctype == "phase_yiT":
            if "xilist" in self._thermodict:
                xi = np.transpose(self._thermodict["xilist"])
                obj_value[1] = np.sum((((phase_list[1:] - xi)/xi)**2)*self.weights['xilist'])

        logger.debug("Obj. breakdown for {}: P {}, zi {}".format(self.name,obj_value[0],obj_value[1]))

        if all(np.isnan(obj_value)):
            obj_total = np.nan
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self._thermodict["Tlist"])}
        return string
        
