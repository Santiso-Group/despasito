r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.parameter_fitting import fit_funcs as ff
from despasito.parameter_fitting.interface import ExpDataTemplate
from despasito.utils.parallelization import MultiprocessingJob


logger = logging.getLogger(__name__)

##################################################################
#                                                                #
#                       Saturation Props                         #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for hildebrand solubility parameters. This data is evaluated with "solubility_parameter". 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of saturation properties.

        * calculation_type : str, Optional, default: 'solubility_parameter'
        * T : list, List of temperature values for calculation
        * P : list, List of pressure values used in calculations
        * xi : list, List of liquid mole fractions used in calculations.
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * density_opts : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case solubility_parameter
    weights : dict, Optional, deafault: {"some_property": 1.0 ...}
        Dicitonary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default: solubility_parameter
        - density_opts (dict) default: {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)
        
        self.name = "solubility_parameter"
        if self.thermodict["calculation_type"] == None:
            self.thermodict["calculation_type"] = "solubility_parameter"
        
        if 'density_opts' not in self.thermodict:
            self.thermodict["density_opts"] = {}

        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
            del data_dict["xi"]
        if "yi" in data_dict:
            self.thermodict["xilist"] = data_dict["yi"]
            del data_dict["yi"]
            logger.info("Vapor mole fraction recorded as 'xi'")
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            del data_dict["T"]
        if "P" in data_dict:
            self.thermodict["Plist"] = data_dict["P"]
            del data_dict["P"]
            if 'P' in self.weights:
                self.weights['Plist'] = self.weights.pop('P')

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

        self.result_keys = ["rhol", "delta"]
        for key in self.result_keys:
            if key in data_dict:
                self.thermodict[key] = data_dict[key]
                del data_dict[key]
                if key in self.weights:
                    if isinstance(self.weights[key],float) and len(self.weights[key]) != len(self.thermodict[key]):
                        raise ValueError("Array of weights for '{}' values not equal to number of experimental values given.".format(key))
                else:
                    self.weights[key] = 1.0

        if "Tlist" not in self.thermodict and "delta" not in self.thermodict:
            raise ImportError("Given solubility data, value(s) for T and delta should have been provided.")

        self.npoints = len(self.thermodict["delta"])
        thermo_keys = ["Plist", "rhol", "Tlist", "xilist", "yilist"]
        for key in thermo_keys:
            if key in self.thermodict and len(self.thermodict[key]) != self.npoints:
                raise ValueError("T, P, and rhol are not all the same length.")

        self.thermodict.update(data_dict)

        logger.info("Data type 'solubility parameter' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(self.thermodict["calculation_type"],", ".join(self.result_keys),self.weights))

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

        # Run thermo calculations
        try:
            output_dict = thermo(self.Eos, **opts)
            output = [output_dict["delta"],output_dict["rhol"]]
        except Exception:
            raise ValueError("Calculation of solubility_parameter failed")

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
        phase_list, len_list = ff.reformat_output(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = np.zeros(2)
        if "delta" in self.thermodict:
            obj_value[0] = ff.obj_function_form(phase_list[0], self.thermodict['delta'], weights=self.weights['delta'], **self.obj_opts)
        if "rhol" in self.thermodict:
            obj_value[1] = ff.obj_function_form(phase_list[1], self.thermodict['rhol'], weights=self.weights['rhol'], **self.obj_opts)

        logger.debug("Obj. breakdown for {}: delta {}, rhol {}".format(self.name,obj_value[0],obj_value[1]))

        if all([(np.isnan(x) or x==0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total

        
