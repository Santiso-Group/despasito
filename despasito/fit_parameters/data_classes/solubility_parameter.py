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

        * name : str, data type, in this case SatProps
        * calctype : str, Optional, default: 'sat_props
        * T : list, List of temperature values for calculation
        * P : list, List of pressure values used in calculations
        * xi : list, List of liquid mole fractions used in calculations.
        * weights : list/float, Either a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case SolubilityParam
    calctype : str, Optional, default: 'solubility_parameter'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, only one should be equal to 1.
    P  : list
        List of pressure.
    """

    def __init__(self, data_dict):

        logger = logging.getLogger(__name__)

        # Self interaction parameters
        self.name = data_dict["name"]
        self._thermodict = {}
        if "calctype" in data_dict:
            self._thermodict["calculation_type"] = data_dict["calctype"]
            self.calctype = data_dict["calctype"]
        else:
            self.calctype = "solubility_parameter"
            self._thermodict["calculation_type"] = "solubility_parameter"

        if "xi" in data_dict:
            self._thermodict["xilist"] = data_dict["xi"]
        if "yi" in data_dict:
            self._thermodict["xilist"] = data_dict["yi"]
            logger.info("Vapor mole fraction recorded as 'xi'")
        if "T" in data_dict:
            self._thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            self._thermodict["rhol"] = data_dict["rhol"]
        if "P" in data_dict:
            self._thermodict["Plist"] = data_dict["P"]
        if "delta" in data_dict:
            self._thermodict["delta"] = data_dict["delta"]

        tmp = ["Tlist", "delta"]
        if not all([x in self._thermodict.keys() for x in tmp]):
            raise ImportError("Given solubility data, value(s) for T and delta should have been provided.")

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = 1.0

        logger.info("Data type 'SolubilityParam' initiated with calctype, {}, and data types: {}".format(self.calctype,", ".join(self._thermodict.keys())))

        if 'rhodict' in data_dict:
            self._thermodict["rhodict"] = data_dict["rhodict"]

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

        # Check bead type
        if 'xilist' not in self._thermodict:
            if len(eos._nui) > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self._thermodict['xilist'] = np.array([[1.0] for x in range(len(self._thermodict['Tlist']))])

        # Run thermo calculations
        try:
            output_dict = thermo(eos, self._thermodict)
            output = [output_dict["delta"],output_dict["rhol"]]
        except:
            raise ValueError("Calculation of solubility_parameter failed")

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

        phase_list = self._thermo_wrapper(eos)

        ## Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = 0
        if "delta" in self._thermodict:
            obj_value += np.sum((((phase_list[0] - self._thermodict["delta"]) / self._thermodict["delta"])**2)*self.weights)
        if "rhol" in self._thermodict:
            obj_value += np.sum((((phase_list[1] - self._thermodict["rhol"]) / self._thermodict["rhol"])**2)*self.weights)

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self._thermodict['T'])}
        return string
        
