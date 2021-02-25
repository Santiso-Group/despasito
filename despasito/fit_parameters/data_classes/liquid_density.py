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
        * calctype : str, Optional, default: 'liquid_properties'
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in liquid_properties calculations
        * weights : dict, A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case rhol
    calctype : str, Optional, default: 'liquid_properties'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, sum(xi) should equal 1
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
            self.calctype = "liquid_properties"
            self._thermodict["calculation_type"] = "liquid_properties"

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = {}

        if "xi" in data_dict:
            self._thermodict["xilist"] = data_dict["xi"]
        if "T" in data_dict:
            self._thermodict["Tlist"] = data_dict["T"]
        if "rhol" in data_dict:
            key = "rhol"
            self._thermodict["rhol"] = data_dict["rhol"]
            if key in self.weights:
                if type(self.weights[key]) != float and len(self.weights[key]) != len(
                    self._thermodict[key]
                ):
                    raise ValueError(
                        "Array of weights for '{}' values not equal to number of experimental values given.".format(
                            key
                        )
                    )

        tmp = ["Tlist", "rhol"]
        if not all([x in self._thermodict.keys() for x in tmp]):
            raise ImportError(
                "Given liquid property data, values for T, xi, and rhol should have been provided."
            )

        if "P" in data_dict:
            if type(data_dict["P"]) == float or len(data_dict["P"]) == 1:
                self._thermodict["Plist"] = (
                    np.ones(len(self._thermodict["Tlist"])) * data_dict["P"]
                )
            else:
                self._thermodict["Plist"] = data_dict["P"]
        else:
            self._thermodict["Plist"] = (
                np.ones(len(self._thermodict["Tlist"])) * 101325.0
            )
            logger.info("Assume atmospheric pressure")

        for key in self._thermodict.keys():
            if key not in self.weights:
                if key != "calculation_type":
                    self.weights[key] = 1.0

        logger.info(
            "Data type 'liquid_properties' initiated with calctype, {}, and data types: {}.\nWeight data by: {}".format(
                self.calctype, ", ".join(self._thermodict.keys()), self.weights
            )
        )

        if "rhodict" in data_dict:
            self._thermodict["rhodict"] = data_dict["rhodict"]
        else:
            self._thermodict["rhodict"] = {
                "minrhofrac": (1.0 / 300000.0),
                "rhoinc": 10.0,
                "vspacemax": 1.0e-4,
            }

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
        if "xilist" not in self._thermodict:
            if len(eos._nui) > 1:
                raise ValueError(
                    "Ambiguous instructions. Include xi to define intended component to obtain saturation properties"
                )
            else:
                self._thermodict["xilist"] = np.array(
                    [[1.0] for x in range(len(self._thermodict["Tlist"]))]
                )

        try:
            output_dict = thermo(eos, self._thermodict)
            output = [output_dict["rhol"]]
        except:
            raise ValueError("Calculation of calc_rhol failed")
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

        # Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = np.sum(
            (
                ((phase_list[0] - self._thermodict["rhol"]) / self._thermodict["rhol"])
                ** 2
            )
            * self.weights["rhol"]
        )

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {
            self.name,
            self.calctype,
            len(self._thermodict["Tlist"]),
        }
        return string
