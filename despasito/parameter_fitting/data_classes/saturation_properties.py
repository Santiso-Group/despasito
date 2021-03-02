r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito import fundamental_constants as constants
from despasito.thermodynamics import thermo
from despasito.parameter_fitting import fit_functions as ff
from despasito.parameter_fitting.interface import ExpDataTemplate
from despasito import fundamental_constants as constants
import despasito.utils.general_toolbox as gtb

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

        * calculation_type (str) - Optional, default='saturation_properties
        * eos_obj (obj) - Equation of state object
        * T (list) - List of temperature values for calculation
        * xi (list) - (or yi) List of liquid mole fractions used in saturation properties calculations, should be 1 for the molecule of focus and 0 for the rest.
        * weights (dict) - A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * objective_method (str) - The 'method' keyword in function despasito.parameter_fitting.fit_functions.obj_function_form.
        * density_opts (dict) - Optional, default={"min_density_fraction":(1.0 / 60000.0), "density_increment":10.0, "max_volume_increment":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.
        * kwargs for :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`

    Attributes
    ----------
    name : str
        Data type, in this case saturation_properties
    Eos : obj
        Equation of state object
    weights : dict, Optional, default: {"some_property": 1.0 ...}
        Dictionary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    obj_opts : dict
        Keywords to compute the objective function with :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`.
    npoints : int
        Number of sets of system conditions this object computes
    result_keys : list
        Thermodynamic property names used in calculation of objective function. In in this case: ["rhol", "rhov", "Psat"]
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
        
        - calculation_type (str) default=saturation_properties
        - density_opts (dict) default={"min_density_fraction":(1.0 / 80000.0), "density_increment":10.0, "max_volume_increment":1.0E-4}
        
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)

        # If required items weren't defined, set defaults
        self.name = "saturation_properties"
        if self.thermodict["calculation_type"] == None:
            self.thermodict["calculation_type"] = "saturation_properties"

        tmp = {
            "min_density_fraction": (1.0 / 80000.0),
            "density_increment": 10.0,
            "max_volume_increment": 1.0e-4,
        }
        if "density_opts" in self.thermodict:
            tmp.update(self.thermodict["density_opts"])
        self.thermodict["density_opts"] = tmp

        # Extract system data
        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
            del data_dict["xi"]
        if "yi" in data_dict:
            self.thermodict["xilist"] = data_dict["yi"]
            logger.info("Vapor mole fraction recorded as 'xi'")
            del data_dict["yi"]
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            del data_dict["T"]
        if "P" in data_dict:
            self.thermodict["Psat"] = data_dict["P"]
            del data_dict["P"]
            if "P" in self.weights:
                if gtb.isiterable(self.weights["P"]) and len(self.weights["P"]) != len(
                    self.thermodict["Psat"]
                ):
                    raise ValueError(
                        "Array of weights for '{}' values not equal to number of experimental values given.".format(
                            "P"
                        )
                    )
                else:
                    self.weights["Psat"] = self.weights.pop("P")

        self.thermodict.update(data_dict)

        thermo_keys = ["xilist", "Tlist"]
        self.result_keys = ["rhol", "rhov", "Psat"]

        key_list = list(set(thermo_keys + self.result_keys))
        self.thermodict.update(gtb.check_length_dict(self.thermodict, key_list))
        for key in self.result_keys:
            if key in self.thermodict:
                self.npoints = np.size(self.thermodict[key])
                break

        if "xilist" not in self.thermodict and self.Eos.number_of_components > 1:
            raise ValueError(
                "Ambiguous instructions. Include xi to define intended component to obtain saturation properties"
            )
        thermo_defaults = [
            np.array([[1.0] for x in range(self.npoints)]),
            constants.standard_temperature,
        ]
        self.thermodict.update(
            gtb.set_defaults(
                self.thermodict, thermo_keys, thermo_defaults, lx=self.npoints
            )
        )

        self.weights.update(
            gtb.check_length_dict(self.weights, self.result_keys, lx=self.npoints)
        )
        self.weights.update(gtb.set_defaults(self.weights, self.result_keys, 1.0))

        if "Tlist" not in self.thermodict:
            raise ImportError(
                "Given saturation data, value(s) for T should have been provided."
            )

        tmp = ["Psat", "rhol", "rhov"]
        if not any([x in self.thermodict for x in tmp]):
            raise ImportError(
                "Given saturation data, values for Psat, rhol, and/or rhov should have been provided."
            )

        logger.info(
            "Data type 'saturation_properties' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(
                self.thermodict["calculation_type"],
                ", ".join(self.result_keys),
                self.weights,
            )
        )

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
        tmp = self.result_keys + ["name", "parameters_guess"]
        for key in tmp:
            if key in opts:
                del opts[key]

        # Run thermo calculations
        try:
            output_dict = thermo(self.Eos, **opts)
            output = [output_dict["Psat"], output_dict["rhol"], output_dict["rhov"]]
        except Exception:
            raise ValueError("Calculation of calc_saturation_properties failed")

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
        obj_value = np.zeros(3)
        if "Psat" in self.thermodict:
            obj_value[0] = ff.obj_function_form(
                phase_list[0],
                self.thermodict["Psat"],
                weights=self.weights["Psat"],
                **self.obj_opts
            )
        if "rhol" in self.thermodict:
            obj_value[1] = ff.obj_function_form(
                phase_list[1],
                self.thermodict["rhol"],
                weights=self.weights["rhol"],
                **self.obj_opts
            )
        if "rhov" in self.thermodict:
            obj_value[2] = ff.obj_function_form(
                phase_list[2],
                self.thermodict["rhov"],
                weights=self.weights["rhov"],
                **self.obj_opts
            )

        logger.debug(
            "Obj. breakdown for {}: Psat {}, rhol {}, rhov {}".format(
                self.name, obj_value[0], obj_value[1], obj_value[2]
            )
        )

        if all([(np.isnan(x) or x == 0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total
