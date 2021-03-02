r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.parameter_fitting import fit_functions as ff
from despasito.parameter_fitting.interface import ExpDataTemplate
import despasito.utils.general_toolbox as gtb
from despasito import fundamental_constants as constants

logger = logging.getLogger(__name__)

##################################################################
#                                                                #
#                              TLVE                              #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for Temperature dependent VLE data. 

    This data could be evaluated with bubble_pressure or dew_pressure. Most entries in the exp. dictionary are converted to attributes. 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of TLVE temperature dependent liquid vapor equilibria

        * calculation_type (str) - Optional, default='bubble_pressure', 'dew_pressure' is also acceptable
        * eos_obj (obj) - Equation of state object
        * T (list) - List of temperature values for calculation
        * xi(yi) (list) - List of liquid (or vapor) mole fractions used in bubble_pressure (or dew_pressure) calculation.
        * weights (dict) - A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * density_opts (dict) - Optional, default={}, Dictionary of options used in calculating pressure vs. mole fraction curves.
        * kwargs for :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`

    Attributes
    ----------
    name : str
        Data type, in this case TLVE
    Eos : obj
        Equation of state object
    weights : dict, Optional, default: {"some_property": 1.0 ...}
        Dictionary corresponding to thermodict, with weighting factor or vector for each system property used in fitting
    obj_opts : dict
        Keywords to compute the objective function with :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`.
    npoints : int
        Number of sets of system conditions this object computes
    result_keys : list
        Thermodynamic property names used in calculation of objective function. In in this case: ["Plist", 'xilist'] or ["Plist", 'yilist']
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations
    
        - calculation_type (str) default=phasexiT or phaseyiT
        - density_opts (dict) default={"min_density_fraction":(1.0 / 300000.0), "density_increment":10.0, "max_volume_increment":1.0E-4}
  
    """

    def __init__(self, data_dict):

        super().__init__(data_dict)

        self.name = "TLVE"
        self.thermodict["density_opts"] = {}
        if "density_opts" in self.thermodict:
            self.thermodict["density_opts"].update(self.thermodict["density_opts"])

        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            del data_dict["T"]
        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
            del data_dict["xi"]
            if "xi" in self.weights:
                self.weights["xilist"] = self.weights.pop("xi")
        if "yi" in data_dict:
            self.thermodict["yilist"] = data_dict["yi"]
            del data_dict["yi"]
            if "yi" in self.weights:
                self.weights["yilist"] = self.weights.pop("yi")
        if "P" in data_dict:
            self.thermodict["Plist"] = data_dict["P"]
            self.thermodict["Pguess"] = data_dict["P"]
            del data_dict["P"]
            if "P" in self.weights:
                self.weights["Plist"] = self.weights.pop("P")

        self.thermodict.update(data_dict)

        thermo_keys = ["Plist", "xilist", "Tlist"]
        self.result_keys = ["Plist", "xilist", "yilist"]

        key_list = list(set(thermo_keys + self.result_keys))
        self.thermodict.update(gtb.check_length_dict(self.thermodict, key_list))
        self.npoints = np.size(self.thermodict["Tlist"])

        self.thermodict.update(
            gtb.set_defaults(
                self.thermodict,
                "Tlist",
                constants.standard_temperature,
                lx=self.npoints,
            )
        )

        self.weights.update(
            gtb.check_length_dict(self.weights, self.result_keys, lx=self.npoints)
        )
        self.weights.update(gtb.set_defaults(self.weights, self.result_keys, 1.0))

        if "Tlist" not in self.thermodict:
            raise ImportError(
                "Given TLVE data, values for T should have been provided."
            )

        thermo_keys = ["xilist", "yilist", "Plist"]
        if not any([key in self.thermodict for key in thermo_keys]):
            raise ImportError(
                "Given TLVE data, mole fractions and/or pressure should have been provided."
            )

        if self.thermodict["calculation_type"] == None:
            if self.thermodict["xilist"]:
                self.thermodict["calculation_type"] = "bubble_pressure"
                logger.warning(
                    "No calculation type has been provided. Assume a calculation type of bubble_pressure"
                )
            elif self.thermodict["yilist"]:
                self.thermodict["calculation_type"] = "dew_pressure"
                logger.warning(
                    "No calculation type has been provided. Assume a calculation type of dew_pressure"
                )
            else:
                raise ValueError("Unknown calculation instructions")

        if self.thermodict["calculation_type"] == "bubble_pressure":
            self.result_keys.remove("xilist")
            del self.weights["xilist"]
        elif self.thermodict["calculation_type"] == "dew_pressure":
            self.result_keys.remove("yilist")
            del self.weights["yilist"]

        logger.info(
            "Data type 'TLVE' initiated with calculation_type, {}, and data types: {}.\nWeight data by: {}".format(
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

        if self.thermodict["calculation_type"] == "bubble_pressure":
            try:
                output_dict = thermo(self.Eos, **opts)
                output = [output_dict["P"], output_dict["yi"]]
            except Exception:
                raise ValueError("Calculation of calc_bubble_pressure failed")

        elif self.thermodict["calculation_type"] == "dew_pressure":
            try:
                output_dict = thermo(self.Eos, **opts)
                output = [output_dict["P"], output_dict["xi"]]
            except Exception:
                raise ValueError("Calculation of calc_dew_pressure failed")

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
        phase_list, len_cluster = ff.reformat_output(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        obj_value = np.zeros(2)

        if "Plist" in self.thermodict:
            obj_value[0] = ff.obj_function_form(
                phase_list[0],
                self.thermodict["Plist"],
                weights=self.weights["Plist"],
                **self.obj_opts
            )

        if self.thermodict["calculation_type"] == "bubble_pressure":
            if "yilist" in self.thermodict:
                yi = np.transpose(self.thermodict["yilist"])
                obj_value[1] = 0
                for i in range(len(yi)):
                    obj_value[1] += ff.obj_function_form(
                        phase_list[1 + i],
                        yi[i],
                        weights=self.weights["yilist"],
                        **self.obj_opts
                    )
        elif self.thermodict["calculation_type"] == "dew_pressure":
            if "xilist" in self.thermodict:
                xi = np.transpose(self.thermodict["xilist"])
                obj_value[1] = 0
                for i in range(len(xi)):
                    obj_value[1] += ff.obj_function_form(
                        phase_list[1 + i],
                        xi[i],
                        weights=self.weights["xilist"],
                        **self.obj_opts
                    )

        logger.debug(
            "Obj. breakdown for {}: P {}, zi {}".format(
                self.name, obj_value[0], obj_value[1]
            )
        )

        if all([(np.isnan(x) or x == 0.0) for x in obj_value]):
            obj_total = np.inf
        else:
            obj_total = np.nansum(obj_value)

        return obj_total
