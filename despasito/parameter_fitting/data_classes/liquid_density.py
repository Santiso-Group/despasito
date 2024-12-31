r"""
Objects for storing and producing objective values for comparing experimental data
to EOS predictions.
"""

import numpy as np
import logging

from despasito.thermodynamics import thermo
from despasito.parameter_fitting import fit_functions as ff
from despasito.parameter_fitting.interface import ExpDataTemplate
import despasito.fundamental_constants as constants
import despasito.utils.general_toolbox as gtb

logger = logging.getLogger(__name__)


##################################################################
#                                                                #
#                       Liquid Density                           #
#                                                                #
##################################################################
class Data(ExpDataTemplate):
    r"""
    Object for liquid density data. This data is evaluated with "liquid_properties".

    This object is initiated in :func:`~despasito.parameter_fitting.fit` with the
    keyword, ``exp_data[*]["data_class_type"]="liquid_density"``.
    The data could be evaluated with
    :func:`~despasito.thermodynamics.calculation_types.liquid_properties`

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of type liquid density.

        * calculation_type (str) - Optional, default='liquid_properties'
        * MultiprocessingObject (obj) - Optional, Initiated
        :class:`~despasito.utils.parallelization.MultiprocessingJob`
        * eos_obj (obj) - Equation of state object
        * T (list) - [K] List of temperature values for calculation
        * P (list) - [Pa] List of pressure values for calculation
        * xi (list) - List of liquid mole fractions used in liquid_properties
        calculations
        * rhol (list) - [mol/:math:`m^3`] Evaluated liquid density values
        * weights (dict) - A dictionary where each key is a system constraint
        (e.g. T or xi) which is also a header used in an optional exp. data file.
        The value associated with a header can be a list as long as the number of
        data points to multiply by the objective value associated with each point,
        or a float to multiply the objective value of this data set.
        * objective_method (str) - The 'method' keyword in function
        despasito.parameter_fitting.fit_functions.obj_function_form.
        * density_opts (dict) - Optional,
        default={"min_density_fraction":(1.0 / 60000.0), "density_increment":10.0,
        "max_volume_increment":1.0E-4}, Dictionary of options used in calculating
        pressure vs. mole fraction curves.
        * kwargs for
        :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`

    Attributes
    ----------
    name : str
        Data type, in this case liquid_density
    Eos : obj
        Equation of state object
    weights : dict, Optional, default: {"some_property": 1.0 ...}
        Dictionary with keys corresponding to those in thermodict, with weighting
        factor or vector for each system property used in fitting
    obj_opts : dict
        Keywords to compute the objective function with
        :func:`~despasito.parameter_fitting.fit_functions.obj_function_form`.
    npoints : int
        Number of sets of system conditions this object computes
    result_keys : list
        Thermodynamic property names used in calculation of objective function. In in
        this case: ["rhol", "phil"]
    thermodict : dict
        Dictionary of inputs needed for thermodynamic calculations

        - calculation_type (str) default=liquid_properties
        - density_opts (dict) default={"min_density_fraction":(1.0 / 300000.0),
        "density_increment":10.0, "max_volume_increment":1.0E-4}

    """

    def __init__(self, data_dict):

        data_dict = data_dict.copy()

        super().__init__(data_dict)

        self.name = "liquid_density"
        tmp = {
            "min_density_fraction": (1.0 / 300000.0),
            "density_increment": 10.0,
            "max_volume_increment": 1.0e-4,
        }
        if "density_opts" in self.thermodict:
            tmp.update(self.thermodict["density_opts"])
        self.thermodict["density_opts"] = tmp

        if self.thermodict["calculation_type"] is None:
            self.thermodict["calculation_type"] = "liquid_properties"

        if "xi" in data_dict:
            self.thermodict["xilist"] = data_dict["xi"]
            del data_dict["xi"]
        if "T" in data_dict:
            self.thermodict["Tlist"] = data_dict["T"]
            del data_dict["T"]
        if "P" in data_dict:
            self.thermodict["Plist"] = data_dict["P"]
            del data_dict["P"]

        self.thermodict.update(data_dict)

        thermo_keys = ["Plist", "xilist", "Tlist"]
        self.result_keys = ["rhol", "phil"]

        key_list = list(set(thermo_keys + self.result_keys))
        self.thermodict.update(gtb.check_length_dict(self.thermodict, key_list))
        self.npoints = np.size(self.thermodict["Tlist"])

        if "xilist" not in self.thermodict and self.Eos.number_of_components > 1:
            raise ValueError("Ambiguous mixture composition. Define xi")
        thermo_defaults = [
            constants.standard_pressure,
            np.array([[1.0] for x in range(self.npoints)]),
            constants.standard_temperature,
        ]
        self.thermodict.update(gtb.set_defaults(self.thermodict, thermo_keys, thermo_defaults, lx=self.npoints))

        self.weights.update(gtb.check_length_dict(self.weights, self.result_keys, lx=self.npoints))
        self.weights.update(gtb.set_defaults(self.weights, self.result_keys, 1.0))

        if "Tlist" not in self.thermodict and "rhol" not in self.thermodict:
            raise ImportError("Given liquid property data, values for T, xi, and rhol should have" " been provided.")

        logger.info(
            "Data type 'liquid_properties' initiated with calculation_type, {}, and "
            "data types: {}.\nWeight data by: {}".format(
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
            A list of the predicted thermodynamic values estimated from thermo
            calculation. This list can be composed of lists or floats
        """

        # Remove results
        opts = self.thermodict.copy()
        tmp = self.result_keys + ["name", "parameters_guess"]
        for key in tmp:
            if key in opts:
                del opts[key]

        try:
            output_dict = thermo(self.Eos, **opts)
            output = [output_dict["rhol"]]
        except Exception:
            raise ValueError("Calculation of calc_liquid_density failed")
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

        # Reformat array of results
        phase_list, len_list = ff.reformat_output(phase_list)
        phase_list = np.transpose(np.array(phase_list))

        # objective function
        obj_value = ff.obj_function_form(
            phase_list, self.thermodict["rhol"], weights=self.weights["rhol"], **self.obj_opts
        )

        logger.info("Obj. breakdown for {}: rhol {}".format(self.name, obj_value))

        if np.isnan(obj_value) or obj_value == 0.0:
            obj_value = np.inf

        return obj_value
