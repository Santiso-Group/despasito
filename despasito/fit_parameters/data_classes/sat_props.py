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
    Object for saturation data. This data is evaluated with "sat_props". 

    Parameters
    ----------
    data_dict : dict
        Dictionary of exp data of saturation properties.

        * name : str, data type, in this case SatProps
        * calctype : str, Optional, default: 'sat_props
        * T : list, List of temperature values for calculation
        * xi : list, List of liquid mole fractions used in saturation properties calculations, should be 1 for the molecule of focus and 0 for the rest.
        * weights : list/float, Either a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
        * rhodict : dict, Optional, default: {"minrhofrac":(1.0 / 60000.0), "rhoinc":10.0, "vspacemax":1.0E-4}, Dictionary of options used in calculating pressure vs. mole fraction curves.

    Attributes
    ----------
    name : str
        Data type, in this case SatProps
    calctype : str, Optional, default: 'sat_props'
        Thermodynamic calculation type
    T : list
        List of temperature values for calculation
    xi : list
        List of liquid mole fractions, only one should be equal to 1.
    """

    def __init__(self, data_dict):

        logger = logging.getLogger(__name__)

        # Self interaction parameters
        self.name = data_dict["name"]
        try:
            self.calctype = data_dict["calctype"]
        except:
            self.calctype = "sat_props"

        data_type = []
        data_type_name = []
        if "xi" in data_dict:
            self.xi = data_dict["xi"]
            data_type.append(self.xi)
            data_type_name.append("xi")
        if "yi" in data_dict:
            self.xi = data_dict["yi"]
            data_type.append(self.xi)
            data_type_name.append("xi")
            logger.info("Vapor mole fraction recorded as 'xi'")
        if "T" in data_dict:
            self.T = data_dict["T"]
            data_type.append(self.T)
            data_type_name.append("T")
        if "rhol" in data_dict:
            self.rhol = data_dict["rhol"]
            data_type.append(self.rhol)
            data_type_name.append("rhol")
        if "rhov" in data_dict:
            self.rhov = data_dict["rhov"]
            data_type.append(self.rhov)
            data_type_name.append("rhov")
        if "P" in data_dict:
            self.Psat = data_dict["P"]
            data_type.append(self.Psat)
            data_type_name.append("Psat")
        if "Psat" in data_dict:
            self.Psat = data_dict["Psat"]
            data_type.append(self.Psat)
            data_type_name.append("Psat")

        tmp = ["T"]
        if not all([hasattr(self,x) for x in tmp]):
            raise ImportError("Given saturation data, value(s) for T should have been provided.")

        tmp = ["Psat","rhol","rhov"]
        if not any([hasattr(self,x) for x in tmp]):
            raise ImportError("Given saturation data, values for Psat, rhol, and/or rhov should have been provided.")

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = 1.0

        try:
            self._rhodict = data_dict["rhodict"]
        except:
            self._rhodict = {"minrhofrac":(1.0 / 80000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

        logger.info("Data type 'sat_props' initiated with calctype, {}, and data types: {}".format(self.calctype,", ".join(data_type_name)))

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
        try:
            self.xi = data_dict["xi"]
        except:
            if len(eos._nui) > 1:
                raise ValueError("Ambiguous instructions. Include xi to define intended component to obtain saturation properties")
            else:
                self.xi = np.array([[1.0] for x in range(len(self.T))])
 
        # Run thermo calculations
        try:
            output_dict = thermo(eos, {"calculation_type":self.calctype,"Tlist":self.T,"xilist":self.xi,"rhodict":self._rhodict})
            output = [output_dict["Psat"],output_dict["rhol"],output_dict["rhov"]]
        except:
            raise ValueError("Calculation of calc_Psat failed")

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
        if hasattr(self,"Psat"):
            obj_value = np.sum((((phase_list[0] - self.Psat) / self.Psat)**2)*self.weights)
        if hasattr(self,"rhol"):
            obj_value = np.sum((((phase_list[1] - self.rhol) / self.rhol)**2)*self.weights)
        if hasattr(self,"rhov"):
            obj_value = np.sum((((phase_list[2] - self.rhov) / self.rhov)**2)*self.weights)

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self.T)}
        return string
        
