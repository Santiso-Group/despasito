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
        * weights : list/float, Either a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
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

        data_type = []
        data_type_name = []
        if "xi" in data_dict: 
            self.xi = data_dict["xi"]
            data_type.append(self.xi)
            data_type_name.append("xi")
        if "T" in data_dict:
            self.T = data_dict["T"]
            data_type.append(self.T)
            data_type_name.append("T")
        if "yi" in data_dict:
            self.yi = data_dict["yi"]
            data_type.append(self.yi)
            data_type_name.append("yi")
        if "P" in data_dict: 
            self.P = data_dict["P"]
            data_type.append(self.P)
            data_type_name.append("P")

        if (not hasattr(self,"P") or not hasattr(self,"T")):
            raise ImportError("Given TLVE data, values for P and T should have been provided.")

        if (not hasattr(self,"xi") and not hasattr(self,"yi")):
            raise ImportError("Given TLVE data, mole fractions should have been provided.")

        if "calctype" not in data_dict:
            logger.warning("No calculation type has been provided.")
            if self.xi:
                self.calctype = "phase_xiT"
                logger.warning("Assume a calculation type of phase_xiT")
            elif self.yi:
                self.calctype = "phase_yiT"
                logger.warning("Assume a calculation type of phase_yiT")
            else:
                raise ValueError("Unknown calculation instructions")
        else:
            self.calctype = data_dict["calctype"]

        if any(np.array([len(x) for x in data_type]) == len(self.xi)) == False:
            raise ValueError("T, P, yi, and xi are not all the same length.")

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = 1.0

        try:
            self._rhodict = data_dict["rhodict"]
        except:
            self._rhodict = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

        logger.info("Data type 'TLVE' initiated with calctype, {}, and data types: {}".format(self.calctype,", ".join(data_type_name)))

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
                output_dict = thermo(eos, {"calculation_type":self.calctype,"Tlist":self.T,"xilist":self.xi,"Pguess":self.P,"rhodict":self._rhodict})
                output = [output_dict['P'],output_dict["yi"]]
            except:
                raise ValueError("Calculation of calc_xT_phase failed")

        elif self.calctype == "phase_yiT":
            try:
                output_dict = thermo(eos, {"calculation_type":self.calctype,"Tlist":self.T,"yilist":self.yi,"Pguess":self.P,"rhodict":self._rhodict})
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

        # objective function
        phase_list = self._thermo_wrapper(eos)
        phase_list, len_cluster = ff.reformat_ouput(phase_list)
        phase_list = np.transpose(np.array(phase_list))
   
        obj_value = np.sum((((phase_list[0] - self.P) / self.P)**2)*self.weights)
        if self.calctype == "phase_xiT":
            yi = np.transpose(self.yi)
            obj_value += np.sum((((phase_list[1:] - yi)/yi)**2)*self.weights)
        elif self.calctype == "phase_yiT":
            xi = np.transpose(self.xi)
            obj_value += np.sum((((phase_list[1:] - xi)/xi)**2)*self.weights)

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self.T)}
        return string
        
