r"""
Objects for storing and producing objective values for comparing experimental data to EOS predictions.    
"""

import sys
import numpy as np
from multiprocessing import Pool

from despasito.thermodynamics import calc
from despasito.fit_parameters import fit_funcs as ff
from despasito.fit_parameters.interface import ExpDataTemplate

##################################################################
#                                                                #
#                              TLVE                              #
#                                                                #
##################################################################
class Data(ExpDataTemplate):

    r"""
    Object for Temperature dependent VLE data. This data could be evaluated with phase_xiT or phase_yiT. Most entries in the exp. dictionary are converted to attributes. 

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

        # Self interaction parameters
        self.name = data_dict["name"]
        try:
            self.calctype = data_dict["calctype"]
        except:
            self.calctype = "phase_xiT"

        try:
            self.xi = data_dict["xi"]
            self.T = data_dict["T"]
            self.yi = data_dict["yi"]
            self.P = data_dict["P"]
        except:
            raise ImportError("Given TLVE data, values for P, T, xi, and yi should have been provided.")

        if any(np.array([len(x) for x in [self.xi, self.yi, self.T, self.P]]) == len(self.xi)) == False:
            raise ValueError("T, P, yi, and xi are not all the same length.")

        try:
            self.weights = data_dict["weights"]
        except:
            self.weights = 1.0

        try:
            self._rhodict = data_dict["rhodict"]
        except:
            self._rhodict = {"minrhofrac":(1.0 / 300000.0), "rhoinc":10.0, "vspacemax":1.0E-4}

    def _thermo_wrapper(self, T, xi, yi, eos, rhodict):

        """
        Generate thermodynamic predictions from eos object

        Parameters
        ----------
        T : float
            Temperature of system [K]
        xi : list
            Liquid mole fractions, must add up to one
        yi : list
            Vapor mole fractions, must add up to one
        eos : obj
            EOS object with updated parameters
        rhodict : dict, Optional, default: {}
            Dictionary of options used in calculating pressure vs. mole fraction curves.

        Returns
        -------
        phase_list : float
            A list of the predicted thermodynamic values estimated from thermo calculation. This list can be composed of lists or floats
        """

        if self.calctype == "phase_xiT":
            try:
                P, yi = calc.calc_xT_phase(xi, T, eos, rhodict=rhodict)
            except:
                raise ValueError("Calculation of calc_xT_phase failed for xi:%s, T:%g" %(str(xi),T))
            return P, yi

        elif self.calctype == "phase_yiT":
            try:
                P, xi = calc.calc_yT_phase(yi, T, eos, rhodict=rhodict)
            except:
                raise ValueError("Calculation of calc_yT_phase failed for yi:%s, T:%g" %(str(yi),T))
            return P, xi

    def objective(self, eos, threads=1):

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

        # Generate input_list to distribute to multiple threads.
      #  input_list = []
        phase_list = []
        for i,T in enumerate(self.T):
      #       input_list.append((T, self.xi[i], self.yi[i], eos, self._rhodict))
             input_list = (T, self.xi[i], self.yi[i], eos, self._rhodict)
             phase_list.append(self._thermo_wrapper(*input_list))
             print(phase_list[-1])

        # Submit data points to be evaluated
        #p = Pool(threads)
        #phase_list = p.map(self._thermo_wrapper, input_list, 1)
        #p.close()
        #p.join()

        # Reformat array of results
        phase_list, len_list = ff.reformat_ouput(phase_list) 

        # objective function
        obj_value = np.sum((((phase_list[0] - self.P) / self.P)**2)*self.weights)
        if self.calctype == "phase_xiT":
            for i in range(1,len_list[1]):
                obj_value += np.sum((((phase_list[1+i] - self.yi[i]))**2)*self.weights)
        elif self.calctype == "phase_yiT":
            for i in range(1,len_list[1]):
                obj_value += np.sum((((phase_list[1+i] - self.xi[i]))**2)*self.weights)

        return obj_value

    def __str__(self):

        string = "Data Set Object\nname: %s\ncalctype:%s\nNdatapts:%g" % {self.name, self.calctype, len(self.T)}
        return string
        
