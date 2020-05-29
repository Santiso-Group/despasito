"""
Contains constants used in the other files.    
"""

import numpy as np
#physical constants

kb = 1.38064852e-23  #Boltzmann constant J/K
me = 9.10938356e-31  #mass of electron kg
h = 6.62607004e-34  #Planck's constant J*S
Nav = 6.02214086e23  #Avogadro's number
R = 8.31446261815324 # Gas constant J/ ( mol*K) 
molecule_per_nm3 = 6.02214086e-4 # mol/m^3 to molecules/nm^3 
m2nm = 1e+9 # meters to nm 
Atometer = 1.0e-10  #conversion of angstroms to meters

ckl_coef = np.array([[0.81096, 1.7888, -37.578, 92.284], [1.0205, -19.341, 151.26, -463.50],
                     [-1.9057, 22.845, -228.14, 973.92], [1.0885, -6.1962, 106.98, -677.64]])
