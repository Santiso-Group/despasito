"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.input_output.read_input as ri
import despasito.fit_parameters as fit
import despasito.fit_parameters.fit_funcs as funcs
import despasito.equations_of_state
import pytest
import copy
import sys
import numpy as np

beads = ['CO2', 'H2O353']
nui = np.array([[1., 0.],[0., 1.]])
beadlibrary = {'H2O353': {'epsilon': 479.56, 'l_a': 6.0, 'l_r': 8.0, 'sigma': 3.0029e-1, 'Sk': 1.0, 'Vks': 1, 'mass': 0.018015}, 'CO2': {'epsilon': 353.55, 'l_a': 6.66, 'l_r': 23.0, 'sigma': 3.741e-1, 'Sk': 1.0, 'Vks': 1, 'mass': 0.04401}}
crosslibrary = {'CO2': {'H2O353': {'epsilon': 432.69}}}
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie", beads=beads, nui=nui, beadlibrary=beadlibrary, crosslibrary=crosslibrary)

## Exp Data dict
Tlist = np.array([353.0])
xilist = np.array([[0.0128, 0.9872]])
yilist = np.array([[0.9896, 0.0104]])
Plist = np.array([7.08e+6])

Tlist2 = np.array([353.0, 353.0])
xilist2 = np.array([[0.0128, 0.9872], [0.014, 0.986]])
yilist2 = np.array([[0.9896, 0.0104], [0.9903, 0.0097]])
Plist2 = np.array([7.08e+6, 8.08e+6])

exp_data_phase_yiT = {'Wiley': {'name': 'TLVE', "eos_obj":eos, 'calculation_type': 'phase_yiT', 'T': Tlist, 'P': Plist, "xi": xilist, "yi": yilist}}
#exp_data_flash = {'Wiley': {'name': 'flash', "eos_obj":eos, 'calculation_type': 'flash', 'T': Tlist2, 'P': Plist2, "xi": xilist2, "yi": yilist2}}
exp_data_flash = {'Wiley': {'name': 'flash', "eos_obj":eos, 'calculation_type': 'flash', 'T': Tlist, 'P': Plist, "xi": xilist, "yi": yilist}}

## Optimization options
opt_params = {"fit_bead" : "CO2", "fit_params": ["epsilon_H2O353"], "epsilon_H2O353_bounds" : [150.0, 600.0]}
thermo_dict0 = {"opt_params": opt_params, "exp_data": exp_data_phase_yiT, "beadparams0": [432.69], "global_dict": {"method": "single_objective"}}


#def test_phase_yiT(eos=eos,thermo_dict=copy.deepcopy(thermo_dict0)):
#
#    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
#    output = fit.fit(thermo_dict)
#        
#    assert output["final_parameters"][0]==pytest.approx(432.69,abs=1.0) and output["objective_value"]==pytest.approx(722.6,abs=1.0)

thermo_dict0["exp_data"] = exp_data_flash
thermo_dict0["min_mole_fraction0"] = 0.98
def test_flash(eos=eos,thermo_dict=copy.deepcopy(thermo_dict0)):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(thermo_dict)
        
    assert output["final_parameters"][0]==pytest.approx(432.69,abs=1.0) and output["objective_value"]==pytest.approx(0.7405,abs=0.0001)



