"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.input_output.read_input as ri
import despasito.parameter_fitting as fit
import despasito.parameter_fitting.fit_funcs as funcs
import despasito.equations_of_state
import pytest
import copy
import sys
import numpy as np

## EOS Object
beads = ['CH3OH']
nui = np.array([[1.]])
beadlibrary = {'CH3OH': {'epsilon': 375.01, 'lambdaa': 6.0, 'lambdar': 16.352, 'sigma': 3.463e-1, 'Sk': 1.0, 'Vks': 2, 'mass': 0.0310337}}
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie",beads=beads,nui=nui,beadlibrary=beadlibrary)

## Exp Data dict
exp_data_sat = {'Wiley': {'name': 'saturation_properties', "eos_obj":eos, 'calculation_type': 'saturation_properties', 'T': np.array([200.]), 'Psat': np.array([6.1000e+01]), 'rhol': np.array([27474.40699]), 'rhov': np.array([3.12109900e-03])}, 
            'Gibbard': {'name': 'saturation_properties', "eos_obj":eos, 'calculation_type': 'saturation_properties', 'T': np.array([288.1506]), 'Psat': np.array([ 9884.4])}}
exp_data_sol = {'Wiley': {'name': 'solubility_parameter', "eos_obj":eos, 'calculation_type': 'solubility_parameter', 'T': np.array([288.1506]), 'P': np.array([11152.285]), 'rhol': np.array([24098.4771]), 'delta': np.array([29161.4886])}}
exp_data_density = {'Wiley': {'name': 'liquid_density', "eos_obj":eos, 'calculation_type': 'liquid_properties', 'T': np.array([288.1506]), 'P': np.array([11152.285]), 'rhol': np.array([24098.4771])}}

## Optimization options
opt_params = {"fit_bead" : "CH3OH", "fit_params": ["epsilon"], "epsilon_bounds" : [300.0, 400.0]}
thermo_dict0 = {"opt_params": opt_params, "beadparams0": [384.0], "global_dict": {"method": "single_objective"}}

def test_fit_import():
#    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.parameter_fitting" in sys.modules

thermo_dict0["exp_data"] = exp_data_sol
def test_solubility_so(eos=eos,thermo_dict=copy.deepcopy(thermo_dict0)):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)
        
    assert output["final_parameters"][0]==pytest.approx(375.01,abs=1.0) and output["objective_value"]<1.1


thermo_dict0["exp_data"] = exp_data_density
def test_density_so(eos=eos,thermo_dict=copy.deepcopy(thermo_dict0)):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)
        
    assert output["final_parameters"][0]==pytest.approx(375.01,abs=1.0) and output["objective_value"]<1.5

thermo_dict0["exp_data"] = exp_data_sat
thermo_dict0["global_dict"] = {"method": "single_objective"}
def test_saturation_de(eos=eos,thermo_dict=copy.deepcopy(thermo_dict0)):

    thermo_dict["density_dict"] = {"pressure_min": 10}
    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert output["final_parameters"][0]==pytest.approx(375.01,abs=1.0) and output["objective_value"]==pytest.approx(5.8195,abs=0.01)
