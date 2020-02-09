"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.input_output.read_input as ri
import despasito.fit_parameters as fit
import despasito.fit_parameters.fit_funcs as funcs
import despasito.equations_of_state
import pytest
import sys
import numpy as np

Tlist = [323.2]
xilist = [[0.01,0.99]]
yilist = [[0.97702, 0.02298]]
Plist = [4492927.45]
rho = np.array([19986.78358869])
P = np.array([4099056.163132072])

## EOS Object
beads = ['CH3OH']
nui = np.array([[1.]])
beadlibrary = {'CH3OH': {'epsilon': 375.01, 'l_a': 6.0, 'l_r': 16.352, 'sigma': 3.463e-10, 'Sk': 1.0, 'Vks': 2, 'mass': 0.0310337}}
eos = despasito.equations_of_state.eos(eos="saft.gamma_mie",beads=beads,nui=nui,beadlibrary=beadlibrary)

## Exp Data dict
exp_data = {'Wiley': {'name': 'sat_props', 'calctype': 'sat_props', 'T': np.array([200.]), 'Psat': np.array([6.1000e+01]), 'rhol': np.array([27474.40699]), 'rhov': np.array([3.12109900e-03])}, 'Gibbard': {'name': 'sat_props', 'calctype': 'sat_props', 'T': np.array([288.1506]), 'Psat': np.array([ 9884.4])}}

## Optimization options
opt_params = {"fit_bead" : "CH3OH", "fit_params": ["epsilon"], "epsilon_bounds" : [150.0, 400.0]}

thermo_dict = {"opt_params": opt_params, "exp_data": exp_data, "basin_dict": {"niter": 1, "niter_success": 1}, "beadparams0": [381.0], "minimizer_dict": {"tol": 1e-1}}


def test_fit_import():
#    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.fit_parameters" in sys.modules

def test_fit_1comp(eos=eos,thermo_dict=thermo_dict):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(eos,thermo_dict)
        
    assert output["final_parameters"][0]==pytest.approx(3042623.2,abs=1e-3) and output["objective_value"][0]<1e-8

