"""
Unit and regression test for the despasito package.
"""

# Import package, test suite, and other packages as needed
import despasito.input_output.read_input as ri
import despasito.parameter_fitting as fit
import despasito.parameter_fitting.fit_functions as funcs
import despasito.equations_of_state
import pytest
import copy
import sys
import numpy as np

beads = ["CO2", "H2O353"]
molecular_composition = np.array([[1.0, 0.0], [0.0, 1.0]])
bead_library = {
    "H2O353": {
        "epsilon": 479.56,
        "lambdaa": 6.0,
        "lambdar": 8.0,
        "sigma": 3.0029e-1,
        "Sk": 1.0,
        "Vks": 1,
        "mass": 0.018015,
    },
    "CO2": {
        "epsilon": 353.55,
        "lambdaa": 6.66,
        "lambdar": 23.0,
        "sigma": 3.741e-1,
        "Sk": 1.0,
        "Vks": 1,
        "mass": 0.04401,
    },
}
cross_library = {"CO2": {"H2O353": {"epsilon": 432.69}}}
Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
    cross_library=cross_library,
)

## Exp Data dict
Tlist = np.array([353.0])
xilist = np.array([[0.0128, 0.9872]])
yilist = np.array([[0.9896, 0.0104]])
Plist = np.array([7.08e6])

Tlist2 = np.array([353.0])
xilist2 = np.array([[0.008, 0.992]])
yilist2 = np.array([[0.9857, 0.0143]])
Plist2 = np.array([4.05e6])

exp_data_dew_pressure = {
    "Wiley": {
        "data_class_type": "TLVE",
        "eos_obj": Eos,
        "calculation_type": "dew_pressure",
        "T": Tlist2,
        "P": Plist2,
        "xi": xilist2,
        "yi": yilist2,
    }
}
exp_data_flash = {
    "Wiley": {
        "data_class_type": "flash",
        "eos_obj": Eos,
        "calculation_type": "flash",
        "T": Tlist,
        "P": Plist,
        "xi": xilist,
        "yi": yilist,
    }
}

## Optimization options
optimization_parameters = {
    "fit_bead": "CO2",
    "fit_parameter_names": ["epsilon_H2O353"],
    "epsilon_H2O353_bounds": [150.0, 600.0],
}
thermo_dict_dew = {
    "optimization_parameters": optimization_parameters,
    "exp_data": exp_data_dew_pressure,
    "parameters_guess": [432.69],
    "global_opts": {"method": "single_objective"},
}

thermo_dict_flash = {
    "optimization_parameters": optimization_parameters,
    "exp_data": exp_data_flash,
    "parameters_guess": [432.69],
    "global_opts": {"method": "single_objective"},
}

def test_dew_pressure(Eos=Eos,thermo_dict=thermo_dict_dew.copy()):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert output["parameters_final"][0]==pytest.approx(432.69,abs=1.0) and output["objective_value"]==pytest.approx(854.19,abs=1.0)

def test_flash(Eos=Eos, thermo_dict=thermo_dict_flash.copy()):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert output["parameters_final"][0] == pytest.approx(432.69, abs=1.0) and output[
        "objective_value"
    ] == pytest.approx(0.730, abs=0.001)
