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

## EOS Object
beads = ["CH3OH"]
molecular_composition = np.array([[1.0]])
bead_library = {
    "CH3OH": {
        "epsilon": 375.01,
        "lambdaa": 6.0,
        "lambdar": 16.352,
        "sigma": 3.463e-1,
        "Sk": 1.0,
        "Vks": 2,
        "mass": 0.0310337,
    }
}
Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
)

beads = ["CH3", "CH2"]
molecular_composition = np.array([[2.0, 4.0], [2.0, 10.0]])
bead_library = {
    "CH3": {
        "epsilon": 256.7662,
        "lambdaa": 6.0,
        "lambdar": 15.04982,
        "sigma": 4.077257e-1,
        "Sk": 0.5725512,
        "Vks": 1,
        "mass": 0.01503502,
    },
    "CH2": {
        "epsilon": 473.3893,
        "lambdaa": 6.0,
        "lambdar": 19.87107,
        "sigma": 4.880081e-1,
        "Sk": 0.2293202,
        "Vks": 1,
        "mass": 0.01402708,
    },
}
cross_library = {"CH3": {"CH2": {"epsilon": 350.77}}}
Eos_mix = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads,
    molecular_composition=molecular_composition,
    bead_library=bead_library,
    cross_library=cross_library,
)


## Exp Data dict
exp_data_sat = {
    "Wiley": {
        "data_class_type": "saturation_properties",
        "eos_obj": Eos,
        "calculation_type": "saturation_properties",
        "T": np.array([200.0]),
        "Psat": np.array([6.1000e01]),
        "rhol": np.array([27474.40699]),
        "rhov": np.array([3.12109900e-03]),
    },
    "Gibbard": {
        "data_class_type": "saturation_properties",
        "eos_obj": Eos,
        "calculation_type": "saturation_properties",
        "T": np.array([288.1506, 303.1427]),
        "Psat": np.array([9884.4, 21874.3]),
    },
}

exp_data_sol = {
    "Wiley": {
        "data_class_type": "solubility_parameter",
        "eos_obj": Eos,
        "calculation_type": "solubility_parameter",
        "T": np.array([288.1506]),
        "P": np.array([11152.285]),
        "rhol": np.array([24098.4771]),
        "delta": np.array([29161.4886]),
    }
}

exp_data_density = {
    "dx.doi.org/10.1021/acs.jced.0c00943": {
        "data_class_type": "liquid_density",
        "eos_obj": Eos_mix,
        "calculation_type": "liquid_properties",
        "T": np.array([293.15]),
        "xi": np.array([[0.1999, 0.8001], [0.7998, 0.2002]]),
        "P": np.array([5e6]),
        "rhol": np.array([4844.03412, 6863.260764]),
    }
}

## Optimization options
optimization_parameters = {
    "fit_bead": "CH3OH",
    "fit_parameter_names": ["epsilon"],
    "epsilon_bounds": [300.0, 400.0],
}
thermo_dict0 = {
    "optimization_parameters": optimization_parameters,
    "parameters_guess": [384.0],
    "global_opts": {"method": "single_objective"},
}

optimization_parameters_mix = {
    "fit_bead": "CH3",
    "fit_parameter_names": ["epsilon_CH2"],
    "epsilon_bounds": [300.0, 400.0],
}
thermo_dict_mix = {
    "optimization_parameters": optimization_parameters_mix,
    "parameters_guess": [350],
    "global_opts": {"method": "single_objective"},
}


def test_fit_import():
    #    """Sample test, will always pass so long as import statement worked"""
    assert "despasito.parameter_fitting" in sys.modules


thermo_dict0["exp_data"] = exp_data_sol


def test_solubility_so(Eos=Eos, thermo_dict=thermo_dict0.copy()):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert (
        output["parameters_final"][0] == pytest.approx(375.01, abs=1.0)
        and output["objective_value"] < 1.1
    )


thermo_dict_mix["exp_data"] = exp_data_density


def test_density_so(Eos=Eos_mix, thermo_dict=thermo_dict_mix.copy()):

    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert (
        output["parameters_final"][0] == pytest.approx(350.0, abs=1.0)
        and output["objective_value"] < 1.5
    )


thermo_dict0["exp_data"] = exp_data_sat
thermo_dict0["global_opts"] = {"method": "single_objective"}


def test_saturation_de(Eos=Eos, thermo_dict=thermo_dict0.copy()):

    thermo_dict["density_opts"] = {"pressure_min": 10}
    thermo_dict = ri.process_param_fit_inputs(thermo_dict)
    output = fit.fit(**thermo_dict)

    assert output["parameters_final"][0] == pytest.approx(375.01, abs=1.0) and output[
        "objective_value"
    ] == pytest.approx(5.7658, abs=0.01)
