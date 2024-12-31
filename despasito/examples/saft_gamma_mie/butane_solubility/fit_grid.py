import numpy as np

import despasito
import despasito.input_output.read_input as io
import despasito.parameter_fitting as fit
from despasito.equations_of_state import initiate_eos

despasito.initiate_logger(console=True, verbose=10)

Eos = initiate_eos(
    eos="saft.gamma_mie",
    beads=["CH2", "CH3"],
    molecular_composition=np.array([[2.0, 2.0]]),
    bead_library=io.json_to_dict("../../library/SAFTgroup.json"),
)

fit.fit(
    optimization_parameters={
        "fit_bead": "CH3",
        "fit_parameter_names": ["epsilon_CH2"],
        "epsilon_CH2_bounds": [150.0, 600.0],
        "parameters_guess": [300.0],
    },
    exp_data={
        "Knovel": {
            "data_class_type": "liquid_density",
            "eos_obj": Eos,
            "calculation_type": "liquid_properties",
            "T": np.array([272.15, 323.15, 298.15]),
            "rhol": np.array([10357.0, 10364.8, 10140.0]),
            "delta": np.array([14453.0, 13700.0, 14100.0]),
        },
    },
    global_opts={"method": "grid_minimization", "Ns": 3},
)
