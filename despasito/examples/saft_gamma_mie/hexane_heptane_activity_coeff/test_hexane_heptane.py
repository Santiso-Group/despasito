import sys
import numpy as np

import despasito
import despasito.input_output as io
import despasito.thermodynamics as thermo
import despasito.equations_of_state

despasito.initiate_logger(console=True, verbose=10)
beadlibrary = io.json_to_dict("../../library/SAFTgroup.json")
crosslibrary = io.json_to_dict("../../library/SAFTcross.json")

T = 320.0
P = 1e5
xi = np.array([0.4, 0.6])

beads = ["CH3", "CH2"]
nui = np.array([[2.0, 4.0], [2.0, 5.0]])

eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads,
    molecular_composition=nui,
    bead_library=beadlibrary,
    cross_library=crosslibrary,
)

output = thermo.thermo(
    eos, calculation_type="activity_coefficient", Tlist=T, Plist=P, xilist=xi
)
