import numpy as np

import despasito
import despasito.input_output.read_input as io
import despasito.thermodynamics as thermo
import despasito.equations_of_state

despasito.initiate_logger(console=True, verbose=10)

Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=["CH3", "CH2"],
    molecular_composition=np.array([[2.0, 4.0], [2.0, 5.0]]),
    bead_library=io.json_to_dict("../../library/SAFTgroup.json"),
    cross_library=io.json_to_dict("../../library/SAFTcross.json"),
)

output = thermo.thermo(
    Eos,
    calculation_type="liquid_properties",
    Tlist=320.0,
    Plist=1e5,
    xilist=np.array([0.4, 0.6]),
)

print("Thermo Output", output)
args = (output["rhol"][0], 320.0, [0.4, 0.6])
print("Helmholtz Contributions:")
print("    Ideal: ", Eos.Aideal(*args))
print("    Monomer: ", Eos.saft_source.Amonomer(*args))
print("    Chain: ", Eos.saft_source.Achain(*args))
