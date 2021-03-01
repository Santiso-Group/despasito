import numpy as np
import despasito

beads = ["CH3", "CH2"]
beads_per_molecule = np.array([[2.0, 8.0]])
bead_library = {
    "CH3": {
        "epsilon": 256.7662,
        "lambdaa": 6.0,
        "lambdar": 15.04982,
        "sigma": 4.077257e-1,
        "Sk": 0.5725512,
        "Vks": 1,
        "mass": 0.015035,
    },
    "CH2": {
        "epsilon": 473.3893,
        "lambdaa": 6.0,
        "lambdar": 19.87107,
        "sigma": 4.880081e-1,
        "Sk": 0.2293202,
        "Vks": 1,
        "mass": 0.014027,
    },
}
cross_library = {"CH3": {"CH2": {"epsilon": 350.770}}}
Eos = despasito.equations_of_state.initiate_eos(
    eos="saft.gamma_mie",
    beads=beads,
    molecular_composition=beads_per_molecule,
    bead_library=bead_library,
    cross_library=cross_library,
    numba=True,
)

rho, T, xi = 553.0, 700.0, np.array([1.0])
AHS = Eos.saft_source.Ahard_sphere(rho, T, xi)
A1 = Eos.saft_source.Afirst_order(rho, T, xi)
A2 = Eos.saft_source.Asecond_order(rho, T, xi)
A3 = Eos.saft_source.Athird_order(rho, T, xi)
Am1 = AHS + A1 + A2 + A3
Am2 = Eos.Amonomer(rho, T, xi)
print(
    "The Monomer Contribution for Helmholtz Energy: {},\n    equals the sum of its components: {}".format(
        Am2, Am1
    )
)
