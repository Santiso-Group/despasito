.. _settingup-label:

Setting-up a Thermodynamic Calculation
======================================

.. contents:: :local:

See documentation below for a walk-through or :ref:`input-schema` for a condensed outline of available options.

.. _basic-use:

Basic Use
#########
In this tutorial we will assume the user is creating the necessary files for the python module use of this package (rather than as an imported library). A basic calculation requires 2 files:

 #. `input.json`: Contains a dictionary of instructions and settings.
 #. `SAFTgroup.json`: A dictionary of single group parameters used in the desired equation of state (EOS).

For this example we focus on carbon dioxide and water modeled with the SAFT-:math:`\gamma`-Mie EOS found in the `examples` directory.

`input_vapor.json`::

    {
        "bead_configuration": [[["CO2", 1]], [["H2O", 1]]],
        "EOSgroup": "SAFTgroup.json",
        "EOScross": "SAFTcross.json",
        "output_file": "out_vapor.txt",
        "calculation_type" : "vapor_properties",
        "Tlist" : [323.2],
        "yilist": [[0.9954, 0.0046]],
        "Plist": [4492927.45]
    }

Notice that the name of this file doesn't really matter, but we use the prefix *input* as a convenient standard to denote where these instructions are. This file is processed by the :func:`~despasito.input_output.read_input.extract_calc_data` function.

The first three lines are used in producing our EOS object. The ``bead_configuration`` line is an list of system components. Each components is defined as a list of group (i.e. segment or bead) types. Each of those groups is then a list of the bead name followed by the number of beads in the components. Notice that connectivity isn't captured by SAFT in this way. The next file, ``EOSgroup``, contains the self-interaction parameters of a certain group. The optional file, ``EOScross``, then contains any cross interaction parameters. If it isn't provided, then the EOS object will estimate these interactions with the defined combining rules. The default EOS is SAFT-:math:`\gamma`-Mie, but any other EOS can be added to DESPASITO using our class interface. Specifying another EOS is explained in, :ref:`contributing-eos`.

The last 4 lines are thermodynamic calculation instructions. The defined ``calculation_type`` dictated the other required entries. In this case, defining the pressure is optional, if it was missing DESPASITO would assume atmospheric pressure.

See :ref:`calculation-types` for a list of available types of thermodynamic calculations.

`SAFTgroup.json`::
    
    {
        "CO2": {
            "epsilon": 207.891,
            "lambdaa": 5.0550,
            "lambdar": 26.408,
            "sigma": 3.05e-1,
            "Sk": 0.84680,
            "Vks": 2,
            "mass": 0.04401,
            "Nk-H": 1,
            "Nk-a1": 1
        },
        "H2O": {
            "epsilon": 266.68,
            "lambdaa": 6.0,
            "lambdar": 17.020,
            "sigma": 3.0063e-1,
            "Sk": 1.0,
            "Vks": 1,
            "mass": 0.018015,
            "Nk-H": 2,
            "Nk-e1": 2,
            "epsilonHB-H-e1": 1985.4,
            "K-H-e1": 101.69e-3
        }
    }

In the `SAFTgroup.json` file, define as many groups as desired. Those used in the calculation are specified in the ``bead_configuration`` line. The parameter keys used are defined in the documentation for the chosen EOS object. As an example, the cross-interactions can be defined as follows:

`SAFTcross.json`::

    {
        "CO2": {
            "H2O": {"epsilon": 226.38, "epsilonHB-H-e1":2200.0, "K-H-e1":91.419e-3}
        }
    }

After creating each of these files, go ahead and run the calculation with:

``python -m despasito -i input_vapor.json -vv``

It's that easy! The result will be two files. A log file, *despasito.log*, contains the details of the calculation at the verbosity level INFO. Although the log file contains the calculation results, a condensed, comma separated format output is also provided.

`out_vapor.txt`::

    # This data was generated in DESPASITO using the thermodynamic calculation: vapor_properties
    # P [Pa], T [K], yi1, yi2, rhov [mol/m^3], phiv1, phiv2, flagv,
     4492927.45, 323.2, 0.9954, 0.0046, 2074.9925043467697, 0.8434455796620214, 0.09770908515893507, 2,

Imported Library
#################

Calculations may also be completed by importing DESPASITO as a library, where additional equation of state quantities are accessible. Here is an example from the package ``examples`` directory.

`hexane_heptane_test.txt`::

    import numpy as np
    
    import despasito
    import despasito.input_output.read_input as io
    import despasito.thermodynamics as thermo
    import despasito.equations_of_state
    
    #despasito.initiate_logger(console=True, verbose=10) # Uncomment to output logs usually written to a file, to the standard output.
    
    Eos = despasito.equations_of_state.initiate_eos(
        eos="saft.gamma_mie",
        beads=["CH3", "CH2"],
        molecular_composition=np.array([[2.0, 4.0], [2.0, 5.0]]),
        bead_library=io.json_to_dict("../../library/SAFTgroup.json"),
        cross_library=io.json_to_dict("../../library/SAFTcross.json"),
    )
    
    output = thermo.thermo(
        Eos, 
        calculation_type="vapor_properties", 
        Tlist=[320.0], 
        Plist=[1e+5], 
        xilist=np.array([[0.4, 0.6]]),
    )

    print("Thermo Output",output)
    args = ( output["rhol"][0], 320.0, [0.4, 0.6])
    print("Helmholtz Contributions:")
    print("    Ideal: ",Eos.Aideal(*args))
    print("    Monomer: ",Eos.saft_source.Amonomer(*args))
    print("    Chain: ",Eos.saft_source.Achain(*args))

The output is then (after formatting for readability):

.. code-block:: python

    Thermo Output {
        'P': array([100000.]), 
        'T': array([320.]), 
        'xi': array([[0.4, 0.6]]), 
        'rhol': array([6906.839179346179]), 
        'phil': array([array([0.48515872, 0.16785448])]), 
        'flagl': array([1]),
    }
    Helmholtz Contributions:
        Ideal:  [-14.0487984]
        Monomer:  [-5.01968519]
        Chain:  [-0.55952224]

Referencing a Library of Parameters
###################################

If you checked our examples folder in DESPASITO, you might have noticed that they don't quite match the files above. You can actually make this process even easier by eliminating the need to make and `SAFTgroup.json` and `SAFTcross.json` file for every calculation by having one file with all parameters, that's what we did. This can be accomplished by keeping the file in one location and providing DESPASITO with the path in one of two ways:

 #. In the string within input.json, include a absolute or relative path to the file.
 #. In the command line, include the `-p` option to define the absolute or relative path.

Other options for controlling the output are also available in the command line implementation. Type ``python -m despasito -h`` to discover more.

Specifying Equation of State (EOS)
##################################

By default, DESPASITO used the SAFT-:math:`\gamma`-Mie equation of state. However, you can change the EOS by adding the ``eos`` keyword to the ``input.json`` file. This option is passed to the :func:`~despasito.equations_of_state.initiate_eos` function, an example of this can be found in the Peng-Robinson calculations in the ``examples`` directory. The syntax for an equation of state is ``module.eos``, where ``module`` is the family the EOS belongs to, and ``eos`` is the equation of state. For Peng-Robinson this entry would be,

    ``"eos": "cubic.peng_robinson"``

and for SAFT-:math:`\gamma`-Mie,

    ``"eos": "saft.gamma_mie"``

See :ref:`EOS-types` for a list of available equations of state.

