.. _startfitting-label:

Get Started Fitting Parameters
======================================

.. contents:: :local:

Files to Fit Parameters
########################
In this tutorial we will assume the user has already read and understood our "Setting-up a Thermodynamic Calculation" tutorial. Setting up an input file for parameter fitting is similar to writing a thermodynamic input script. Again, two files are needed for a basic calculation:

 #. `input.json`: Contains a dictionary of instructions and settings.
 #. `EOSgroup.json`: A dictionary of single group parameters (or estimates of parameters) used in the desired equation of state (EOS).

Parameters can be fit for one component at a time, and for as many parameters as desired can be fit. For this example we focus on methanol modeled with the SAFT-:math:`\gamma`-Mie EOS found in the `examples` directory.

`input_fit.json`::

    {
        "bead_configuration": [[["CH3OH", 1]]],
        "EOSgroup": "SAFTgroup.json",
        "optimization_parameters": {
            "fit_bead" : "CH3OH",
            "fit_parameter_names": ["epsilon"],
            "epsilon_bounds" : [150.0, 400.0]
        },
        "Wiley": {
            "data_class_type": "sat_props",
            "calctype": "sat_props",
            "file": "methanol_saturation.csv"
        },
        "Gibbard": {
            "data_class_type": "sat_props",
            "file": "methanol_psat.csv",
            "weights" : {
                "Psat" : 0.3,
                "rhov" : [0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
            }
        }
    }

Note that the name of this file doesn't really matter, but we use the standard prefix *input* to denote where these instructions are. This file is processed by the :func:`~despasito.input_output.read_input.process_param_fit_inputs` function.

The order of these items does not matter, but the first three items must be included, plus one of the follow dictionaries discussed below. The `bead_configuration` and `EOSgroup` are described in our basic thermodynamic calculation tutorial. Our new entry, `optimization_parameters` defines this as a parameter fitting job and outlines the basic instructions.

* `fit_bead`: One bead type may have parameters fit at a time, and must be present in the `bead_configuration` definition.
* `fit_parameter_names`: At least one parameter must be listed here. Allowed parameter names are EOS specific. Any of the self-interaction parameters may be listed, in this case its 'epsilon', but another can be added to this vector. If one wants to fit a cross-interaction parameter, use the parameter name followed by an underscore and another bead type used in `bead_configuration`. For example, let's say we were fitting one heavy atom per bead for methanol. Then `"bead_configuration": [[["CH3",1],["OH",1]]]` and if we wanted to fit self interaction parameter, epsilon, for "CH3" as well as the cross-interaction with "OH", then `"fit_parameter_names": ["epsilon","epsilon_OH"]`.
* `parameters_guess`: This optional array contains the initial guess for each respective parameter in `fit_parameter_names` and is the same length. If this vector is not provided, then the initial guess is taken from `EOSgroup`.
* `*_bounds`: This optional array is always of length two, containing the high and low limits for the parameter that replaces the asterisk in this definition. Notice above that "epsilon_bounds" defines the bounds of our chosen fit_param in this example.

Experimental data is the last mandatory entry in the input file. In our example we have two sets of experimental data, "Wiley" and "Gibbard". The dictionary name doesn't matter. If the entry "data_class_type" is in this dictionary, then DESPASITO will identify it as experimental data. These arbitrary names provide flexibility and are used in the output to allow you to identify if a particular data set has a high penalty compared to other data sets when fitting these parameters. This dictionary has a few entries.

* `data_class_type`: This defines the data_class_type object used in parameter fitting. See :ref:`data-types` for a list of available data types.
* `calctype`: Optional...ish. This defines the thermodynamic calculation used by the data_class_type chosen, see the data_class_type for the allowed options. In many cases there is only one option and you don't need to include it, however, the TLVE data_class_type can fit parameters to Tx data or Ty data. We added a feature where the object will try to guess which of these matches the provided data.
* `file`: The file name may be included if the data is in the same directory, or the path can be included with it. This file is expected to contain comma separated values, although it doesn't matter whether a .txt or .csv extension is used. The top line is skipped to allow inclusion of references. Column headers are the second line. Note that column headers should be thermo properties (e.g. T, P, x1, x2, y1, y2) without suffixes. Mole fractions x1, x2, ... should be in the same order as in the bead_configuration line of the input file. No mole fractions should be left out. The column headers in the file are dictionary keys used to initiate the data_class_type object.
* `rhodict`: Optional, include options for calculating the density vector that is the foundation of all the thermodynamic calculcations. See :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays` for details.
* `weights`: This dictionary allows the user to manually weight the influence of experimental data by some factor. This may be accomplished with a single factor multiplied by the entire array, or a vector of the same length as the experimental data given. The default is that all data has a weight of 1, but in the example above, the data from "Gibbard" is weighted individually in the case of vapor density for the purposes of this tutorial. Maybe we know that the instrument used for collecting this data is not as accurate with low values, now we can account for that.

After this input file, copy the SAFTgroup.json file from the example `despasito/examples/CH3OH_fit` and go ahead and run the calculation with:
``python -m despasito -i input_fit.json -vv``, 
It's that easy!

.. note:: Try the --jit option to speed it up.

DESPASITO uses global optimization methods from `scipy.optimize <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ for parameter fitting. The optional dictionary, ``global_opts`` may then be included for options specific to the function. Currently, brute, differential_evolution, and basinhopping are available. Our preference is to use the differential evolution method so we will elaborate here.


