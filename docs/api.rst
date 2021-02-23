
API Documentation
=================

DESPASITO has been primarily designed as a command line tool but can be used as an imported package.

See examples directory or Input/Output documentation for input file structures.

Command Line
------------
.. argparse::
   :ref: despasito.main.get_parser
   :prog: python -m despasito

Imported Package
----------------
Once installed, DESPASITO can be easily imported with ``import despasito``.
Each module will then need to replicate our main command line function and call each module in succession.

#. Provide ``input.json`` file name to ``read_input.extract_calc_data`` to obtain the appropriate dictionaries for the equation of state (Eos) object, and the thermodynamic calculation.
#. Use Eos dictionary of bead types and parameters to generate Eos object used in thermodynamic calculations
#. Choose calculation type and use Eos object and thermodynamic dictionary to start calculation.

Some steps could be skipped if your script already contains the appropriate output of a given step.

The intermediate logging provided in the Input File Schema can also be accessed for imported functions through the :func:`initiate_logger <despasito.initiate_logger>`.

Input File Schema
-----------------

General Keywords
________________
 * **bead_configuration**: (list[list[list]]) - A list containing the system components. Each component is defined as a list of the bead types of which it is composed. Each bead type is represented by a list where the first entry is the bead name (used in the EOSgroup file) and second entry is the integer number of beads in the component. See tutorial, :ref:`settingup-label`
 * **EOSgroup**: (str) - The filename of .json file containing a dictionary of single group parameters used in the desired equation of state (EOS).
 * **EOScross**: (str) - The filename of .json file containing a dictionary of group-group cross interaction parameters used in the desired equation of state (EOS).
 * **calculation_type**: (str) - Any :func:`calculation_type <despasito.thermodynamics.calculation_types>` that is supported by the thermodynamic module
 * **output_file**: (str) Optional - default: despasito_out.txt.
 * **eos**: (str) Optional - default: ``saft.gamma_mie``. Supported :func:`EOS class <despasito.equations_of_state.initiate_eos>` to be used in thermodynamic computations.
 * **eos_\***: Optional - Any keyword that needs to be passed to an equation of state object should be preceded by "eos\_" to be includes in the eos dictionary. (e.g. num_rings for saft.gamma_mie should be included as eos_num_rings)
 * **\***: Required or optional keywords from the chosen :func:`calculation_type <despasito.thermodynamics.calculation_types>`. See appropriate doc string for more details.


Parameter Fitting Keywords
__________________________
During parameter fitting, **calculation_type** and **bead_configuration** are optional and can be defined in the experimental data dictionaries instead. If these keywords aren't provided in the experimental data dictionary, these higher level keywords are required for the experimental data dictionaries to inherit.


 * **optimization_parameters**: The presence of this keyword signifies that a parameter fitting calculation is requested.

      * **fit_bead**: (str) - Name of bead whose parameters are being fit, must be in bead list of bead_configuration
      * **fit_parameter_names**: (list[str]) - This list contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
      * **parameters_guess**: (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from Eos object.
      * **\*_bounds**: (list[float]) Optional, default is provided by Eos object. By putting a parameter name before the "bounds" suffix, the lower and upper limit of the parameter is defined for the fitting process. Defining the bounds is recommended for rapid convergence.

 * **\***: Experimental data dictionaries may be defined using any keyword, although this key is later used in distinguishing the respective objective function value. Any number of experimental data dictionaries may be included. A keyword is specified as an experimental data structure with the presence of the keyword "data_class_type" and an entry.

      * **data_class_type**: (str) Must be a supported :ref:`data class <data-types>` for fitting
      * **calctype**: (str) Optional - Any :func:`calculation_type <despasito.thermodynamics.calculation_types>` that is supported by the thermodynamic module
      * **file**: (str) Optional - File of experimental data, See fitting :ref:`data class <data-types>` for file headers
      * **bead_configuration**: (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from Eos object. This allows the specified bead to be fit using multiple systems.
      * **weights**: (dict) - A dictionary where each key is the header used in the exp. data file. The value associated with a header can be a list as long as the number of data points to multiply by the objective value associated with each point, or a float to multiply the objective value of this data set.
      * **\***: Optional - Instead of a file of experimental data, the relevant arrays may be included here. All other optional entries for the :func:`calculation_type <despasito.thermodynamics.calculation_types>` may be included.

 * **global_opts**: (dict), Optional - Specify details of global optimization method.

      * **method**: (str), default: 'differential_evolution', Global optimization method used to fit parameters. See :func:`~despasito.parameter_fitting.fit_functions.global_minimization`.
      * **\***: any keyword used by the defined global optimization method.

 * minimizer_opts: (dict), Optional - Dictionary used to define minimization type used by the global optimization method

      * **method**: (str) - Method available to scipy.optimize.minimize
      * **options**: (dict) - This dictionary contains the kwargs available to the chosen method


