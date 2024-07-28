# ChangeLog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.3.0] - 2024-07-28
### Add
 - Error for abnormally large array sizes for density
 - JOSS paper

### Changed
 - Move ownership to organization, Santiso-Group from jaclark5; pdatd links.
 - Update installation instructions to ensure compilation of cython
 - Fix cython Xika_6
 - Fix association site bonding volume
 - Improved saft gamma mie association site processing
 - Remove fortran modules


## [0.2.0] - 2021-04-05
### Add
 - Require command line option, --console, to print to standard output
 - Add command line flag to use pure python implementation
 - Compilation/Implementation flags consolidated into class
 - Allow fitting algorithms to use worker processes
 - initiate_logger function to allow logs to be printed to the screen when imported as a python library
 - Check for parameter boundaries
 - DESPASITO logo
 - Custom grid_minimization method where a grid of starting points can be minimized or a grid of parameters change be held constant while those remaining are minimized. 
 - Annotated saft_variant_example.py to guide a developer in adding a new variant of SAFT.
 - Option to calculate SAFT association sites from bond distances instead of bonding volumes.
 - EOS combining rule factory function and referenced module
 - Ability to use MAPSCI package as a plug-in to estimate cross interaction parameters.
 - Global optimization option to perform a single objective function evaluation
 - Added options for objective function form
 - Infrastructure for global optimization constraints 
 - Tests for logging 
 - Tests: Added single_objective tests for each data class: TLVE, saturation_properties, solubility_parameter, liquid_density
 - Penalty in parameter optimization for thermo calculations that fail
 - Option in flash calculation to set maximum mole fraction
 - In flash calculation, added a count on the number of Ki resets to first switch Ki values and then to cancel if not converging
 - Made intermediate parameter file for shgo and differential_evolution
 - kwargs to calc_saturation_pressure, allow a set minimum pressure and acceptance tolerance
 - kwargs to thermodynamics.calc.pressure_vs_volume to remove need for extrapolation to zero pressure
 - Updated thermodynamics.calc.calc_Prange_xi to use Pmin and Pmax as hard boundaries and more effectively determine if a super critical fluid is desired
 - Add tested for Cython and numba implementations
 - Contributing description to documentation
 - Consolidates redundant checks into module in utils directory
 - Activity coefficient calculation_type, examples, and test
 - THANKS.txt and CONTRIBUTING.md

### Changed
 - Bug fixes
 - Add python 3.8 to allowed list
 - changed length scale in SAFT EOS to be nanometers instead of meters.
 - Changed variable and function names to be more pythonic
 - Make differential evolution the default parameter fitting algorithm
 - Separated command line parser from implementation
 - Consolidated common methods into EOS and data_class_type interfaces
 - Switch continuous integration from Travis and AppVeyor to GitHub actions
 - Allow check that Cython is installed before testing or use
 - setup.py to check for Fortran compiler and handle failed Fortran compilation 
 - Updated calc_type.py functions to easily pass options to supporting functions. 
 - Update supporting functions in calc.py to accept all options and report those that aren't used. 
 - Update except statements to execute base class, Exception
 - Removed unused Fortran modules
 - Remove UML diagrams


## [0.1.0] - 2020-06-13
### Added
 - python flag is added to command line to allow pure python implementation, this is not recommended in practice for association site calculations.
### Changed
 - Refactored SAFT EOS. Now general SAFT class contains association site contribution, and calculates ideal contribution from factory method. This SAFT class has generalized method to update parameters. Other Helmholtz contributions may now be specified and initiated by this general class.
 - gamm_mie and gamma_sw are classes for specific SAFT EOS types, each handling their required parameters to calculation monomer and chain contributions to the Helmholtz energy.
 - Input files now used nanometers as a length scale instead of meters.
 - Density and energy parameters are now scaled so that all calculations are within precision limits, removing precision loss in association site calculations.
 - Newer calc_Xika algorithm is implemented in FORTRAN, improving performance.


## [0.1.0] - 2020-02-10
### Added
 - docs updated with parameter fitting tutorial
 - thermodynamic modules now supports calculation of Hildebrand solubility parameters
 - fitting module now supports fitting to solubility parameters
 - global parameter fitting methods differential_evolution and brute

### Changed
 - README.md updated with pip install instructions
 - docs updated with pip install instructions
 - separated basinhopping algorithm from overall parameters fitting algorithm to allow other options
 - updated method of calculating parameter fitting objective functions
 - updated rhov calculation
 - updated method of weighting experimental data in parameter fitting to use dictionaries


## [0.1.0] - 2019-12-16
### Added
 - CHANGELOG.md file
 - Start-up tutorial in readthedocs

### Changed
 - Documentation bug fixes
 - SAFT-gamma-mie and peng-robinson equations of state
 - Support for fitting with: TVLE, saturation, and liquid property data to binary systems