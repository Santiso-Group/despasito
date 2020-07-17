# ChangeLog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2019-12-16
### Added
 - CHANGELOG.md file
 - Start-up tutorial in readthedocs

### Changed
 - Documentation bug fixes
 - SAFT-gamma-mie and peng-robinson equations of state
 - Support for fitting with: TVLE, saturation, and liquid property data to binary systems

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

## [0.1.0] - 2020-06-13
### Added
 - python flag is added to command line to allow pure python implementation, this is not recommended in practice for association site calculations.
### Changed
 - Refactored SAFT EOS. Now general SAFT class contains association site contribution, and calculates ideal contribution from factory method. This SAFT class has generalized method to update parameters. Other Helmholtz contributions may now be specified and initiated by this general class.
 - gamm_mie and gamma_sw are classes for specific SAFT EOS types, each handling their required parameters to calculation monomer and chain contributions to the Helmholtz energy.
 - Input files now used nanometers as a length scale instead of meters.
 - Density and energy parameters are now scaled so that all calculations are within precision limits, removing precision loss in association site calculations.
 - Newer calc_Xika algorithm is implemented in FORTRAN, improving performance.
