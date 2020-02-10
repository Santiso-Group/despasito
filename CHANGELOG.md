# ChangeLog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2019-12-16
### Added
 - CHANGELOG.md file
 - Start-up tutorial in readthedocs

### Changed
 - Documentation bug fixes
 - SAFT-gamma-mie and peng-robinson equations of state
 - Support for fitting with: TVLE, saturation, and liquid property data to binary systems

## [0.1.1] - 2020-02-10
### Added
 - docs updated with parameter fitting tutorial
 - thermodynamic modules now supports calculation of hildebrand solubility parameters
 - fitting module now supports fitting to solubility parameters
 - global parameter fitting methods differential_evolution and brute

### Changed
 - README.md updated with pip install instructions
 - docs updated with pip install instructions
 - separated basinhopping algoithm from overall parameters fitting algoirthm to allow other options
 - updated method of calculating parameter fitting objective functions
 - updated rhov calculation
 - updated method of weighting experimental data in parameter fitting to use dictionaries

