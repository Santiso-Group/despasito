""" Routines for parsing input .json files to dictionaries for program use.
"""

import logging
import json
import numpy as np
import os

logger = logging.getLogger(__name__)


def append_data_file_path(input_dict, path="."):
    r"""
   Appends path to data file(s).

   Parameters
   ----------
   input_dict
       Dictionary of input (json) data 
   path: str
       relative path to append to existing data files 
   """
    if "EOSgroup" in input_dict:
        input_dict["EOSgroup"] = os.path.join(path, input_dict["EOSgroup"])

    if "EOScross" in input_dict:
        input_dict["EOScross"] = os.path.join(path, input_dict["EOScross"])

    for key, val in input_dict.items():
        if isinstance(val, dict) and "file" in val:
            input_dict[key]["file"] = os.path.join(path, input_dict[key]["file"])


def json_to_dict(filename):
    r"""
    Extract json file as a dictionary

    Parameters
    ----------
    filename : str
        File name and path leading to json file location
  
    Returns
    -------
    dictionary : dict
        Dictionary resulting from json file
    """

    with open(filename, "r") as f:
        output = f.read()
    dictionary = json.loads(output)

    return dictionary


def extract_calc_data(input_fname, path=".", **thermo_dict):

    r"""
    Parse dictionary from .json input file into a dictionaries.

    Resulting dictionaries are used for creating the equation of state object, and for passing instructions for thermodynamic calculations.

    Parameters
    ----------
    input_fname : str
        The file name of a .json file in the current directory containing (1) the paths to equation of state parameters, (2) :mod:`~despasito.thermodynamics.calculation_types` and inputs for thermodynamic calculations (e.g. density options for :func:`~despasito.thermodynamics.calc.pressure_vs_volume_arrays`).
    path : str, Optional, default="."
        Path to `input_fname`
    thermo_dict
        Additional keyword arguments

    Returns
    -------
    eos_dict : dict
        Dictionary of bead definitions and parameters used to later initialize Eos object. :func:`despasito.equations_of_state.initiate_eos`
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. :func:`despasito.thermodynamics.thermo`
    output_file : str
        Output from calculation. Default is None, but an alternative can be defined as output_file keyword argument.
    """

    ## Extract dictionary from input file
    input_dict = json_to_dict(input_fname)

    # Look for data (library) files in the user-supplied path
    append_data_file_path(input_dict, path)

    if "output_file" in input_dict:
        output_file = input_dict["output_file"]
    else:
        output_file = None

    ## Make bead data dictionary for EOS
    # process input file
    if "bead_configuration" in input_dict:
        beads, molecular_composition = process_bead_data(
            input_dict["bead_configuration"]
        )
        eos_dict = {"beads": beads, "molecular_composition": molecular_composition}
    elif "optimization_parameters" in input_dict:
        eos_dict = {}
    else:
        raise ValueError(
            "Bead configuration line is missing for thermodynamic calculation."
        )

    # read EOS groups file
    eos_dict["bead_library"] = json_to_dict(input_dict["EOSgroup"])
    if "EOScross" in input_dict:
        eos_dict["cross_library"] = json_to_dict(input_dict["EOScross"])
        logger.info("Cross interaction parameters have been accepted")
    else:
        logger.info("No EOScross file specified")

    # Extract relevant system state inputs
    EOS_dict_keys = [
        "bead_configuration",
        "EOSgroup",
        "EOScross",
        "output_file"
    ]
    for key, value in input_dict.items():
        if key.startswith("eos_"):
            new_key = "_".join(key.split("_")[1:])
            eos_dict[new_key] = input_dict[key]
        elif key == "eos":
            eos_dict["eos"] = input_dict["eos"]
        elif key not in EOS_dict_keys:
            thermo_dict[key] = value

    for key in ["numba", "cython", "python"]:
        if key in thermo_dict:
            eos_dict[key] = thermo_dict[key]
            del thermo_dict[key]

    if "optimization_parameters" not in thermo_dict:
        logger.info(
            "The following thermo calculation parameters have been provided: {}\n".format(
                (", ".join(thermo_dict.keys()))
            )
        )
    else:  # parameter fitting
        thermo_dict = process_param_fit_inputs(thermo_dict)
        for exp_key in thermo_dict["exp_data"]:
            if "eos_dict" not in thermo_dict["exp_data"][exp_key]:
                thermo_dict["exp_data"][exp_key]["eos_dict"] = {}
            for key, value in eos_dict.items():
                if key not in thermo_dict["exp_data"][exp_key]["eos_dict"]:
                    thermo_dict["exp_data"][exp_key]["eos_dict"][key] = value
        tmp = ""
        for key, value in thermo_dict["exp_data"].items():
            tmp += " {} ({}),".format(key, value["data_class_type"])
        logger.info(
            "The bead, {}, will have the parameters {}, fit using the following data:\n {}".format(
                thermo_dict["optimization_parameters"]["fit_bead"],
                thermo_dict["optimization_parameters"]["fit_parameter_names"],
                tmp,
            )
        )

    return eos_dict, thermo_dict, output_file


def file2paramdict(filename, delimiter=" "):

    r"""
    Converted file directly into a dictionary.

    Each line in the input file is a key followed by a value in the resulting dictionary.

    Parameters
    ----------
    filename : str
        File of keys and values
    delimiter : str, Optional, default=" "
        String separating key and value within file

    Returns
    -------
    dictionary : dict
        Resulting dictionary
    """

    dictionary = {}
    with open(filename, "r") as filedata:
        for line in filedata:
            line.rstrip()
            linearray = line.split(delimiter)
            if len(linearray) == 2:
                try:
                    dictionary[linearray[0]] = eval(linearray[1])
                except Exception:
                    if line[1].isdigit():
                        dictionary[linearray[0]] = float(linearray[1])
                    else:
                        dictionary[linearray[0]] = linearray[1]

    return dictionary


def make_xi_matrix(filename):

    r"""
    Extract system information from input .json file of parameters.

    Parameters
    ----------
    filename : str
        Name of .json file containing system and thermodynamic information. This function deals with the system information.

    Returns
    -------
    xi : list[float]
        Mole fraction of component, only relevant for parameter fitting
    beads : list[str]
        List of unique bead names used among components
    molecular_composition : numpy.ndarray
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

    comp = json_to_dict(filename)
    beads, molecular_composition = process_bead_data(comp)

    return xi, beads, molecular_composition


def process_bead_data(bead_data):

    r"""
    Process system information from input file.

    Parameters
    ----------
    bead_data : list[list]
        List of strings and dictionaries from .json file

    Returns
    -------
    beads : list[str]
        List of unique bead names used among components
    molecular_composition : numpy.ndarray
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

    # find list of unique beads
    beads = []
    for i in range(len(bead_data)):
        for j in range(len(bead_data[i])):
            if bead_data[i][j][0] not in beads:
                beads.append(bead_data[i][j][0])
    beads.sort()

    molecular_composition = np.zeros((len(bead_data), len(beads)))
    for i in range(len(bead_data)):
        for j in range(len(bead_data[i])):
            for k in range(np.size(beads)):
                if bead_data[i][j][0] == beads[k]:
                    molecular_composition[i, k] = bead_data[i][j][1]
    return beads, molecular_composition


def process_param_fit_inputs(thermo_dict):

    r"""
    Process parameter fitting information.

    Parameters
    ----------
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is directly from the input file.

        - optimization_parameters (dict) - Parameters used in basin fitting algorithm

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of bead_configuration
            - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
            - \*_bounds (list[float]), Optional - This list contains the minimum and maximum of the parameter from a parameter listed in fit_parameter_names, represented in place of the asterisk. See input file instructions for more information.
            - parameters_guess (list[float]), Optional - Initial guess in parameters being fit. Should be the same length at fit_parameter_names and contain a reasonable guess for each parameter. If this is not provided, a guess is made based on the type of parameter from Eos object.

        - *name* (dict) - Dictionary of a data set that the parameters are fit to. Each dictionary is added to the exp_data dictionary before being passed to the fitting algorithm. Each *name* is used as the key in exp_data. *name* is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - file (str) - File of experimental data 
            - data_class_type (str) - One of the supported data type objects to fit parameters
            - calculation_type (str) - One of the supported thermo calculation types that would be associated with the chosen data_class_type

    Returns
    -------
    new_thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is reformatted and includes imported data. Dictionary values below are altered before being passed on, all other key and value sets are blindly passed on.

        - optimization_parameters (dict) - Parameters used in basin fitting algorithm

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of bead_configuration
            - fit_parameter_names (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
            - parameters_guess (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from Eos object.
            - \*_bounds (list[float]), Optional - This list contains the minimum and maximum of the parameter from a parameter listed in fit_parameter_names, represented in place of the asterisk. See input file instructions for more information.

        - exp_data (dict) - This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - data_class_type (obj) - One of the supported data type objects to fit parameters
    """

    # Initial new dictionary that will have dictionary for extracted data
    new_thermo_dict = {"exp_data": {}}

    for key, value in thermo_dict.items():
        if isinstance(value, dict) and "data_class_type" in value:
            new_thermo_dict["exp_data"][key] = process_exp_data(value)
        else:
            new_thermo_dict[key] = value

    test1 = set(["exp_data", "optimization_parameters"]).issubset(
        list(new_thermo_dict.keys())
    )
    test2 = set(["fit_bead", "fit_parameter_names"]).issubset(
        list(new_thermo_dict["optimization_parameters"].keys())
    )
    if not all([test1, test2]):
        raise ValueError(
            "An exp_data dictionary (dictionary with 'data_class_type' key) as well as an optimization_parameters dictionary with 'fit_bead' and 'fit_parameter_names' must be provided."
        )

    return new_thermo_dict


def process_exp_data(exp_data_dict):

    r"""
    Convert experimental data dictionary into form used by parameter fitting module. 

    Note that there should be one dictionary per data set. All data is extracted from data files.

    Parameters
    ----------
    exp_data : dict
       This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details. Dictionary values below are altered before being passed on, all other key and value sets are blindly passed on.

       - data_class_type (str) - One of the supported data type objects to fit parameters
       - file (str) - File name in current working directory, or path to desired experimental data. See experimental data page for examples of acceptable format.

    Returns
    -------
    exp_data : dict
        Reformatted dictionary of experimental data. This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details. Dictionary values below are altered from input dictionary, all other key and value sets are blindly passed on, or extracted from data file with process_exp_data_file function.

        - data_class_type (str) - One of the supported data type objects to fit parameters
    """

    exp_data = {}
    for key, value in exp_data_dict.items():
        if key == "data_class_type":
            exp_data["data_class_type"] = value
        elif key == "file":
            file_dict = process_exp_data_file(value)
            exp_data.update(file_dict)
        elif key == "bead_configuration":
            beads, molecular_composition = process_bead_data(value)
            exp_data["eos_dict"] = {
                "beads": beads,
                "molecular_composition": molecular_composition,
            }
        else:
            exp_data[key] = value

    return exp_data


def process_exp_data_file(fname):

    r"""
    Import data file and convert columns into dictionary entries.

    The headers in the file are the dictionary keys. The top line is skipped, and column headers are the second line. Note that column headers should be thermo properties (e.g. T, P, x1, x2, y1, y2) without suffixes. Mole fractions x1, x2, ... should be in the same order as in the bead_configuration line of the input file. No mole fractions should be left out.

    Parameters
    ----------
    fname : str
        File name or path to experimental data file

    Returns
    -------
    file_dict : dict
        Dictionary of experimental data from file.
    """

    try:
        data = np.transpose(
            np.genfromtxt(fname, delimiter=",", names=True, skip_header=1)
        )
    except Exception:
        raise ValueError(
            "Cannot import '{}', Check data file formatting.".format(fname)
        )
    file_dict = {name: data[name] for name in data.dtype.names}

    # Sort through properties
    key_del = []
    xi, yi, zi = [[], [], []]
    for key, value in file_dict.items():
        if "#" in key:
            key.replace("#", "").replace(" ", "")

        # Assuming mole fractions are listed starting at x1 and continue in order
        if key.startswith("x"):
            xi.append(value)
        elif key.startswith("y"):
            yi.append(value)
        elif key.startswith("z"):
            zi.append(value)
        else:
            continue
        key_del.append(key)

    for key in key_del:
        file_dict.pop(key, None)

    if xi:
        file_dict["xi"] = np.transpose(np.array([np.array(x) for x in xi]))

    if yi:
        file_dict["yi"] = np.transpose(np.array([np.array(y) for y in yi]))

    if zi:
        file_dict["zi"] = np.transpose(np.array([np.array(z) for z in zi]))

    return file_dict
