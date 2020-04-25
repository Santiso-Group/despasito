"""

Routines for parsing input .json files to dictionaries for program use.

"""

import logging
import json
import numpy as np
import os

logger = logging.getLogger(__name__)

######################################################################
#                                                                    #
#                   Appends path to find data files                  #
#                                                                    #
######################################################################

def append_data_file_path(input_dict, path='.'):
   """
   Appends path to data file(s).

   Parameters
   ----------
   input_dict
       Dictionary of input (json) data 
   path: str
       relative path to append to existing data files 
   """
   if 'EOSgroup' in input_dict:
      input_dict['EOSgroup'] = os.path.join(path, input_dict['EOSgroup'])

   if 'EOScross' in input_dict:
      input_dict['EOScross'] = os.path.join(path, input_dict['EOScross'])

   for key, val in input_dict.items():
      if 'file' in val:
         input_dict[key]['file'] = os.path.join(path, input_dict[key]['file'])

######################################################################
#                                                                    #
#                  Extract Bead Data                                 #
#                                                                    #
######################################################################
def extract_calc_data(input_fname, path='.', **args):

    """
    Parse dictionary from .json input file into a dictionaries.

    Resulting dictionaries are used for creating the equation of state object, and for passing instructions for thermodynamic calculations.

    Parameters
    ----------
    input_fname : str
        The file name of a .json file in the current directory containing (1) the paths to equation of state parameters, (2) :mod:`~despasito.thermodynamics.calc_types` and inputs for thermodynamic calculations (e.g. density options for :func:`~despasito.thermodynamics.calc.PvsRho`).

    Returns
    -------
    eos_dict : dict
        Dictionary of bead definitions and parameters used to later initialize eos object. :func:`despasito.equations_of_state.eos`
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. :func:`despasito.thermodynamics.thermo`
    """

    ## Extract dictionary from input file
    with open(input_fname, 'r') as input_file:
        input_dict = json.loads(input_file.read())

    # Look for data (library) files in the user-supplied path
    append_data_file_path(input_dict, path)

    if "output_file" in input_dict:
        output_file = input_dict["output_file"]
    else:
        output_file = None

    ## Make bead data dictionary for EOS
    #process input file
    if 'beadconfig' in input_dict:
        beads, nui = process_bead_data(input_dict['beadconfig'])
        eos_dict = {'beads':beads,'nui':nui}
    elif "opt_params" in input_dict:
        eos_dict = {}
    else:
        raise ValueError("Bead configuration line is missing for thermodynamic calculation.")

    #read EOS groups file
    with open(input_dict['EOSgroup'], 'r') as f:
        output = f.read()
    eos_dict['beadlibrary'] = json.loads(output)

    #read EOS cross file
    try:
        with open(input_dict['EOScross'], 'r') as f:
            output = f.read()
        eos_dict['crosslibrary'] = json.loads(output)
        logger.info("Cross interaction parameters have been accepted")
    except:
        logger.info("No EOScross file specified")

    try:
        eos_dict['sitenames'] = input_dict['association_site_names']
        logger.info('Association sites have been accepted')
    except:
        logger.info('No association sites specified')

    if "eos" in input_dict:
        eos_dict['eos'] = input_dict["eos"]

    ## Make dictionary of data needed for thermodynamic calculation
    thermo_dict = {}
    for key, value in args.items():
        thermo_dict[key] = value
    # Extract relevant system state inputs
    EOS_dict_keys = ['beadconfig', 'EOSgroup', 'EOScross','association_site_names',"output_file","eos"]
    for key, value in input_dict.items():
        if key not in EOS_dict_keys:
            thermo_dict[key] = value

    if "opt_params" not in thermo_dict:
        logger.info("The following thermo calculation parameters have been provided: {}\n".format((", ".join(thermo_dict.keys()))))
    else: # parameter fitting
        thermo_dict = process_param_fit_inputs(thermo_dict)
        for exp_key in thermo_dict["exp_data"]:
            if "eos_dict" not in thermo_dict["exp_data"][exp_key]:
                thermo_dict["exp_data"][exp_key]["eos_dict"] = {}
            for key, value in eos_dict.items():
                if key not in thermo_dict["exp_data"][exp_key]["eos_dict"]:
                    thermo_dict["exp_data"][exp_key]["eos_dict"][key] = value
        tmp = ""
        for key, value in thermo_dict["exp_data"].items():
            tmp += " {} ({}),".format(key,value["name"])
        logger.info("The bead, {}, will have the parameters {}, fit using the following data:\n {}".format(thermo_dict["opt_params"]["fit_bead"],thermo_dict["opt_params"]["fit_params"],tmp))

    return eos_dict, thermo_dict, output_file

######################################################################
#                                                                    #
#                  Extract Density Plot  Parameters                  #
#                                                                    #
######################################################################
def file2paramdict(filename,delimiter=" "):

    """
    Converted file directly into a dictionary.

    Each line in the input file is a key followed by a value in the resulting dictionary.

    Parameters
    ----------
    filename : str
        File of keys and values
    delimiter : str, Optional, default: " "
        String separating key and value within file

    Returns
    -------
    dictionary : dict
        Resulting dictionary
    """

    dictionary = {}
    with  open(filename, "r") as filedata:
        for line in filedata:
            line.rstrip()
            linearray = line.split(delimiter)
            if len(linearray) == 2:
                try:
                    dictionary[linearray[0]] = eval(linearray[1])
                except:
                    if line[1].isdigit():
                        dictionary[linearray[0]] = float(linearray[1])
                    else:
                        dictionary[linearray[0]] = linearray[1]
    
    return dictionary

######################################################################
#                                                                    #
#                  Make Mole Frac. Matrix                            #
#                                                                    #
######################################################################
def make_xi_matrix(filename):

    """
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
    nui : numpy.ndarray
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

    f = open(filename, 'r').read()
    comp = json.loads(f)
    beads, nui = process_bead_data(comp)
    return xi, beads, nui


######################################################################
#                                                                    #
#                  Process Bead Data                                 #
#                                                                    #
######################################################################
def process_bead_data(bead_data):

    """
    Process system information from input file.

    Parameters
    ----------
    bead_data : list[list]
        List of strings and dictionaries from .json file

    Returns
    -------
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

    #find list of unique beads
    beads = []
    for i in range(len(bead_data)):
        for j in range(len(bead_data[i])):
            if bead_data[i][j][0] not in beads:
                beads.append(bead_data[i][j][0])
    beads.sort()

    nui = np.zeros((len(bead_data), len(beads)))
    for i in range(len(bead_data)):
        for j in range(len(bead_data[i])):
            for k in range(np.size(beads)):
                if bead_data[i][j][0] == beads[k]:
                    nui[i, k] = bead_data[i][j][1]
    return beads, nui

######################################################################
#                                                                    #
#                  Parameter Fitting Data                            #
#                                                                    #
######################################################################
def process_param_fit_inputs(thermo_dict):

    """
    Process parameter fitting information.

    Parameters
    ----------
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is directly from the input file.

        - opt_params (dict) - Parameters used in basin fitting algorithm

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
            - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
            - *_bounds (list[float]), Optional - This list contains the minimum and maximum of the parameter from a parameter listed in fit_params, represented in place of the asterisk. See input file instructions for more information.
            - beadparams0 (list[float]), Optional - Initial guess in parameters being fit. Should be the same length at fit_params and contain a reasonable guess for each parameter. If this is not provided, a guess is made based on the type of parameter from eos object.

        - *name* (dict) - Dictionary of a data set that the parameters are fit to. Each dictionary is added to the exp_data dictionary before being passed to the fitting algorithm. Each *name* is used as the key in exp_data. *name* is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - file (str) - 
            - datatype (str) - One of the supported data type objects to fit parameters
            - calctype (str) - One of the supported thermo calculation types that would be associated with the chosen datatype

    Returns
    -------
    new_thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is reformatted and includes imported data. Dictionary values below are altered before being passed on, all other key and value sets are blindly passed on.

        - opt_params (dict) - Parameters used in basin fitting algorithm

            - fit_bead (str) - Name of bead whose parameters are being fit, should be in bead list of beadconfig
            - fit_params (list[str]) - This list of contains the name of the parameter being fit (e.g. epsilon). See EOS documentation for supported parameter names. Cross interaction parameter names should be composed of parameter name and the other bead type, separated by an underscore (e.g. epsilon_CO2).
            - bounds (numpy.ndarray) - List of lists of length two, of length equal to fit_params. If no bounds were given then the default parameter boundaries are [0,1e+4], else bounds given as *_bounds in input file are used.
            - beadparams0 (list[float]), Optional - Initial guess in parameter. If one is not provided, a guess is made based on the type of parameter from eos object.

        - exp_data (dict) - This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details.

            - name (obj) - One of the supported data type objects to fit parameters
    """

    # Initial new dictionary that will have dictionary for extracted data
    new_thermo_dict = {"exp_data":{}}

    for key, value in thermo_dict.items():
        if key == "opt_params": 
            new_opt_params = {}
            keys_del = []
            new_opt_params["bounds"] = [[0,1e+4] for x in range(len(value["fit_params"]))]
            for key2, value2 in value.items():
                if key2 == "fit_bead":
                    new_opt_params["fit_bead"] = value["fit_bead"]
                elif key2 == "fit_params":
                    new_opt_params["fit_params"] = value["fit_params"]
                elif key2 == "beadparams0":
                    new_opt_params["beadparams0"] = value["beadparams0"]
                elif "bounds" in key2:
                    tmp  = key2.replace("_bounds","")
                    if tmp in value["fit_params"]:
                        ind = value["fit_params"].index(tmp)
                        new_opt_params["bounds"][ind] = value2
                    else:
                        logger.warning("Bounds for parameter type '{}' were given, but this parameter is not defined to be fit.".format(tmp))
                else:
                    continue
                keys_del.append(key2)
            for key2 in keys_del:
                value.pop(key2,None)

            if value:
               logger.info("The opt_params keys: {}, were not used.".format(", ".join(list(value.keys()))))
            new_thermo_dict[key] = new_opt_params

        elif (type(value) == dict and "datatype" in value):
            new_thermo_dict["exp_data"][key] = process_exp_data(value)

        else:
            new_thermo_dict[key] = value

    # Move opt_params values to thermo_dict
    keys = ["bounds","minimizer_dict"]
    for key in keys:
        if key in new_thermo_dict["opt_params"]:
            new_thermo_dict[key] = new_thermo_dict["opt_params"][key]
            new_thermo_dict["opt_params"].pop(key,None)

    test1 = set(["exp_data","opt_params"]).issubset(list(new_thermo_dict.keys()))
    test2 = set(["fit_bead","fit_params"]).issubset(list(new_thermo_dict["opt_params"].keys()))
    if not all([test1,test2]):
        raise ValueError("An exp_data and opt_params dictionary with, fit_beads and fit_params must be given")

    return new_thermo_dict

######################################################################
#                                                                    #
#                  Process Experimental Data                         #
#                                                                    #
######################################################################
def process_exp_data(exp_data_dict):

    """
    Convert experimental data dictionary into form used by parameter fitting module. 

    Note that there should be one dictionary per data set. All data is extracted from data files.

    Parameters
    ----------
    exp_data : dict
       This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details. Dictionary values below are altered before being passed on, all other key and value sets are blindly passed on.

       - datatype (str) - One of the supported data type objects to fit parameters
       - file (str) - File name in current working directory, or path to desired experimental data. See experimental data page for examples of acceptable format.

    Returns
    -------
    exp_data : dict
        Reformatted dictionary of experimental data. This dictionary is made up of a dictionary for each data set that the parameters are fit to. Each key is an arbitrary string used to identify the data set and used later in reporting objective function values during the fitting process. See data type objects for more details. Dictionary values below are altered from input dictionary, all other key and value sets are blindly passed on, or extracted from data file with process_exp_data_file function.

        - name (str) - One of the supported data type objects to fit parameters
    """

    exp_data = {}
    for key, value in exp_data_dict.items():
        if key == "datatype":
            exp_data["name"] = value
        elif key == "file":
            file_dict = process_exp_data_file(value)
            exp_data.update(file_dict)
        elif key == "beadconfig":
            beads, nui = process_bead_data(value)
            exp_data["eos_dict"] = {'beads':beads,'nui':nui}
        else:
            exp_data[key] = value

    return exp_data

######################################################################
#                                                                    #
#                  Process Experimental Data                         #
#                                                                    #
######################################################################
def process_exp_data_file(fname):

    """
    Import data file and convert columns into dictionary entries.

    The headers in the file are the dictionary keys. The top line is skipped, and column headers are the second line. Note that column headers should be thermo properties (e.g. T, P, x1, x2, y1, y2) without suffixes. Mole fractions x1, x2, ... should be in the same order as in the beadconfig line of the input file. No mole fractions should be left out.

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
        data = np.transpose(np.genfromtxt(fname, delimiter=',',names=True,skip_header=1))
    except:
        raise ValueError("Cannot import '{}', Check data file formatting.".format(fname))
    file_dict = {name:data[name] for name in data.dtype.names}
    
    # Sort through properties
    key_del = []
    xi, yi, zi = [[],[],[]]
    for key, value in file_dict.items():
       if "#" in key:
           key.replace("#","").replace(" ","")

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
        file_dict.pop(key,None)

    if xi:
        file_dict["xi"] = np.transpose(np.array([np.array(x) for x in xi]))

    if yi:
        file_dict["yi"] = np.transpose(np.array([np.array(y) for y in yi]))

    if zi:
        file_dict["zi"] = np.transpose(np.array([np.array(z) for z in zi]))

    return file_dict



