"""

Routines for passing input files from .json files to dictionaries and extracting relevant information for program use, as well as write properly structures .json files for later calculations.
    
.. todo::
    * extract_calc_data input_fname: Add link to available thermodynamic calculations
    * extract_calc_data density_fname: Add link to available density options
"""

import logging
import json
import collections
import numpy as np
import os

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
   if 'SAFTgroup' in input_dict:
      input_dict['SAFTgroup'] = os.path.join(path, input_dict['SAFTgroup'])

   if 'SAFTcross' in input_dict:
      input_dict['SAFTcross'] = os.path.join(path, input_dict['SAFTcross'])

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
    Uses dictionary from .json input file to process and divide information into two dictionaries, one for creating the equation of state, and one for the thermodynamic calculations.

    Parameters
    ----------
    input_fname : str
        The file name of a .json file in the current directory containing (1) the paths to equation of state parameters, (2) calculation type and inputs for thermodynamic calculations.

    Returns
    -------
    eos_dict : dict
        Dictionary of bead definitions and parameters used to later initialize eos object
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting
    """

    logger = logging.getLogger(__name__)

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
    xi, beads, nui = process_bead_data(input_dict['beadconfig'])
    eos_dict = {'xi':xi,'beads':beads,'nui':nui}

    #read SAFT groups file
    with open(input_dict['SAFTgroup'], 'r') as f:
        output = f.read()
    eos_dict['beadlibrary'] = json.loads(output)

    #read SAFT cross file
    try:
        with open(input_dict['SAFTcross'], 'r') as f:
            output = f.read()
        eos_dict['crosslibrary'] = json.loads(output)
        logger.info("Cross interaction parameters have been accepted")
    except:
        logger.info("No SAFTcross file specified")

    try:
        eos_dict['sitenames'] = input_dict['association_site_names']
        logger.info('Association sites have been accepted')
    except:
        logger.info('No association sites specified')

    ## Make dictionary of data needed for thermodynamic calculation
    thermo_dict = {}
    # Extract relevant system state inputs
    EOS_dict_keys = ['beadconfig', 'SAFTgroup', 'SAFTcross','association_site_names',"output_file"]
    for key, value in input_dict.items():
        if key not in EOS_dict_keys:
            thermo_dict[key] = value

    if "opt_params" not in thermo_dict:
        logger.info("The following thermo calculation parameters have been provided: {}\n".format((", ".join(thermo_dict.keys()))))
    else: # parameter fitting
        thermo_dict = process_param_fit_inputs(thermo_dict)
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
    Converted file directly into a dictionary where each line is a key followed by a value.

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

    logger = logging.getLogger(__name__)

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
#                  Extract Saft Parameters                           #
#                                                                    #
######################################################################
def write_SAFTgroup(library, filename):

    """
    Sort and export dictionary of input SAFT parameters into .json file.

    Parameters
    ----------
    library : dict
        Dictionary of parameters to be sorted and exported 

    Returns
    -------
    filename : str
        Filename (with or without path) of .json file of parameters
    """

    logger = logging.getLogger(__name__)

    #sort and write SAFT dic
    for i in library:
        library[i] = collections.OrderedDict(sorted(list(library[i].items()), key=lambda tup: tup[0].lower()))
    f = open(filename, 'w')
    json.dump(library, f, indent=4)


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

    logger = logging.getLogger(__name__)

    f = open(filename, 'r').read()
    comp = json.loads(f)
    xi, beads, nui = process_bead_data(comp)
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
    xi : list[float]
        Mole fraction of component, only relevant for parameter fitting
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.ndarray
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

    logger = logging.getLogger(__name__)

    #find list of unique beads
    beads = []
    xi = np.zeros(len(bead_data))
    for i in range(len(bead_data)):
        xi[i] = bead_data[i][0]
        for j in range(len(bead_data[i][1])):
            if bead_data[i][1][j][0] not in beads:
                beads.append(bead_data[i][1][j][0])
    beads.sort()

    nui = np.zeros((np.size(xi), len(beads)))
    for i in range(len(bead_data)):
        for j in range(len(bead_data[i][1])):
            for k in range(np.size(beads)):
                if bead_data[i][1][j][0] == beads[k]:
                    nui[i, k] = bead_data[i][1][j][1]
    return xi, beads, nui

######################################################################
#                                                                    #
#                  Parameter Fitting Data                            #
#                                                                    #
######################################################################
def process_param_fit_inputs(thermo_dict):

    """
    Process parameter fitting information and data formatting

    Parameters
    ----------
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is directly from the input file.

    Returns
    -------
    new_thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations or parameter fitting. This dictionary is reformatted and includes imported data.
    """

    logger = logging.getLogger(__name__)

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
                elif "bounds" in key2:
                    tmp  = key2.replace("_bounds","")
                    ind = value["fit_params"].index(tmp)
                    new_opt_params["bounds"][ind] = value2
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

        elif key == "beadparams0":
            new_thermo_dict[key] = value
        else:
            new_thermo_dict[key] = value

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
    Process raw experimental data dictionary into one that can be used by the parameter fitting module. Note that there should be one dictionary per data set. All data is extracted from data files.

    Parameters
    ----------
    exp_data_dict : dict
        Raw dictionary of experimental data information, there is one dictionary per set

    Returns
    -------
    exp_data : dict
        Reformatted dictionary of experimental data
    """

    logger = logging.getLogger(__name__)

    exp_data = {}
    for key, value in exp_data_dict.items():
        if key == "datatype":
            exp_data["name"] = value
        elif key == "file":
            file_dict = process_exp_data_file(value)
            exp_data.update(file_dict)
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
    Import data file and convert columns into dictionary entries, where the header is the dictionary key. The top line is skipped, and column headers are the second line. Note that column headers should be thermo properties (e.g. T, P, x1, x2, y1, y2) without suffixes. Mole fractions x1, x2, ... should be in the same order as in the beadconfig line of the input file. No mole fractions should be left out.

    Parameters
    ----------
    fname : str
        File name or path to experimental data file

    Returns
    -------
    file_dict : dict
        Dictionary of experimental data from file.
    """

    logger = logging.getLogger(__name__)

    data = np.genfromtxt(fname, delimiter=',',names=True,skip_header=1).T
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

    if xi: file_dict["xi"] = np.array([np.array(x) for x in xi]).T
    if yi: file_dict["yi"] = np.array([np.array(y) for y in yi]).T
    if zi: file_dict["zi"] = np.array([np.array(z) for z in zi]).T

    return file_dict

######################################################################
#                                                                    #
#                  Write Thermodynamic Ouput                         #
#                                                                    #
######################################################################
def writeout_dict(output_dict,calctype,output_file="thermo_output.txt"):
    """
    Import dictionary of both input and output data to produce a file. A line in the top clarifies the calculation type done.

    Parameters
    ----------
    output_dict : dict
        Dictionary of given and calculated information from thermodynamic module
    output_file : str, Optional, default: thermo_output.txt
        Name of output file

    Returns
    -------
    File of data saved to current directory
    """

    # Define units
    units = {"T":"K","P":"Pa","Psat":"Pa","rhol":"mol/m^3","rhov":"mol/m^3"}

    # Make comment line
    comment = "# This data was generated in DESPASITO using the thermodynamic calculation: "+calctype
    
    # Make results matrix
    print(output_dict)
    keys = []
    matrix = []
    for key, value in output_dict.items():
        tmp_matrix = np.array(value).T
        if len(tmp_matrix.shape)==1:
            keys.append(key)
            matrix.append(value)
        else:
            for i in range(len(tmp_matrix)):
                keys.append(key+str(i+1))
                matrix.append(tmp_matrix[i])
    matrix = np.array(matrix).T

    # Make header line
    header = "#"
    for key in keys:
        if key in units:
            unit = " [{}]".format(units[key])
        else:
            unit = ""
        header += " {}{},".format(key,unit)

    # Write to file
    with open(output_file,"w") as f:
        f.write(comment+"\n")
        f.write(header+"\n")
        for row in matrix:
            f.write((' {},' * len(row)).format(*row)+"\n")
            



