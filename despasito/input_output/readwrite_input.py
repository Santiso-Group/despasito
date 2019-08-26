"""

    Routines for pasing input files from .json files to dictionaries and extracting relaven information for program use, as well as write properly structures .json files for later calculations.
    
"""

import json
import collections
import numpy as np

######################################################################
#                                                                    #
#                  Extract Bead Data                                 #
#                                                                    #
######################################################################
def process_commandline(args):

    """
    Interprets command line arguments when package is called with `python -m despasito`. The minimum is the name of the input file.
    
    Optionally the flag --dens_params or -dp may be added followed by the name of a file containing optional parameters for the density calculation.

    Parameters
    ----------
    args : list
        Command line arguments of despasito package

    Returns
    -------
    calc_type : str
        String of supported thermodynamic calculation type in this package
    eos_dict : dict
        Dictionary of bead definitions and parameters used to later initialize eos object
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations 
    """

    kwargs = {}
    if len(args) > 2:
        if ("--dens_params" in args or "-dp" in args):
            try:
                ind = args.index("--dens_params")
            except:
                ind = args.index("-dp")
                
            kwargs['density_fname'] = args[ind+1]
    input_fname = args[1]

    calctype, eos_dict, thermo_dict = extract_calc_data(input_fname,**kwargs)
    
    return calctype, eos_dict, thermo_dict 

######################################################################
#                                                                    #
#                  Extract Bead Data                                 #
#                                                                    #
######################################################################
def extract_calc_data(input_fname,density_fname='input_density_params.txt'):

    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    input_fname : str
        The file name of a .json file in the current directory containing (1) the paths to equation of state parameters, (2) calculation type and inputs for thermodynamic calculations.
        .. todo:: Add link to available thermodynamic calculations
    density_fname : str, Optional, default: input_density_params.txt
        This file is converted directly into a dictionary where each line is a key followed by a value, with a space in between. 
        .. todo:: Add link to available density options

    Returns
    -------
    calc_type : str
        String of supported thermodynamic calculation type in this package
    eos_dict : dict
        Dictionary of bead definitions and parameters used to later initialize eos object
    thermo_dict : dict
        Dictionary of instructions for thermodynamic calculations 
    """

    ## Extract dictionary from input file
    input_file = open(input_fname, 'r').read()
    input_dict = json.loads(input_file)

    ## Check for type of calculation
    try:
        calctype = input_dict['calculation_type']
    except:
        calctype = 'none'
        raise Exception('No calculation type specified')
    
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
        print("Note: Cross interaction parameters have been accepted")
    except:
        print("Note: No SAFTcross file specified")

    try:
        eos_dict['sitenames'] = input_dict['association_site_names']
        print('Note: Association sites have been accepted')
    except:
        print('Note: No association sites specified')

    ## Make dictionary of data needed for thermodynamic calculation
    thermo_dict = {}
    # Extract relevent system state inputs
    EOS_dict_keys = ['beadconfig', 'SAFTgroup', 'SAFTcross']
    for key, value in input_dict.items():
        if key not in EOS_dict_keys:
            thermo_dict[key] = value
    tmp = ", ".join(thermo_dict.keys())
    print("Note: The following thermo calculation parameters have been provided: %s\n" % tmp)

    try:
        rho_dic = file2paramdict(density_fname)
        thermo_dict['rhodict'] = rho_dic
        print('Note: Density plot parameters have been accepted from '+density_fname)
    except:
        pass

    return calctype, eos_dict, thermo_dict

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

    dictionary = {}
    with  open(filename, "r") as filedata:
        print(filedata)
        for line in filedata:
            line.rstrip()
            linearray = line.split(delimiter)
            print(linearray)
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
        Mole fraction of component, only relevent for parameter fitting
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.array
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

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
        Mole fraction of component, only relevent for parameter fitting
    beads : list[str]
        List of unique bead names used among components
    nui : numpy.array
        Array of number of components by number of bead types. Defines the number of each type of group in each component.
    """

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

