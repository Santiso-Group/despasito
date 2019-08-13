"""
    despasito
    DESPASITO: Determining Equilibrium State and Parameters Applied to SAFT, Intended for Thermodynamic Output
    
    Routines for pasing input files from .json files to dictionaries and extracting relaven information.
    
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
        thermo_dict['rhodic'] = rho_dic
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

    dictionary = {}
    with  open(filename, "r") as filedata:
        for line in filedata:
            line.rstrip()
            linearray = line.split(':')
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
