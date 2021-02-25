"""

Routines for writing .txt and .json output files files from dictionaries.

"""

# import logging
import json
import collections
import numpy as np

# import os

######################################################################
#                                                                    #
#                  Extract Saft Parameters                           #
#                                                                    #
######################################################################
def write_EOSparameters(library, filename):

    """
    Sort and export dictionary of input parameters into .json file.

    Parameters
    ----------
    library : dict
        Dictionary of parameters to be sorted and exported 

    Returns
    -------
    filename : str
        Filename (with or without path) of .json file of parameters
    """

    # logger = logging.getLogger(__name__)

    # sort and write SAFT dict
    for i in library:
        library[i] = collections.OrderedDict(
            sorted(list(library[i].items()), key=lambda tup: tup[0].lower())
        )
    f = open(filename, "w")
    json.dump(library, f, indent=4)


######################################################################
#                                                                    #
#                  Write Thermodynamic Output                         #
#                                                                    #
######################################################################
def writeout_thermo_dict(output_dict, calctype, output_file="thermo_output.txt"):
    """
    Write out result of thermodynamic calculation.

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
    units = {
        "T": "K",
        "P": "Pa",
        "Psat": "Pa",
        "rhol": "mol/m^3",
        "rhov": "mol/m^3",
        "delta": "Pa^(1/2)",
    }

    # Make comment line
    comment = (
        "# This data was generated in DESPASITO using the thermodynamic calculation: "
        + calctype
    )

    # Make results matrix
    keys = []
    matrix = []
    for key, value in output_dict.items():
        tmp_matrix = np.transpose(np.array(value))
        if len(tmp_matrix.shape) == 1:
            keys.append(key)
            matrix.append(np.array(value))
        else:
            for i in range(len(tmp_matrix)):
                keys.append(key + str(i + 1))
                matrix.append(np.array(tmp_matrix[i]))
    matrix = np.transpose(np.array(matrix))

    # Make header line
    header = "#"
    for key in keys:
        if key in units:
            unit = " [{}]".format(units[key])
        else:
            unit = ""
        header += " {}{},".format(key, unit)

    # Write to file
    with open(output_file, "w") as f:
        f.write(comment + "\n")
        f.write(header + "\n")
        for row in matrix:
            f.write((" {}," * len(row)).format(*row) + "\n")


######################################################################
#                                                                    #
#                  Write Parameter Fitting Output                    #
#                                                                    #
######################################################################
def writeout_fit_dict(output_dict, eos, output_file="fit_output.txt"):
    """
    Write out result of fitting calculation.

    Import fitting results dictionary to write out fitting results. In the future, the EOS dictionary will be used to generate new parameter .json files.

    Parameters
    ----------
    output_dict : dict
        Dictionary of given and calculated information from thermodynamic module.
    eos : obj
        Equation of state output that writes pressure, max density, chemical potential, updates parameters, and evaluates objective functions. For parameter fitting algorithm See equation of state documentation for more details.
    output_file : str, Optional, default: thermo_output.txt
        Name of output file

    Returns
    -------
    File of data saved to current directory
    """

    header = (
        "DESPASITO was used to fit parameters for the bead {} Obj. Value: {}\n".format(
            output_dict["fit_bead"], output_dict["objective_value"]
        )
        + "Parameter, Value\n"
    )
    with open(output_file, "w") as f:
        f.write(header)
        for i in range(len(output_dict["fit_parameters"])):
            f.write(
                "{}, {}".format(
                    output_dict["fit_parameters"][i], output_dict["final_parameters"][i]
                )
            )

    # Add function to sort through and write out values
    # step 1, Update parameters for EOS object
    # step 2, write out libraries
