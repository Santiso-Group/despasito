""" Routines for writing .txt and .json output files files from dictionaries.
"""

# import logging
import json
import collections
import numpy as np

# logger = logging.getLogger(__name__)


def write_EOSparameters(library, filename):

    """
    Sort and export dictionary of input parameters into .json file.

    Parameters
    ----------
    library : dict
        Dictionary of parameters to be sorted and exported 
    filename : str
        Filename (with or without path) of .json file of parameters

    """

    # sort and write SAFT dict
    for i in library:
        library[i] = collections.OrderedDict(
            sorted(list(library[i].items()), key=lambda tup: tup[0].lower())
        )
    f = open(filename, "w")
    json.dump(library, f, indent=4)


def writeout_thermo_dict(output_dict, calctype, output_file="thermo_output.txt"):
    """
    Write out result of thermodynamic calculation.

    Import dictionary of both input and output data to produce a file. A line in the top clarifies the calculation type done.

    Parameters
    ----------
    output_dict : dict
        Dictionary of given and calculated information from thermodynamic module
    calculation_type : str
        Thermodynamic calculation type used
    output_file : str, Optional, default="thermo_output.txt"
        Name of output file

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
    comment = "# This data was generated in DESPASITO using the thermodynamic calculation: {}".format(
        calctype
    )

    # Make results matrix
    keys = []
    matrix = []
    for key, value in output_dict.items():
        if np.size(value[0]) > 1:
            tmp_matrix = np.transpose(np.stack(value))
        else:
            if len(np.shape(value[0])) == 0:
                tmp_matrix = np.array(value)
            else:
                tmp_matrix = np.concatenate(value, axis=0)

        if len(tmp_matrix.shape) == 1:
            keys.append(key)
            matrix.append(np.array(tmp_matrix))
        else:
            for i in range(len(tmp_matrix)):
                keys.append(key + str(i + 1))
                matrix.append(np.array(tmp_matrix[i]))
    matrix = np.transpose(np.stack(matrix))

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


def writeout_fit_dict(output_dict, output_file="fit_output.txt"):
    """
    Write out result of fitting calculation.

    Import fitting results dictionary to write out fitting results. In the future, the EOS dictionary will be used to generate new parameter .json files.

    Parameters
    ----------
    output_dict : dict
        Dictionary of given and calculated information from thermodynamic module.
    output_file : str, Optional, default="fit_output.txt"
        Name of output file

    """

    header = (
        "DESPASITO was used to fit parameters for the bead {} Obj. Value: {}\n".format(
            output_dict["fit_bead"], output_dict["objective_value"]
        )
        + "Parameter, Value\n"
    )
    with open(output_file, "w") as f:
        f.write(header)
        for i in range(len(output_dict["fit_parameter_names"])):
            f.write(
                "{}, {}\n".format(
                    output_dict["fit_parameter_names"][i],
                    output_dict["parameters_final"][i],
                )
            )
