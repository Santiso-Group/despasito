import re
import os
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy as np

# from scipy.ndimage.filters import gaussian_filter1d


def plotPxy(
    system, T, specs, title, sim_types, basepath="", debug_plots=False, plot_show=False
):

    r"""
    Plot experimental Pxy data, assume binary mixture where only x1 and y1 are listed in data structures
    
    Parameters
    ----------
    system : str
        This string is the root of all paths and file names.
    T : list[float]
        List of temeratures at which Pxy data was evaluated
    specs : list[str]
        Last term in subdirectory name separated by underscores of "system". Used here to differentiate systems with different bead sizes
    title : str
        Title in figures of output data.
    sim_types : list[str]
        List of simulation types, right now the available options are "ext" for extended mixing rules, "saf" for standard saft mixing rules, and "paper" for binary interaction parameters already published in journal articles.
    basepath : str, Optional, default=""
        Allows path to simulation data files.
    debug_plots : bool, Optional, default=False
        Choose whether to save debug plots
    debug_show : str, Optional, default=False
        Show plots as figures instead of closing after saving, useful for getting a closer look at data.

    Returns
    -------
    Save figures according settings

    """

    col = "rbgcmk"
    style = ["-", "--", ":", "-.", "", " "]

    Nfigs = len(specs)
    figs = {}
    for i in range(Nfigs):
        figs["fig{}".format(i + 1)] = plt.figure(figsize=(10, 6))

    # Extract and plot Exp Data
    fname_exp = system + "_exp.txt"
    exp_data = extract_exp_Pxy_data(fname_exp)
    plot_exp_Pxy_data(Nfigs, T, exp_data, c_array=col)

    # Extract and plot simulation data
    for j, bd in enumerate(specs):
        fig_num = j + 1
        for i, t in enumerate(T):
            for k, sim in enumerate(sim_types):

                # Check if target file exists
                fname = "{}{}_{}_{}/out_{}_{}_{}.txt".format(
                    basepath, system, t, bd, system, t, sim
                )
                if not os.path.isfile(fname):
                    print("File doesn't exist: {}".format(fname))
                    continue
                sim_data, debug_data = extract_sim_Pxy_data(fname)

                # Plot data
                if sim == "ext":
                    label = "Extended {} K {}".format(t, bd)
                elif sim == "saf":
                    label = "SAFT {} K {}".format(t, bd)
                elif sim == "paper":
                    label = "Paper {} K {}".format(t, bd)
                plot_sim_Pxy_data(
                    sim_data,
                    debug_data,
                    label,
                    fig_num=fig_num,
                    style=style[k],
                    color=col[i],
                    debug_plots=debug_plots,
                )

        art = [plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))]
        plt.title(title)
        plt.xlabel("$x_1$, $y_1$")
        plt.ylabel("P (MPa)")
        plt.savefig(
            "{}_{}.png".format(system, bd),
            dpi=400,
            additional_artists=art,
            bbox_inches="tight",
        )
        if plot_show:
            fig = figs["fig{}".format(j + 1)]
            fig.subplots_adjust(right=0.7)
            fig.subplots_adjust(left=0.1)

    if plot_show:
        plt.show()
    plt.close("all")


def extract_exp_Pxy_data(fname):
    r"""
    Extract experimental data from file.
    
    Parameters
    ----------
    fname : str
        name of experimental data
    
    Returns
    -------
    temperatures : numpy.ndarray
        Array of temperatures used 
    data : list[list[list[float]]]
        line_np_array = np.array(linearray[:-1],float)
        Array of data in the shape of len(temperatures), by (number of components -1)*2+1, by the number of pressure and composition values. The second dimension is P, x1, x2, x(n-1), y1, y2, y(n-1) so that if there are two components then the number of mole fraction columns is 1.
    labels : list
        A list of strings the same length as temperatures. The string usually represents reference information.    
    """

    # Extract data from csv file
    with open(fname, "r") as f:
        contents = f.readlines()

    # Extract data
    temperatures = []
    exp_data = []
    labels = []
    T = 0
    flag = 0

    for i, line in enumerate(contents):
        # linearray = line.split()
        linearray = re.findall(r"[^,\s]+", line)

        # skip comments and empty lines, as well as first header line
        if not linearray or "#" in linearray[0] or i == 0:
            continue

        # If the line is a string, update the label
        if linearray[0].replace(".", "", 1).isdigit() == False:
            label_tmp = line
            flag = 1
            continue

        # Initialize new data set if necessary
        if T != float(linearray[0]) or flag == 1:
            if flag == 1:
                flag = 0
            T = float(linearray[0])
            labels.append(label_tmp)
            if len(exp_data) > 0:
                exp_data[-1] = np.transpose(np.array(exp_data[-1]))
                exp_data[-1][0] = exp_data[-1][0] * 1e6
            exp_data.append([])
            temperatures.append(T)

        linearray = [np.nan if x == "None" else float(x) for x in linearray]
        exp_data[-1].append(np.array(linearray[1:], float))

    # Reformat last data set
    exp_data[-1] = np.transpose(np.array(exp_data[-1]))
    exp_data[-1][0] = exp_data[-1][0] * 1e6

    data = [np.array(temperatures), labels, exp_data]

    return data


def plot_exp_Pxy_data(Nfigures, calc_temps, data, style="o><^", c_array="rbgcmk"):

    r"""
    Plot experimental Pxy data, assume binary mixture where only x1 and y1 are listed in data structures
    
    Parameters
    ----------
    Nfigures : int
        The number of duplicate figures to plot the data points
    calc_temps : numpy.ndarray
        Array of temperatures to be plotted. The placement corresponds to the color used in c_array.
    data : list[numpy.ndarray,list[str],list[numpy.ndarray]
        Array of experimental Pxy data. The first string contains the temperature, and the second the source of data, and the third is a numpy array in the shape of len(temperatures), by (number of components -1)*2+1, by the number of data points. The second dimension is P, x1, x2, x(n-1), y1, y2, y(n-1), so that if there are two components then the number of mole fraction columns is 1.
    style : str, Optional, default="o"
        Any allowed matplotlib type.
    c_array : str
        A string of at least as many calc_temperatures

    Returns
    -------
    Experimental data is plotted on figures 1 through Nfigures

    """

    if type(calc_temps) not in [list, np.ndarray]:
        calc_temps = [calc_temps]

    for i, temp in enumerate(calc_temps):

        # Find indices of experimental data corresponding to desired temperature
        print(temp, data[0])
        jnd = np.where(np.abs(temp - data[0]) < 1.0)[0]

        for j in range(len(jnd)):
            tmplabel = "{} at {}K".format(data[1][jnd[j]], data[0][jnd[j]])
            for f in range(Nfigures):
                fig = f + 1
                plt.figure(fig)
                plt.plot(
                    data[2][jnd[j]][1],
                    data[2][jnd[j]][0],
                    style[j] + c_array[i],
                    label=tmplabel,
                )
                plt.plot(data[2][jnd[j]][2], data[2][jnd[j]][0], style[j] + c_array[i])


def extract_sim_Pxy_data(fname):
    r"""
    Extract simulation data from file. Assuming a binary system where mole fractions for both components are present.
    
    Parameters
    ----------
    fname : str
        name of experimental data file with path
    
    Returns
    -------
    sim_data : numpy.ndarray
        Array of simulation data, P, x1, y1.
    debug_data : numpy.ndarray
        Extra data from output file containing information for debugging or assessing the data. The first and second columns are the density flags for the liquid and vapor phase respectively. See thermodynamics.calc.calc_vapor_density or calc.rhol for more details. The third columns contains the objective function, this should be at least less than 1e-5.
    """

    # Extract data from csv file
    data = np.transpose(np.genfromtxt(fname, skip_header=2, delimiter=","))

    sim_data = np.array([data[3], data[1], data[4]])
    debug_data = np.array([data[6], data[7], data[8]])

    return sim_data, debug_data


def plot_sim_Pxy_data(
    sim_data,
    debug_data,
    label,
    fig_num=1,
    style="",
    color="k",
    linewidth=2,
    debug_plots=False,
    debug_path="",
    debug_cut=1e-5,
):

    r"""
    Plot experimental Pxy data, assume binary mixture where only x1 and y1 are listed in data structures
    
    Parameters
    ----------
    sim_data : numpy.ndarray
        Array of simulation data, P, x1, y1.
    debug_data : numpy.ndarray
        Extra data from output file containing information for debugging or assessing the data. The first and second columns are the density flags for the liquid and vapor phase respectively. See thermodynamics.calc.calc_vapor_density or calc.rhol for more details. The third columns contains the objective function, this should be at least less than 1e-5.
    label : str
        Label for phase diagram figure
    fig_num : int, Optional, default=1
        Figure number to plot data
    style : str, Optional, default=""
        Line style from matplotlib, default is solid
    color : str, Optional, default="k"
        Color accepted by matplotlib, default is black.
    linewidth : int, Optional, default=1
        Linewidth accepted by matplotlib.
    debug_plots : bool, Optional, default=False
        Choose whether to save debug plots
    debug_path : str, Optional, default=""
        Path to save debug plots in
    debug_cut : float
        Cutoff for when objective function is not considered low enough

    Returns
    -------
    Experimental data is plotted according to inputs

    """

    plt.figure(fig_num)
    plt.plot(sim_data[1], sim_data[0], style + color, label=label)
    plt.plot(sim_data[2], sim_data[0], style + color)

    if debug_plots:

        N_current_figures = plt.get_fignums()
        fig_num2 = N_current_figures[-1] + 1

        name = label.replace(" ", "_")

        # Save Flag Data
        plt.figure(fig_num2)
        plt.plot(debug_data[0], "o-", color="k", label="x1 flag")
        plt.plot(debug_data[1], "o-", color="b", label="y1 flag")
        plt.xlabel("Data Point")
        plt.ylabel("Density Flag")
        plt.ylim((0.0, 4.0))
        plt.legend(loc="best")
        if debug_path:
            debug_path = "/" + debug_path
        plt.savefig("{}{}_flags.pdf".format(debug_path, name))
        plt.close(fig_num2)

        # Save Obj Data
        log_obj = np.log10(debug_data[2])
        plt.figure(fig_num2)
        plt.plot(log_obj, "o-", color="k")
        plt.xlabel("Data Point")
        plt.ylabel("log10 Obj Value")
        plt.ylim((-15, 3))
        if debug_path:
            debug_path = "/" + debug_path
        plt.savefig("{}{}_obj.pdf".format(debug_path, name))
        plt.close(fig_num2)

        # Plot flags
        markers = ["s", "X"]
        color = ["b", "y"]
        plt.figure(fig_num)
        for i in range(len(sim_data)):
            # Check objective function
            if log_obj[i] > debug_cut:
                jnd = 1
            else:
                jnd = 0
            # Check liquid flags
            if debug_data[0][i] not in [1, 2]:
                ind1 = 1
            else:
                ind1 = 0
            # Check vapor flags
            if debug_data[1][i] not in [0, 2, 4]:
                ind2 = 1
            else:
                ind2 = 0
            # Plot new point
            if ind1 == 1:
                plt.plot(
                    sim_data[1][i], sim_data[0][i], marker=markers[ind1], color="k"
                )
            if ind2 == 1 or jnd == 1:
                plt.plot(
                    sim_data[2][i],
                    sim_data[0][i],
                    marker=markers[ind2],
                    color=color[jnd],
                )


def ADD_Pxy_data(
    exp_data, sim_data, filename="ADD.csv", column_string="", column_header=""
):

    r"""
    Calculate the %ADD between calulated and experimental values.
    
    Parameters
    ----------
    exp_data : numpy.ndarray
        Array of experimental data.
    sim_data : numpy.ndarray
        Array of simulation data to compare to experimental data. The columns must match. If a direct comparison is to be made, the vectors must be of identical length. One might also have a list of such array, in which case column_string may be a list of the same length.
    filename : str, Optional, default=ADD.csv
        File name to which data should be saved.
    column_string : str or list[str], Optional, default=""
        A string or list of strings containing comma separated values. This allows the output spreadsheet to have more specificity.
    column_header : str, Optional, default=""
        If the file, filename, doesn't exist, a new .csv file will be created with this string of comma separated values.

    Returns
    -------
    Entries are added to spreadsheet

    """

    if type(sim_data[0][0]) not in [list, np.ndarray]:
        sim_data = [sim_data]

    l_c = len(sim_data[0][0])

    lines = []
    for i, data in enumerate(sim_data):

        # Calculate %AAD
        AAD_tmp = []
        for j in range(l_c):
            AAD_tmp.append(
                str(np.nanmean(np.abs((data[j] - exp_data[j]) / exp_data[j])) * 100)
            )

        # Assemble line
        if type(column_string) == list:
            line = column_string[i]
        else:
            line = column_string

        # Add AAD to line
        line += ", " + ", ".join(ADD_tmp)
        lines.append(line)

    if not os.path.isfile(filename):
        with open(filename, "w") as f:
            f.write(column_header)

    with open(filename, "w") as f:
        for line in lines:
            f.write(line)
