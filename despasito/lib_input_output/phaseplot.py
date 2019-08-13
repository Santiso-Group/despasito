import sys
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import csv
import phaseplot as pp

system = "co2_cyclohexane"
title = "Pxy of (1) Carbon Dioxide and (2) Cyclohexane"
basepath = '../'
T = [273, 298]
specs = ['_1_1/']

pp.plotPxy(system, basepath, T, specs, title)

col = 'rbgcmk'
################## Experimental Data ###########################
fname_exp = system + "_exp.txt"

# Extract Exp Data
with open(fname_exp, 'rb') as f:
    reader = csv.reader(f)
    csvlist = list(reader)
f.close()

data = list(map(list, list(*csvlist)))
isstr = [i for i, s in enumerate(data[0]) if s.replace('.', '', 1).isdigit() == False]
data_label = [[data[0][i], [], []] for i in isstr[1:]]

exp_data = [[] for i in range(len(isstr) - 1)]
flag_sets = 1
fig = plt.figure(figsize=(10, 6))

for i in range(1, len(isstr)):
    T0 = 0
    if i > len(isstr) - 2:
        i_end = len(data[0])
    else:
        i_end = isstr[i + 1]

    for j, t in enumerate(data[0][isstr[i] + 1:i_end]):
        if t != T0:
            T0 = t
            data_label[i - 1][1].append(T0)
            data_label[i - 1][2].append(isstr[i] + 1 + j)
            exp_data[i - 1].append([])
    for j in range(len(exp_data[i - 1])):
        if j > len(exp_data[i - 1]) - 2:
            j_end = len(data[0])
        else:
            j_end = data_label[i - 1][2][j + 1]

        for k in [1, 2, 3]:
            tmpdata = data[k][data_label[i - 1][2][j]:j_end]
            for ii, s in enumerate(tmpdata):
                if s == None or s == ' ':
                    s = np.nan
                tmpdata[ii] = s
            exp_data[i - 1][j].append(tmpdata)

        coltmp = col[T.index(int(float(data_label[i - 1][1][j])))]
        tmplabel = data_label[i - 1][0] + ' at ' + data_label[i - 1][1][j] + 'K'
        plt.plot(exp_data[i - 1][j][1], exp_data[i - 1][j][0], 'o' + coltmp, label=tmplabel)
        plt.plot(exp_data[i - 1][j][2], exp_data[i - 1][j][0], 'o' + coltmp)

###################### Simulations #################

for i, t in enumerate(T):
    for j, bd in enumerate(specs):

        fname_ext = basepath + system + '_' + str(t) + bd + 'out_' + system + '_' + str(t) + '_ext.txt'
        fname_saf = basepath + system + '_' + str(t) + bd + 'out_' + system + '_' + str(t) + '_saf.txt'

        # Data is Temperature (K), Pressure (Pa), xi, yi
        if op.isfile(fname_ext):
            with open(fname_ext, 'rb') as f:
                reader = csv.reader(f)
                csvlist = list(reader)
            f.close()
            data_ext = list(map(list, list(*csvlist)))
            #data_ext[data_ext==None]=np.nan
            data_ext[1] = np.array([float(k) for k in data_ext[1][1:]])
            data_ext[2] = np.array([float(k) for k in data_ext[2][1:]])
            data_ext[4] = np.array([float(k) for k in data_ext[4][1:]])

            tmplabel = 'Extended Rules at ' + str(t) + 'K'
            plt.plot(data_ext[2], data_ext[1] * 1e-6, col[i], label=tmplabel)
            plt.plot(data_ext[4], data_ext[1] * 1e-6, col[i])

        if op.isfile(fname_saf):
            with open(fname_saf, 'rb') as f:
                reader = csv.reader(f)
                csvlist = list(reader)
            f.close()
            data_saf = list(map(list, list(*csvlist)))
            #data_ext[data_ext==None]=np.nan
            data_saf[1] = np.array([float(k) for k in data_saf[1][1:]])
            data_saf[2] = np.array([float(k) for k in data_saf[2][1:]])
            data_saf[4] = np.array([float(k) for k in data_saf[4][1:]])

            tmplabel = 'SAFT Rules at ' + str(t) + 'K'
            plt.plot(data_saf[2], data_saf[1] * 1e-6, '--' + col[i], label=tmplabel)
            plt.plot(data_saf[4], data_saf[1] * 1e-6, '--' + col[i])

#plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.legend(loc='best')
art = [plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))]
plt.title(title)
plt.xlabel('$x_1$, $y_1$')
plt.ylabel('P (MPa)')
plt.savefig(system + '.png', dpi=400, additional_artists=art, bbox_inches="tight")
fig.subplots_adjust(right=0.7)
fig.subplots_adjust(left=0.1)
plt.show()
