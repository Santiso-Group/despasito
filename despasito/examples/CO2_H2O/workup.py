
import numpy as np
import matplotlib.pyplot as plt

data_old = np.genfromtxt("old_method.csv", delimiter=",").T
data_new = np.genfromtxt("new_method.csv", delimiter=",").T

change = np.log10(np.abs(data_new-data_old))

if len(np.shape(change)) != 1:
    for i,x in enumerate(change):
        x3 = [xx for xx in x if np.abs(xx) not in [np.inf,np.nan]]
        x2 = [xx if np.abs(xx) not in [np.inf] else np.nan for xx in x]
        if x3:
            print(i,len(x3), max(x3), "{}%".format(len(x3)/len(x2)*100.0), max(data_new[i]))
        plt.plot(np.arange(len(x2)), x2,".", label=str(i))
    plt.legend(loc="best")
    plt.show()
else:
    for i in range(len(change)):
        print("New: {}, Old: {}, Oerror: {}".format(data_new[i],data_old[i],change[i]))

