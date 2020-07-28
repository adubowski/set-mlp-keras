# Author: Decebal Constantin Mocanu et al.;
# Plot performance of all three models on CIFAR10

# This is a pre-alpha free software and was tested with Python 3.5.2, Keras 2.1.3, Keras_Contrib 0.0.2, Tensorflow 1.5.0, Numpy 1.14;
# The code is distributed in the hope that it may be useful, but WITHOUT ANY WARRANTIES; The use of this software is entirely at the user's own risk;
# For an easy understanding of the code functionality please read the following articles.

# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}

#@Article{Mocanu2016XBM,
#author="Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio",
#title="A topological insight into restricted Boltzmann machines",
#journal="Machine Learning",
#year="2016",
#volume="104",
#number="2",
#pages="243--270",
#doi="10.1007/s10994-016-5570-z",
#url="https://doi.org/10.1007/s10994-016-5570-z"
#}

#@phdthesis{Mocanu2017PhDthesis,
#title = "Network computations in artificial intelligence",
#author = "D.C. Mocanu",
#year = "2017",
#isbn = "978-90-386-4305-2",
#publisher = "Eindhoven University of Technology",
#}

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

x = [98.85, 97.12, 94.25, 88.5, 71.2]
colors = ['b','g','r','c','m','y', 'k']
sparsities = {"e10": "Sparsity 98.85%", "e20": "Sparsity 97.12%", "e50": "Sparsity 94.25%",
              "e100": "Sparsity 88.50%", "e500": "Sparsity 71.2%", "dense": "Dense Network"}
vars = {"acc": "Accuracy", "loss": "Loss"}
functions = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "selu": "SELU", "softsign": "Softsign", "softplus": "Softplus", "srelu": "SReLU"}
suffix = " on training set"
val_suffix = " on validation set"
for var in vars.keys():

    for s in sparsities.keys():
        color_index = 0

        plt.title("Overfitting - " + vars[var] + "\n" + sparsities[s])
        plt.xlabel("Epochs[#]")
        plt.ylabel("CIFAR10\n Overfitting [%]")
        for function in functions.keys():
            l = functions[function]
            data = np.loadtxt("results/cifar10/" + function + "/set_mlp_" + s + "_" + var + ".txt")
            # plt.plot(data*100,'r',label= l + suffix)

            val_data = np.loadtxt("results/cifar10/" + function + "/set_mlp_" + s + "_val_" + var + ".txt")

            data2 = data*100 - val_data*100

            plt.plot(data2, colors[color_index],label= l)
            color_index += 1
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 9})
        plt.legend(loc=2, prop={'size': 8})
        plt.grid(True)
        # plt.xlim([70, 100])
        if var == "acc":
            plt.ylim([-10, 20])
        plt.tight_layout()
        plt.savefig("results/cifar10/overfitting_" + var + "_" + s + ".pdf")
        plt.close()



