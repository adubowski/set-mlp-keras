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

x = [98.85, 97.12, 94.25, 88.5, 71.2]
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
vars = {"acc": "Accuracy", "val_acc": "Accuracy", "loss": "Loss", "val_loss": "Loss on the validation set"}
functions = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "selu": "SELU", "softsign": "Softsign", "softplus": "Softplus", "srelu": "SReLU"}
for var in vars.keys():
    plt.xlabel("Sparsity[%]")
    plt.ylabel("CIFAR10\n" + vars[var] + " [%]")
    if var == "acc" or var == "loss":
        plt.title("Sparsity sweep on the training set")
    else:
        plt.title("Sparsity sweep on the validation set")
    sparsity_sweep = {}
    sparsity_sweep_val = {}
    color_index = 0
    for function in functions.keys():

        eps10 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e10_" + var + ".txt")
        eps20 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e20_" + var + ".txt")
        eps50 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e50_" + var + ".txt")
        eps100 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e100_" + var + ".txt")
        eps500 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e500_" + var + ".txt")
        dense = np.loadtxt("results/cifar10/" + function + "/set_mlp_dense_" + var + ".txt")

        sparsity_sweep[function] = [eps10[-1]*100, eps20[-1]*100, eps50[-1]*100, eps100[-1]*100, eps500[-1]*100]

        plt.plot(x, [dense[-1] * 100] * 5, ':' + colors[color_index], label=functions[function] + " Dense")
        plt.plot(x, sparsity_sweep[function], colors[color_index], label=functions[function] + " Sparse")
        color_index += 1

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., prop={'size': 6})
    plt.legend(loc=4, prop={'size': 9}, ncol=2)
    plt.xlim([70, 100])
    if var == "acc" or var == "val_acc":
        plt.ylim([0,90])
    else:
        plt.ylim([0, 200])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/cifar10/sparsity_sweep_" + var + ".pdf")
    plt.close()
