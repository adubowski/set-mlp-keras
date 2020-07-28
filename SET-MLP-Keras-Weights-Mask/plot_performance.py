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

vars = {"acc": "Accuracy", "val_acc": "Accuracy on the validation set", "loss": "Loss", "val_loss": "Loss on the validation set"}
functions = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "selu": "SELU", "softsign": "Softsign", "softplus": "Softplus", "srelu": "SReLU"}
for function in functions.keys():
    for var in vars.keys():
        plt.title(functions[function] + " Performance")
        plt.xlabel("Epochs[#]")
        plt.ylabel("CIFAR10\n" + vars[var] + " [%]")

        eps10=np.loadtxt("results/cifar10/" + function + "/set_mlp_e10_" + var + ".txt")
        eps20=np.loadtxt("results/cifar10/" + function + "/set_mlp_e20_" + var + ".txt")
        eps50=np.loadtxt("results/cifar10/" + function + "/set_mlp_e50_" + var + ".txt")
        eps100=np.loadtxt("results/cifar10/" + function + "/set_mlp_e100_" + var + ".txt")
        eps500=np.loadtxt("results/cifar10/" + function + "/set_mlp_e500_" + var + ".txt")
        dense=np.loadtxt("results/cifar10/" + function + "/set_mlp_dense_" + var + ".txt")

        plt.plot(eps10*100,'r',label="Sparsity 98.85%")
        plt.plot(eps20*100,'b',label="Sparsity 97.12%")
        plt.plot(eps50*100,'g',label="Sparsity 94.25%")
        plt.plot(eps100*100,'m',label="Sparsity 88.50%")
        plt.plot(eps500*100,'c',label="Sparsity 71.2%")
        plt.plot(dense*100,'k',label="Dense network")

        plt.legend(loc=4, prop={'size': 8})
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results/cifar10/" + function + "/performance_" + function + "_" + var + ".pdf")
        plt.close()
