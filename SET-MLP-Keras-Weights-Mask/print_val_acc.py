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

import numpy as np

functions = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "selu": "SELU", "softsign": "Softsign", "softplus": "Softplus", "srelu": "SReLU"}
sparsity_sweep = {}
for function in functions.keys():
    eps10 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e10_val_acc.txt")
    eps20 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e20_val_acc.txt")
    eps50 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e50_val_acc.txt")
    eps100 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e100_val_acc.txt")
    eps500 = np.loadtxt("results/cifar10/" + function + "/set_mlp_e500_val_acc.txt")
    dense = np.loadtxt("results/cifar10/" + function + "/set_mlp_dense_val_acc.txt")
    sparsity_sweep[function] = [round(eps10[-1],4), round(eps20[-1],4), round(eps50[-1],4), round(eps100[-1],4), round(eps500[-1],4), round(dense[-1],4)]
print(sparsity_sweep)
