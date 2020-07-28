# sparse-evolutionary-artificial-neural-networks
* Proof of concept implementations of various sparse artificial neural network models with adaptive sparse connectivity trained with the Sparse Evolutionary Training (SET) procedure.  
* The following implementations are distributed in the hope that they may be useful, but without any warranties; Their use is entirely at the user's own risk.

###### Implementation 1 - SET-MLP with Keras and Tensorflow (SET-MLP-Keras-Weights-Mask)

* Proof of concept implementation of Sparse Evolutionary Training (SET) for Multi Layer Perceptron (MLP) on CIFAR10 using Keras and a mask over weights.  
* This implementation can be used to test SET in varying conditions, using the Keras framework versatility, e.g. various optimizers, activation layers, tensorflow.  
* Also it can be easily adapted for Convolutional Neural Networks or other models which have dense layers.
* Variants of this implementation have been used to perform the experiments from Reference 1 with MLP and CNN.  
* However, due the fact that the weights are stored in the standard Keras format (dense matrices), this implementation can not scale properly.  
* If you would like to build an SET-MLP with over 100000 neurons, please use Implementation 2.

###### References

For an easy understanding of these implementations please read the following articles. Also, if you use parts of this code in your work, please cite the corresponding ones:

1. @article{Mocanu2018SET,
  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
  journal =       {Nature Communications},
  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
  year =          {2018},
  doi =           {10.1038/s41467-018-04316-3},
  url =           {https://www.nature.com/articles/s41467-018-04316-3 }}

2. @article{Mocanu2016XBM,
author={Mocanu, Decebal Constantin and Mocanu, Elena and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
title={A topological insight into restricted Boltzmann machines},
journal={Machine Learning},
year={2016},
volume={104},
number={2},
pages={243--270},
doi={10.1007/s10994-016-5570-z},
url={https://doi.org/10.1007/s10994-016-5570-z }}

3. @phdthesis{Mocanu2017PhDthesis,
title = {Network computations in artificial intelligence},
author = {Mocanu, Decebal Constantin},
year = {2017},
isbn = {978-90-386-4305-2},
publisher = {Eindhoven University of Technology},
url={https://pure.tue.nl/ws/files/69949254/20170629_CO_Mocanu.pdf }
}

4. @article{Liu2019onemillion,
  author =        {Liu, Shiwei and Mocanu, Decebal Constantin and Mocanu and Ramapuram Matavalam, Amarsagar Reddy and Pei, Yulong Pei and Pechenizkiy, Mykola},
  journal =       {arXiv:1901.09181},
  title =         {Sparse evolutionary Deep Learning with over one million artificial neurons on commodity hardware},
  year =          {2019},
  url={https://arxiv.org/abs/1901.09181 }
}

SET shows that large sparse neural networks can be built if topological sparsity is created from the design phase, before training. There are many algorithmic and implementation improvements which can be made. If you find this work interesting, please share the links to this Github page and to Reference 1. For any question, suggestion, feedback please feel free to contact me by email.

###### Community

Some time ago, I had a very pleasant unexpected surprise when I found out that Michael Klear released "Synapses". This library implements SET layers in PyTorch and as Michael says it is "truly sparse". For more details please read his article:

https://towardsdatascience.com/the-sparse-future-of-deep-learning-bce05e8e094a   

And try out "Synapses" yourself:

https://github.com/AlliedToasters/synapses

Many things can be improved in "Synapses". If interested, please contact and help Michael in developing further the project.

###### Update 4 June 2020

Our paper "Topological insights into sparse neural networks" https://arxiv.org/pdf/2006.14085.pdf has been accepted at ECMLPKDD 2020. It proposes Neural Network Sparse Topology Distance (NNSTD) to measure the distance between different sparse neural networks. The code is here https://github.com/Shiweiliuiiiiiii/Sparse_Topology_Distance. Also, it shows in a principled manner that sparse training easily unveils a plenitude of sparse sub-networks with very different topologies which outperform the dense networks. 

Many thanks,   
Decebal
