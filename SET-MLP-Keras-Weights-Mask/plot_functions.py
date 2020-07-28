import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def sreluPlot(x):
    if x > 0.8:
        return 0.8 + 0.2 * (x - 0.8)
    elif x > -0.8:
        return x
    else:
        return -0.8 + 0.2 * (x + 0.8)


colors = ['b','g','r','c','m','y', 'k']
functions = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "selu": "SELU", "softsign": "Softsign", "softplus": "Softplus", "srelu": "SReLU"}

x_data = np.arange(-3, 3, 0.01)

relu = tf.keras.activations.relu(x_data)
sigmoid = tf.keras.activations.sigmoid(x_data)
tanh = tf.keras.activations.tanh(x_data)
selu = tf.keras.activations.selu(x_data)
softsign = tf.keras.activations.softsign(x_data)
softplus = tf.keras.activations.softplus(x_data)

srelu = []
for x in x_data:
    srelu.append(sreluPlot(x))

plt.plot(x_data, relu.numpy(), color = 'b', label="ReLU")
plt.plot(x_data, sigmoid.numpy(), color = 'g', label="Sigmoid")
plt.plot(x_data, tanh.numpy(), color = 'r', label="Tanh")
plt.plot(x_data, selu.numpy(), color = 'c', label="SELU")
plt.plot(x_data, softsign.numpy(), color = 'm', label="Softsign")
plt.plot(x_data, softplus.numpy(), color = 'y', label="Softplus")
plt.plot(x_data, srelu, 'k', label="SReLU")

plt.legend(loc=2, prop={'size': 10})
plt.grid(True)
# plt.xlim([70, 100])
plt.tight_layout()

plt.show()

# plt.savefig("activation_functions.pdf")
# plt.close()