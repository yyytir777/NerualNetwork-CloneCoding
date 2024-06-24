import numpy as np
from two_layer_net import TwoLayerNet

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)

grads = net.numerical_gradient(x, t)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)