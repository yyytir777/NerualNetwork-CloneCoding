import numpy as np
from neuralnet_mnist import get_data, init_network, predict

x, t = get_data()
network = init_network()

# W1, W2, W3 = network['W1'], network['W2'], network['W3']

# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

accuracy_cnt = 0
# 데이터를 하나씩 처리
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt / len(x))))