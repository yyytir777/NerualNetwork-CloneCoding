import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # 0이상 train_size미안의 수 중에서 무작위로 batch_size만큼의 수를 choice

x_batch = x_train[batch_mask]

