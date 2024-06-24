import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dataset.mnist import load_mnist
from deep_convnet import DeepConvNet
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("./vol1/ch08/deep_convnet_params.pkl")
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                epochs=20, mini_batch_size=100,
                optimizer='Adam', optimizer_param={'lr':0.001},
                evaluate_sample_num_per_epoch=1000)

accuracy = network.accuracy(x_test, t_test)
print(accuracy)