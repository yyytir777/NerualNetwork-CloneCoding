import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape)
print('t', t.shape)