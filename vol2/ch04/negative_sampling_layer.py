import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from common.layers import Embedding

class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding()
        self.params = self.embed.params
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh