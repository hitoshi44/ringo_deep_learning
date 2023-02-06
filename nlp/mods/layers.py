from mods.functions import *
import numpy as np

# ForwardLayer: Interface
#  - forward( np.array ) -> np.array

class Sigmoid:
    def __init__(self) -> None:
        self.params = []
        self.grads  = []
        self.out = None
    def forward(self, X):
        self.out = 1 / (1+np.exp(-X))
        return self.out
    def backward(self, Delta):
        return Delta * (1.0 - self.out) * self.out

# 全結合層
class Affine:
    def __init__(self, W, b) -> None:
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.X = None
    def forward(self, X):
        W, b = self.params
        self.X = X
        return np.dot(X, W) + b
    def backward(self, Delta):
        self.grads[0][...] = np.dot(self.X.T, Delta) # dL/dW
        self.grads[1][...] = np.sum(Delta)           # dL/dB axis=0
        return np.dot(Delta, self.params[0].T)       # dL/dX
#
class MatMul:
    def __init__(self, W) -> None:
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.X = None
    def forward(self, X):
        W,= self.params
        self.X = X
        return np.dot(X, W)
    def backward(self, Delta):
        self.grads[0][...] = np.dot(self.X.T, Delta) # dL/dW
        return np.dot(Delta, self.W.T)               # dL/dX

class SoftmaxWithLoss:
    def __init__(self) -> None:
        self.params = []
        self.grads  = []
        self.Y = None # softmax の出力
        self.T = None # 教師ラベル
    def forward(self, X, T):
        self.T = T
        self.Y = softmax(X)
        if self.T.size == self.Y.size:
            self.T = self.T.argmax(axis=1)
        return cross_entropy_error(self.Y, self.T)
    def backward(self, dout=1):
        batch_size = self.T.shape[0]
        dx = self.Y.copy()
        dx[np.arange(batch_size), self.T] -= 1
        dx *= dout
        return dx / batch_size
        


