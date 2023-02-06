import numpy as np
from mods.layers import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, s_in, s_hidden, s_out) -> None:
        
        W1 = 0.01 * np.random.randn(s_in, s_hidden)
        B1 = np.zeros( s_hidden )
        W2 = 0.01 * np.random.randn(s_hidden, s_out)
        B2 = np.zeros(s_out)

        self.layers = [
            Affine(W1, B1),
            Sigmoid(),
            Affine(W2, B2),
        ]
        self.loss_layer = SoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads  += layer.grads
    def predict(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    def forward(self, X, T):
        return self.loss_layer.forward(
            self.predict(X), 
            T,
        )
    def backward(self, Delta=1):
        Delta = self.loss_layer.backward(Delta)
        for layer in reversed(self.layers):
            Delta = layer.backward(Delta)
        return Delta