import numpy as np
from mods.layer import Layer

def mse_loss(O, Y):
    n = len(O)
    return (
        np.sum(np.square(O-Y)) /n, # loss
        np.multiply(2.0,(O-Y)) /n, # delta
    )

class NeuralNetwork:
    def __init__(self,
        dimension, # int の タプル
        loss_function, # 損失関数
    ) -> None:
        self.nin = dimension[0]
        self.nout = dimension[len(dimension) - 1]
        self.loss_function = loss_function

        self.Layers = self._build_layers(dimension)
        self.Loss = .0

    def fire(self, Input):
        output = Input
        for layer in self.Layers:
            output = layer.fire(output)
        return output

    def forward(self, Input):
        output = Input
        for layer in self.Layers:
            output = layer.forward(output)
        return output 

    def backward(self, Delta):
        delta = Delta
        for layer in reversed(self.Layers):
            delta = layer.backward(delta)
        return delta
        
    def update(self, alpha):
        for layer in self.Layers:
            layer.update(alpha)
        self.Loss = .0

    def calc_loss(self, O, Y):
        (loss, delta) = self.loss_function(O, Y)
        self.Loss += loss
        return delta

    def _build_layers(self, dimesion) -> list:
        result = []
        for i in range(len(dimesion)-1):
            result.append(
                Layer(dimesion[i], dimesion[i+1])
            )
        return result