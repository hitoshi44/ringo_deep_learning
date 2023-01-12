import numpy as np
from layer import Layer

class NeuralNetwork:
    def __init__(self,
        dimension, # int の タプル
        loss_function, # 損失関数
    ) -> None:
        self.nin = dimension[0]
        self.nout = dimension[len(dimension) - 1]
        self.loss_function = loss_function

        self.Layers = self._build_layers(dimension)

    def _build_layers(self, dimesion) -> list:
        result = []
        for i in range(len(dimesion)-1):
            result.append(
                Layer(dimesion[i], dimesion[i+1])
            )
        return result