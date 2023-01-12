import numpy as np

def Sigmoid(X):
    return 1.0 / (1.0+np.exp(-X))

class Layer:
    def __init__(self, nin, nout) -> None:
        self.nin = nin
        self.nout = nout
        self.W = np.random.random( (nout, nin) )-0.5
        self.B = np.random.random( nout ) -0.5

        self.Input = None
        self.Output = None

    def fire(self, Input):
        return Sigmoid(np.dot(self.W, Input)+self.B)

    def forward(self, Input):
        self.Input = Input
        self.Output= Sigmoid(np.dot(self.W,Input) + self.B)
        return self.Output