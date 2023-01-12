import numpy as np

def Sigmoid(X):
    return 1.0 / (1.0+np.exp(-X))
def d_Sigmoid(Sig):
    return np.multiply( 1.0-Sig, Sig )

class Layer:
    def __init__(self, nin, nout) -> None:
        self.nin = nin
        self.nout = nout
        self.W = np.random.random( (nout, nin) )-0.5
        self.B = np.random.random( nout ) -0.5
        self.dW = np.zeros( (nout , nin) )
        self.dB = np.zeros( nout )

        self.Input = None
        self.Output = None

    def fire(self, Input):
        return Sigmoid(np.dot(self.W, Input)+self.B)

    def forward(self, Input):
        self.Input = Input
        self.Output= Sigmoid(np.dot(self.W,Input) + self.B)
        return self.Output

    def backward(self, Delta):
        delta_dsig = Delta * d_Sigmoid(self.Output)
        self.dW += delta_dsig.reshape(self.nout, 1) * self.Input
        self.dB += delta_dsig
        return np.dot(delta_dsig, self.W)           
        
    def update(self, alpha):
        self.W -= alpha * self.dW
        self.B -= alpha * self.dB
        self.dW.fill(0)
        self.dB.fill(0)