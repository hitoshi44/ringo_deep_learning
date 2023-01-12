import numpy as np

class Layer:
    def __init__(self, nin, nout) -> None:
        self.nin = nin
        self.nout = nout