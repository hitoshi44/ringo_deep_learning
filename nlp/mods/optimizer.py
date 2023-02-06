import numpy as np

class SGD:
    def __init__(self, lr=1e-2) -> None:
        self.lr = lr
    def update(self, params, grads):
        # params にパラメータへのポインタが格納されている仮定
        for i in range(len(grads)):
            params[i] -= self.lr * grads[i]

    