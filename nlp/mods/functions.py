import numpy as np

def softmax(X):
    if X.ndim == 2:
        X = X - X.max(axis=1, keepdims=True)
        X = np.exp(X)
        return X / X.sum(axis=1, keepdims=True)
    elif X.ndim == 1:
        X = X - np.max(X)
        return np.exp(X) / np.sum(np.exp(X))

def cross_entropy_error(Y, T):
    if Y.ndim == 1:
        T = T.reshape(1, T.size) #?? (d,) => (1, d) って事だと思う
        Y = Y.reshape(1, Y.size) #??
    if T.size == Y.size:
        T = T.argmax(axis=1)
    
    batch_size = Y.shape[0]

    return -np.sum(np.log(Y[np.arange(batch_size), T] + 1e-7 )) / batch_size