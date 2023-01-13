import numpy as np
import gzip

data_dir = "./mods/dataset/"

def np_load_train():
    res = []
    with gzip.open( data_dir + 'train-images-idx3-ubyte.gz') as f:
        data = np.frombuffer(f.read(),np.uint8, offset=16)
        res.append(data.reshape(-1, 784) / 256)
    with gzip.open( data_dir + 'train-labels-idx1-ubyte.gz') as f:
        res.append(np.frombuffer(f.read(), np.uint8, offset=8))
    return (res[0],res[1])

def np_load_test():
    res = []
    with gzip.open( data_dir + 't10k-images-idx3-ubyte.gz') as f:
        data = np.frombuffer(f.read(),np.uint8, offset=16)
        res.append(data.reshape(-1, 784) / 256)
    with gzip.open( data_dir + 't10k-labels-idx1-ubyte.gz') as f:
        res.append(np.frombuffer(f.read(), np.uint8, offset=8))
    return (res[0],res[1])