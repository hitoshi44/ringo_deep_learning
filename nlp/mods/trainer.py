import numpy as np

class Trainer:
    def __init__(self, Model, Optimizer):
        self.Model = Model
        self.Optimizer = Optimizer
        self.Loss = []
    def fit(self, Data, Label, epoch, batch, **kwargs): # max_grad もあとで追加
        self._set_args(kwargs)
        if not self._check(Data, Label):
            print(self._msg)
            return
        data_size = len(Data)
        max_iters = data_size // batch
        total_loss = 0
        loss_count = 0
        for e in range(epoch):
            if self.shuffle:
                shuffle = np.random.permutation(data_size)
            X = Data[shuffle]
            T = Label[shuffle]
            for i in range(max_iters):
                # forward, backward, update
                total_loss += self.Model.forward(
                    X[i*batch:(i+1)*batch],
                    T[i*batch:(i+1)*batch],
                )
                self.Model.backward()
                self.Optimizer.update(
                    self.Model.params,
                    self.Model.grads,
                )
                # inc
                loss_count += 1
                if (i+1) % 10 == 0:
                    self.Loss.append( total_loss / loss_count )
                    total_loss, loss_count = 0, 0
    def _check(self, Data, Label):
        return len(Data) == len(Label)
    def _set_args(self, kwargs):
        self.shuffle = kwargs["shuffle"]
        
        