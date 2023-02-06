import numpy as np
from mods.layers import Affine, Sigmoid, SoftmaxWithLoss
from mods.networks import TwoLayerNet
from mods.trainer import Trainer
from mods.optimizer import SGD

from datasets import spiral
import matplotlib.pyplot as plt

train_data, train_label = spiral.load_data()
model = TwoLayerNet(s_in=2, s_hidden=10, s_out=3)
optimizer = SGD(1.0)

trainer = Trainer(model, optimizer)
trainer.fit(
    train_data, train_label,
    epoch=400, batch=30,
    shuffle=True,
)

plt.plot(np.arange(len(trainer.Loss)), trainer.Loss, label='train')
plt.xlabel('iteration x10')
plt.ylabel('Loss')
plt.show()

# 境界領域のプロット
x = train_data
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()