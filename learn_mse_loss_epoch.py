import numpy as np
from mods import mnist, network

(images, labels) = mnist.np_load_train()
NN = network.NeuralNetwork((784,100,10), network.mse_loss)


alpha = 0.56
e = 0
while True:
    alpha /= 2.0
    if alpha < 0.01:
        break
    label = np.zeros(10)
    i = 0
    size = len(images)
    chunk = range(50)
    while i < size:
        for _ in chunk:
            label.fill(0)
            label[labels[i]] = 1.0
            output = NN.forward(images[i])
            delta = NN.calc_loss(output, label)
            NN.backward( delta )
            i+=1
        
        NN.update(alpha)

    (tests, labels) = mnist.np_load_train()
    hit = 0
    for (image, label) in zip(tests, labels):
        pred = np.argmax(NN.fire(image))
        if pred == label:
            hit += 1
    e+=1
    print("At epoch:", e, " alpha:", alpha)
    print("Accuracy :", hit/len(tests))
    print("  ", hit, " / ", len(tests))