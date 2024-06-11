import itertools
from typing import List

import numpy as np
from numpy._typing import NDArray

from neural_net.models.abstract_model import AbstractModel
from neural_net.layers.abstract_layer import AbstractLayer

class Sequential(AbstractModel):
    def __init__(self, layers: List[AbstractLayer], lr: float = 1e-3, tol=1e-4):
        self.layers = layers
        self.layers[0].add_bias()
        self.lr = lr
        self.tol = tol

    def fit(self, X: NDArray, Y: NDArray) -> None:
        input_x = np.c_[np.ones(X.shape[0]), X]
        iters = 0
        epoch_size = len(X)
        y_shape=Y[0].shape
        sum_deltas = np.zeros(y_shape)
        for x, y in itertools.cycle(zip(input_x, Y)):

            # Feed Forward
            outputs = [x.reshape(1, -1)]
            xw = []
            for layer in self.layers:
                xw = np.dot(outputs[-1],layer.weights)
                outputs.append(layer.forward(outputs[-1]))

            y_hat = outputs[-1]
            print(y_hat)

            delta_out = - ((y / y_hat) + (1 - y) / (1 - y_hat)) * self.layers[-1].activation.derivative(y_hat)
            sum_deltas += np.abs(delta_out).reshape(y_shape)


            if iters % epoch_size==0:
                if np.mean(sum_deltas) < self.tol:
                    break
                sum_deltas = np.zeros(Y[0].shape)

            # Back Propagate
            deltas = [delta_out]
            for i, layer in reversed(list(enumerate(self.layers[:-1]))):
                y_hidden = outputs[i + 1]
                hidden_delta = layer.activation.derivative(y_hidden) * (deltas[-1] @ self.layers[i + 1].weights.T)
                deltas.append(hidden_delta)

            # Update weights
            for i, layer in enumerate(self.layers[::-1]):
                layer.update_weights(self.lr, deltas[i], outputs[-i - 2].T)

            iters += 1

            if iters % 1000 == 0:
                print(f"Iter {iters}")
        print(f"Total iters: {iters}")


    def predict(self, x: NDArray) -> NDArray:
        input_ = np.c_[np.ones(x.shape[0]), x]

        for layer in self.layers:
            input_ = layer.forward(input_)

        return input_