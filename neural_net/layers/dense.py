import numpy as np
from numpy._typing import NDArray

from neural_net.functions.activations.abstract_activation import AbstractActivation
from neural_net.layers.abstract_layer import AbstractLayer

class Dense(AbstractLayer):
    def __init__(self, input_shape: int, neurons: int, activation: AbstractActivation):
        self.input_shape = input_shape
        self.neurons = neurons
        self.activation = activation
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        self.weights = 0.01 * np.random.random((self.input_shape, self.neurons))

    def add_bias(self) -> None:
        self.input_shape += 1
        self._initialize_weights()

    def activate(self, x: NDArray) -> NDArray:
        return self.activation.fun(x)

    def forward(self, x: NDArray) -> NDArray:
        return self.activate(np.dot(x, self.weights))

    def update_weights(self, lr: float, delta: NDArray, input_: NDArray) -> None:
        self.weights += lr * (input_ @ delta)