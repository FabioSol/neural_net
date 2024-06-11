from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy._typing import NDArray

from neural_net.functions.activations.abstract_activation import AbstractActivation

class AbstractLayer(ABC):

    input_shape:int
    neurons:int
    activation:AbstractActivation
    weights: NDArray

    @abstractmethod
    def _initialize_weights(self) -> None:
        pass

    @abstractmethod
    def add_bias(self) -> None:
        pass

    @abstractmethod
    def activate(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        pass

    @abstractmethod
    def update_weights(self, lr: float, delta: NDArray, input_: NDArray) -> None:
        pass