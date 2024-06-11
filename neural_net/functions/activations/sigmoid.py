from typing import Union

from neural_net.functions.activations.abstract_activation import AbstractActivation
import numpy as np
from numpy.typing import NDArray


class Sigmoid(AbstractActivation):
    @staticmethod
    def fun(x: Union[float, NDArray]) -> Union[float, NDArray]:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(sigmoid: Union[float, NDArray]) -> Union[float, NDArray]:
        return sigmoid * (1 - sigmoid)




