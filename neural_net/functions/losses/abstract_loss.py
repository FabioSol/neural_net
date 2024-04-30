from typing import Union

from numpy._typing import NDArray

from neural_net.functions.abstract_function import AbstractFunction, abstractmethod

class AbstractError(AbstractFunction):

    @abstractmethod
    @staticmethod
    def fun(y_true: Union[float, NDArray],y_hat: Union[float, NDArray]) -> float:
        pass