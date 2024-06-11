from typing import Union

from numpy._typing import NDArray

from neural_net.functions.abstract_function import AbstractFunction, abstractmethod

class AbstractLoss(AbstractFunction):

    @staticmethod
    @abstractmethod
    def fun(y_true: Union[float, NDArray],y_pred: Union[float, NDArray]) -> float:
        pass

    @staticmethod
    @abstractmethod
    def derivative(y_true: Union[float, NDArray], y_pred: Union[float, NDArray]) -> float:
        pass