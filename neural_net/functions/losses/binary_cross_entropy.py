from typing import Union

from numpy._typing import NDArray

from neural_net.functions.losses.abstract_loss import AbstractLoss

class BinaryCrossEntropy(AbstractLoss):
    @staticmethod
    def fun(y_true: Union[float, NDArray], y_pred: Union[float, NDArray]) -> float:
        pass

    @staticmethod
    def derivative(y_true: Union[float, NDArray], y_pred: Union[float, NDArray]) -> float:
        pass

