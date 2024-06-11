from typing import Union

from numpy._typing import NDArray

from neural_net.functions.abstract_function import AbstractFunction, abstractmethod

class AbstractActivation(AbstractFunction):
    @staticmethod
    @abstractmethod
    def fun(x:Union[float, NDArray])->Union[float, NDArray]:
        pass

    @staticmethod
    @abstractmethod
    def derivative(x: Union[float, NDArray]) -> Union[float, NDArray]:
        pass
