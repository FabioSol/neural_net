from abc import ABC, abstractmethod

from numpy._typing import NDArray


class AbstractModel(ABC):
    @abstractmethod
    def fit(self, X: NDArray, Y: NDArray) -> None:
        pass

    @abstractmethod
    def predict(self, x: NDArray) -> NDArray:
        pass