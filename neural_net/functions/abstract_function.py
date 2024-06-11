from abc import ABC, abstractmethod

class AbstractFunction(ABC):

    @staticmethod
    @abstractmethod
    def fun(*args,**kwargs):
        pass

    @staticmethod
    @abstractmethod
    def derivative(*args, **kwargs):
        pass

