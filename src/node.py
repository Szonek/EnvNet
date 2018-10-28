from abc import ABCMeta
from abc import abstractmethod
from src.utils.error_handler import ErrorHandler


class Node:
    __metaclass__ = ABCMeta

    def __init__(self, prim):
        self.id = prim.id
        self.primitive = prim
        self.users = []
        self.dependencies = []
        self.output_memory = None

    def input(self, idx=0):
        if idx > len(self.dependencies)-1:
            ErrorHandler.raise_error("No input with index: ", idx)
        return self.users[idx]

    @abstractmethod
    def execute(self):
        ErrorHandler.raise_error("[ERROR] Execute not implemented for: " + self.id + "!!!")