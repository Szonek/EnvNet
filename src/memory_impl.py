import numpy as np
import copy

from src.utils.error_handler import  ErrorHandler


class MemoryImpl:
    """
    If attaching this class to nodes (i.e weights) please
        add doc with memory layout desciription (what every dimensions mean).
    """
    def __init__(self, shape):
        ErrorHandler.is_type_generic(shape, tuple)
        self.shape = shape
        self.__data = np.empty(shape)

    def fill_data(self, new_data):
        copied_data = new_data
        if isinstance(copied_data, list):
            copied_data = np.asarray(new_data, dtype=float)
        else:
            ErrorHandler.is_type_generic(copied_data, np.ndarray)
            ErrorHandler.is_equal(self.shape, copied_data.shape)
        self.__data = copied_data


    def get_original_data(self):
        return self.__data

    """
    It doesnt return original data. It copies the data.
    """
    def get_data(self):
        return copy.deepcopy(self.__data)

    def size(self):
        return np.prod(self.shape)

    def get_shape(self):
        return self.shape

    def copy(self):
        new_memory = MemoryImpl(self.shape)
        new_data = self.get_data()
        new_memory.fill_data(new_data)
        return new_memory
