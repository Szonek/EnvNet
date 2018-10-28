from src.utils.error_handler import  ErrorHandler
import numpy as np


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
            copied_data = np.asarray(new_data)
        else:
            ErrorHandler.is_type_generic(copied_data, np.ndarray)
        #TODO: ADD CHECKS HERE, IF THE DATA HAS THE CORRET SHAPE
        self.__data = copied_data



    """
    It doesnt return original data. It copies the data.
    """
    def get_data(self):
        return list(self.__data)

    def size(self):
        return np.prod(self.shape)

    def get_shape(self):
        return self.shape