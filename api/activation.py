from src.primitive_base import PrimitiveBase
from enum import Enum
from src.utils.error_handler import ErrorHandler


class ActivationFunctions(Enum):
    NONE = 0
    RELU = 1


class Activation(PrimitiveBase):
    def __init__(self, id, input, activation_function):
        ErrorHandler.is_type_generic(activation_function, ActivationFunctions)
        super().__init__(id, [input])
        self.activation = activation_function
