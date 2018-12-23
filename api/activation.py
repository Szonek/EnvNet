from src.primitive_base import PrimitiveBase
from enum import Enum
from src.utils.error_handler import ErrorHandler


class ActivationFunctions(Enum):
    """
    NONE - copies the data to the output
    RELU - max(0, input)
    """
    NONE = 0
    RELU = 1


class Activation(PrimitiveBase):
    """
    Activation layer.
    Activation function should be proper Enum: ActivationFunctions.
    Output size = input size.
    """
    def __init__(self, id, input, activation_function):
        ErrorHandler.is_type_generic(activation_function, ActivationFunctions)
        super().__init__(id, [input])
        self.activation = activation_function
