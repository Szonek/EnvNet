from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Concatenation(PrimitiveBase):
    def __init__(self, id, inputs, axis=0):
        ErrorHandler.is_list(inputs)
        ErrorHandler.is_type_generic(axis, int)
        super().__init__(id, inputs)
        self.axis = axis
