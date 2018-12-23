from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Concatenation(PrimitiveBase):
    """
    Concat inputs along axis dimension.
    Input sizes should be the same (beside axis dimension)
    """
    def __init__(self, id, inputs, axis=0):
        ErrorHandler.is_list(inputs)
        ErrorHandler.is_type_generic(axis, int)
        super().__init__(id, inputs)
        self.axis = axis
