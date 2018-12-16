from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Reshape(PrimitiveBase):
    def __init__(self, id, input, new_shape):
        ErrorHandler.is_type_generic(new_shape, tuple)
        super().__init__(id, [input])
        self.new_shape = new_shape
