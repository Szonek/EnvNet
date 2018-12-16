from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Softmax(PrimitiveBase):
    def __init__(self, id, input):
        super().__init__(id, [input])
