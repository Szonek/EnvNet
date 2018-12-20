from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Linear(PrimitiveBase):
    def __init__(self, id, input, weights_id, bias_id):
        ErrorHandler.is_string(weights_id)
        ErrorHandler.is_string(bias_id)
        super().__init__(id, [input, weights_id, bias_id])
