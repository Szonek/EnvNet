from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler


class Convolution(PrimitiveBase):
    """
    Performs convolution over input data.
    Weights_id and bias_id have to be id's of the
    proper container nodes.
    """
    def __init__(self, id, input, weights_id, bias_id):
        ErrorHandler.is_string(weights_id)
        ErrorHandler.is_string(bias_id)
        super().__init__(id, [input, weights_id, bias_id])
