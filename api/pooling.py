from src.primitive_base import PrimitiveBase
from enum import Enum
from src.utils.error_handler import ErrorHandler


class PoolingType(Enum):
    """
    MAX - get max value from pooling window
    """
    MAX = 0


class Pooling(PrimitiveBase):
    """
    Perfroms pooling mode over input.
    Pooling shape - has to be 2d tuple (currently square size).
    Pooling type - PoolingType Enum.
    Stride - curently not supported.
    """
    def __init__(self, id, input, pooling_shape, pooling_type, stride=None):
        ErrorHandler.is_type_generic(pooling_shape, tuple)
        ErrorHandler.is_type_generic(pooling_type, PoolingType)
        super().__init__(id, [input])
        self.pooling_shape = pooling_shape
        self.pooling_type = pooling_type
        self.stride = stride
