from src.primitive_base import PrimitiveBase


class Softmax(PrimitiveBase):
    """
    Performs softmax over input over batches.
    """
    def __init__(self, id, input):
        super().__init__(id, [input])
