from src.primitive_base import PrimitiveBase


class Softmax(PrimitiveBase):
    """
    Performs softmax over input over batches.
    """
    def __init__(self, id, input, do_log):
        super().__init__(id, [input])
        self.do_log = do_log
