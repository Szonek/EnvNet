from src.utils.error_handler import ErrorHandler


class PrimitiveBase:
    """
    Base class of every primitive base.
    """
    def __init__(self, id, inputs):
        ErrorHandler.is_string(id)
        ErrorHandler.is_list(inputs)
        self.id = id
        self.inputs = inputs
