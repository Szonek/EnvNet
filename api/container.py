from src.primitive_base import PrimitiveBase
from src.utils.error_handler import ErrorHandler
from api.memory import Memory


class Container(PrimitiveBase):
    """
    Placeholder for the data.
    It can contain input data or weights data.
    """
    def __init__(self, id, memory):
        ErrorHandler.is_type_generic(memory, Memory)
        super().__init__(id, [])
        self.memory = memory

    def fill(self, new_data):
        self.memory.fill(new_data)
