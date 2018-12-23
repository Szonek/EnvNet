from src.node import Node
from api.reshape import Reshape
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl


class ReshapeNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.new_shape = primitive.new_shape

    def execute(self):
        input_memory = self.dependencies[0].output_memory
        self.output_memory = MemoryImpl(self.new_shape)
        self.output_memory.fill_data(input_memory.get_data().reshape(self.new_shape))
