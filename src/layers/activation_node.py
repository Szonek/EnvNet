from src.node import Node
from api.activation import ActivationFunctions
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl
import numpy as np


class ActivationNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.activation_func = primitive.activation

    def do_activation(self, value):
        if self.activation_func == ActivationFunctions.RELU:
            return max(0, value)
        elif self.activation_func == ActivationFunctions.NONE:
            return value
        else:
            ErrorHandler.raise_error("[ERROR] Activation function for: " + self.id + " not implemented.")

    def execute(self):
        input_memory = self.dependencies[0].output_memory
        output_shape = input_memory.get_shape()
        self.output_memory = MemoryImpl(output_shape)
        vectorised_func = np.vectorize(self.do_activation)
        self.output_memory.fill_data(vectorised_func(input_memory.get_data()))
