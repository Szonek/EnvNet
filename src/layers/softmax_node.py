from src.node import Node
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl
import numpy as np


class SoftmaxNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)

    def execute(self):
        input_memory = self.dependencies[0].output_memory
        if len(input_memory.shape) != 4:
            ErrorHandler.raise_error(self.id, " input memory needs to 4 dim!")
        inp_data = input_memory.get_original_data()
        output_shape = input_memory.get_shape()
        self.output_memory = MemoryImpl(output_shape)
        out_data = self.output_memory.get_original_data()
        for N in range(input_memory.shape[0]):
            sum_batch = np.sum(np.exp(inp_data[N]))
            for C in range(input_memory.shape[1]):
                for H in range(input_memory.shape[2]):
                    for W in range(input_memory.shape[3]):
                        out_data[N][C][H][W] = np.exp(inp_data[N][C][H][W])/sum_batch
        pass
