from src.node import Node
from api.linear import Linear
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl
import numpy as np


class LinearNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)

    def calc_output_shape(self):
        batch = self.input(0).output_memory.shape[0]
        out_ch = self.input(1).output_memory.shape[0]
        return (batch, out_ch)

    def execute(self):
        output_shape = self.calc_output_shape()
        input_memory = self.dependencies[0].output_memory
        inp_data = input_memory.get_data()
        inp_ch = self.input(0).output_memory.shape[1]

        self.output_memory = MemoryImpl(output_shape)
        out_data = self.output_memory.get_original_data()
        out_ch = output_shape[1]
        weights_data = self.dependencies[1].output_memory.get_data()
        bias_data = self.dependencies[2].output_memory.get_data()
        batch = output_shape[0]
        for B in range(batch):
            for OUT_CH in range(out_ch):
                out_data[B][OUT_CH] = bias_data[OUT_CH]
        for B in range(batch):
            for OUT_CH in range(out_ch):
                value = 0
                for IN_CH in range(inp_ch):
                    value += inp_data[B][IN_CH] * weights_data[OUT_CH][IN_CH]
                out_data[B][OUT_CH] += value
        pass

