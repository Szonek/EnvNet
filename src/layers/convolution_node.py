from src.node import Node
from api.convolution import Convolution
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl
import numpy as np


class ConvolutionNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)

    def calc_output_shape(self):
        batch = self.input(0).output_memory.shape[0]
        in_wh = self.input(0).output_memory.shape[2]
        out_ch = self.input(1).output_memory.shape[0]
        kernel_size = self.input(1).output_memory.shape[2]
        if self.input(0).output_memory.shape[2] != self.input(0).output_memory.shape[3]:
            ErrorHandler.raise_error("Currently unsymetric input size is not supported.")
        if self.input(1).output_memory.shape[2] != self.input(1).output_memory.shape[3]:
            ErrorHandler.raise_error("Currently unsymetric kernel size is not supported.")
        if self.input(0).output_memory.shape[1] != self.input(1).output_memory.shape[1]:
            ErrorHandler.raise_error("Diffrent size of features for input and weights!")
        hw_out = (in_wh - kernel_size) + 1
        return (batch, out_ch, hw_out, hw_out)

    def do_convolution(self, input_values, weights_values):
        out = 0
        w_a_flatten = weights_values.flatten()
        for i in range(len(input_values)):
            out += input_values[i] * w_a_flatten[i]
        return out

    def execute(self):
        output_shape = self.calc_output_shape()
        input_memory = self.dependencies[0].output_memory
        input_ch = self.input(0).output_memory.shape[1]
        self.output_memory = MemoryImpl(output_shape)
        stride = 1
        kernel_size = self.input(1).output_memory.shape[2]
        out_data = self.output_memory.get_original_data()

        weights_data = self.dependencies[1].output_memory.get_data()
        bias_data = self.dependencies[2].output_memory.get_data()
        in_H = input_memory.shape[2]
        in_W = input_memory.shape[3]
        input_data = input_memory.get_data()
        for N in range(output_shape[0]):
            for C in range(output_shape[1]):
                bias_value = bias_data[C]
                for H in range(output_shape[2]):
                    for W in range(output_shape[3]):
                        out_data[N][C][H][W] = bias_value

        for N in range(output_shape[0]):
            for C in range(output_shape[1]):
                for IN_C in range(input_ch):
                    weights_values = weights_data[C][IN_C]
                    for H in range(output_shape[2]):
                        for W in range(output_shape[3]):
                            inp_idx_h = H * stride
                            inp_idx_w = W * stride
                            values = []
                            for i in range(kernel_size):
                                if inp_idx_h+i < in_H:
                                    for j in range(kernel_size):
                                        if inp_idx_w+j < in_W:
                                            values.append(input_data[N][IN_C][inp_idx_h + i][inp_idx_w + j])
                            out_data[N][C][H][W] += self.do_convolution(values, weights_values)