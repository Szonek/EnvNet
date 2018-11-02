from src.node import Node
from api.pooling import PoolingType
from src.utils.error_handler import ErrorHandler
from src.memory_impl import MemoryImpl
import numpy as np


class PoolingNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.pooling_shape = primitive.pooling_shape
        self.pooling_type = primitive.pooling_type
        self.stride = primitive.stride
        if self.stride is None:
            self.stride = self.pooling_shape

    def do_pooling(self, values):
        if self.pooling_type == PoolingType.MAX:
            return max(values)
        else:
            ErrorHandler.raise_error("[ERROR] Pooling type for: " + self.id + " not implemented.")

    def calc_output_shape(self, input_shape):
        ErrorHandler.is_type_generic(input_shape, tuple)
        nout = input_shape[0]
        cout = input_shape[1]
        hout = int((input_shape[2] - (self.pooling_shape[0]-1)-1)/self.stride[0] + 1)
        wout = int((input_shape[3] - (self.pooling_shape[1]-1)-1)/self.stride[1] + 1)
        return (nout, cout, hout, wout)

    def execute(self):
        input_memory = self.dependencies[0].output_memory
        in_H = input_memory.shape[2]
        in_W = input_memory.shape[3]
        input_data = input_memory.get_data()
        if len(input_memory.shape) != 4:
            ErrorHandler.raise_error(self.id, " input memory needs to 4 dim!")
        out_N, out_C, out_H, out_W = self.calc_output_shape(input_memory.shape)
        self.output_memory = MemoryImpl((out_N, out_C, out_H, out_W))
        out_data = self.output_memory.get_original_data()
        for N in range(out_N):
            for C in range(out_C):
                for H in range(out_H):
                    for W in range(out_W):
                        inp_idx_h = H * self.stride[0]
                        inp_idx_w = W * self.stride[1]
                        values = []
                        for i in range(self.pooling_shape[0]):
                            if inp_idx_h+i < in_H:
                                for j in range(self.pooling_shape[1]):
                                    if inp_idx_w+j < in_W:
                                        values.append(input_data[N][C][inp_idx_h+i][inp_idx_w+j])
                        out_data[N][C][H][W] = self.do_pooling(values)



