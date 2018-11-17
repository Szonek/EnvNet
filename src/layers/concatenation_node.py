from src.node import Node
from src.memory_impl import MemoryImpl
import numpy as np


class ConcatenationNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.axis = primitive.axis

    def execute(self):
        inputs_mems = []
        for dep in self.dependencies:
            inputs_mems.append(dep.output_memory.get_original_data())
        concated = np.concatenate(inputs_mems, axis=self.axis)
        self.output_memory = MemoryImpl(concated.shape)
        self.output_memory.fill_data(concated)
