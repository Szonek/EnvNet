from src.node import Node
from src.memory_impl import MemoryImpl


class ContainerNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.memory = primitive.memory

    def execute(self):
        print("test execute container node")
