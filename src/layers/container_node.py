from src.node import Node


class ContainerNode(Node):
    def __init__(self, primitive):
        super().__init__(primitive)
        # we can add specific members here
        self.memory = primitive.memory._Memory__impl

    def execute(self):
        self.output_memory = self.memory.copy()

