from src.node import Node
from api.activation import ActivationFunctions
from src.utils.error_handler import ErrorHandler


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
        output_shape = self.dependencies[0].output_memory.get_shpae()
        print(output_shape)
