import sys
import time

from src.utils.error_handler import ErrorHandler
from src import graph
from src.layers.activation_node import ActivationNode
from src.layers.container_node import ContainerNode
from src.layers.pooling_node import PoolingNode
from src.layers.concatenation_node import ConcatenationNode
from src.layers.softmax_node import SoftmaxNode
from src.layers.convolution_node import ConvolutionNode
from src.layers.reshape_node import Reshape
from src.layers.linear_node import Linear


class NetworkImpl:
    """
    Implementation of network.
    It has to make all of the optimizations
    and calcatuions over primitives (nodes).
    """
    def __init__(self, primitives, dump_graph):
        ErrorHandler.is_list(primitives)
        self.graph = graph.Graph()
        # [1] Create nodes.
        self.__create_nodes(primitives)
        # [2] Connect nodes.
        self.graph.make_connections()
        # [3] Mark inputs and outputs nodes
        self.graph.set_inputs_and_outputs()
        # [4] Calc DFS (execution order)
        self.graph.calc_dfs()
        if dump_graph is True:
            self.__dump_graph()

    def __create_nodes(self, primitives):
        for prim in primitives:
            primitive_type = type(prim).__name__
            primitive_class_name = primitive_type + "Node"
            module_name = "src.layers." + primitive_type.lower() + "_node"
            typed_node = getattr(sys.modules[module_name], primitive_class_name)
            self.graph.add_node(typed_node(prim))

    def __dump_graph(self):
        graphviz_sir = self.graph.to_graph_viz_format()
        with open("env_net_network_dump_" + str(int(time.time())) + ".env", 'w') as file:
            file.write(graphviz_sir)

    def execute(self):
        network_outputs = {}
        for node in self.graph.exec_order:
            node.execute()
            if node.is_output is True:
                network_outputs[node.id] = node.output_memory.get_data()
        return network_outputs
