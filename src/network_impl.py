import sys
import time

from src.utils.error_handler import ErrorHandler
from src import graph
from src.layers.activation_node import ActivationNode


class NetworkImpl:
    def __init__(self, primitives, dump_graph):
        ErrorHandler.is_list(primitives)
        self.graph = graph.Graph()
        # [1] Create nodes.
        self.__create_nodes(primitives)
        # [2] Connect nodes.
        self.graph.make_connections()
        if dump_graph is True:
            self.__dump_graph()
        # [X] Next steps

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
        for id, node in self.graph.nodes_map.items():
            node.execute()
        pass