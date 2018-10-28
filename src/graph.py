from src.utils.error_handler import ErrorHandler
from src.node import Node


class Graph:
    def __init__(self):
        self.nodes_map = {}     # map {id, node}
        self.inputs = []        # points to input nodes
        self.outputs = []       # points to output nodes

    def add_node(self, node):
        ErrorHandler.is_type_generic(node, Node)
        if node.id in self.nodes_map:
            ErrorHandler.raise_error(node.id + "is already in nodes_map!!!")
        self.nodes_map[node.id] = node
