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

    def set_inputs_and_outputs(self):
        for _, node in self.nodes_map.items():
            if len(node.dependencies) == 0:
                self.inputs.append(node)
            if len(node.users) == 0:
                self.outputs.append(node)
                node.is_output = True

    def make_connections(self):
        for _, node in self.nodes_map.items():
            input_ids = node.primitive.inputs
            if len(input_ids) != 0:
                for inp_id in input_ids:
                    node.dependencies.append(self.nodes_map[inp_id])
                    self.nodes_map[inp_id].users.append(node)

    def to_graph_viz_format(self):
        graphviz_str = "digraph G { \n"
        for node_id, node in self.nodes_map.items():
            graphviz_str += node.id + " "
            graphviz_str += "[label=\"" + node.id + "\\n" + \
                            type(node).__name__ +  "\\n" +\
                            str(node.execution_number) + "\"]\n"
            for dep in node.users:
                graphviz_str += node_id + " -> " + dep.id + "\n"
        graphviz_str += "}"
        return graphviz_str