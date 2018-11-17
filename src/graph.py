from src.utils.error_handler import ErrorHandler
from src.node import Node


class Graph:
    def __init__(self):
        self.nodes_map = {} # map {id, node}
        self.inputs = []   # input (currently one input accepted)
        self.outputs = []   # points to output nodes
        self.exec_order = []

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
        for node_id, node in self.nodes_map.items():
            input_ids = node.primitive.inputs
            if len(input_ids) != 0:
                for inp_id in input_ids:
                    if self.nodes_map.get(inp_id) is None:
                        ErrorHandler.raise_error("There is no " + str(inp_id) + ", which is input to: "+ str(node_id))
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

    def calc_dfs(self):
        for input in self.inputs:
            self.__calc_dfs_visit(input)
        #and update execution numbers
        idx = 0
        for node in self.exec_order:
            node.execution_number = idx
            idx += 1


    def __calc_dfs_visit(self, node):
        for dep in node.dependencies:
            if dep.execution_number == -1:  # if dep already not visited
                self.__calc_dfs_visit(dep)
        if node.execution_number == -1:  # if node not already visited
            self.exec_order.append(node)
            node.execution_number = 0  # for now its just a mark that node has been visited
            for user in node.users:
                self.__calc_dfs_visit(user)