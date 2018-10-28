from src.network_impl import NetworkImpl

class Network:
    """
    Placeholder for primitives and whole graph.
    Allows for execution of topologies (CNN networks).

    Attributes:
        primitives (list): List of primitives. Network is builded upon
            this list. No more primitive will be added to the network.
    """
    def __init__(self, primitives, dump_graph=False):
        self.__impl = NetworkImpl(primitives, dump_graph)

    """
    Just an example function.
    """
    def execute(self):
        return self.__impl.execute()