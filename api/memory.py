from src.memory_impl import MemoryImpl


class Memory:
    """
    Placeholder for data (input, weights, constans etc.)
    Shape is a tuple describing the memory layout.
    """
    def __init__(self, shape):
        self.__impl = MemoryImpl(shape)

    def fill(self, new_data):
        self.__impl.fill_data(new_data)

    def size(self):
        return self.__impl.size()

