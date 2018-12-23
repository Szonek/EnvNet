from api.network import Network
from api.convolution import Convolution
from api.container import Container
from api.memory import Memory
from api.activation import Activation, ActivationFunctions
from api.pooling import Pooling, PoolingType
from api.reshape import Reshape
from api.linear import Linear
import numpy as np


class Model:
    def __init__(self, weights_folder="", dump_graph=False):
        self.input_layer = None
        self.output_values = []
        self.net = self.__create_model(weights_folder, dump_graph)

    def get_input_size(self):
        return 32

    def __create_model(self, weights_folder, dump_graph):
        layers = []
        #input
        self.input_layer = Container("input", Memory((1, 1, 32, 32)))

        layers.append(self.input_layer)

        if weights_folder[-1] != "/" or \
            weights_folder[-2:] != "\\":
            weights_folder += "/"

        #first convolution
        weights_1 = Container("weights_1", Memory((6, 1, 5, 5)))
        biases_1 = Container("biases_1", Memory((6,)))
        weights_1.fill(np.loadtxt(weights_folder + "conv1.weight").reshape((6, 1, 5, 5)))
        biases_1.fill(np.loadtxt(weights_folder + "conv1.bias").reshape((6,)))
        conv_1 = Convolution("conv_1", "input", "weights_1", "biases_1")
        relu_1 = Activation("activ_1", "conv_1", ActivationFunctions.RELU)
        pool_1 = Pooling("pool_1", "activ_1", (2, 2), PoolingType.MAX)
        layers.extend([weights_1, biases_1, conv_1, relu_1, pool_1])

        #2nd convolution
        weights_2 = Container("weights_2", Memory((16, 6, 5, 5)))
        biases_2 = Container("biases_2", Memory((16,)))
        weights_2.fill(np.loadtxt(weights_folder + "conv2.weight").reshape((16, 6, 5, 5)))
        biases_2.fill(np.loadtxt(weights_folder + "conv2.bias").reshape((16,)))
        conv_2 = Convolution("conv_2", "pool_1", "weights_2", "biases_2")
        relu_2 = Activation("activ_2", "conv_2", ActivationFunctions.RELU)
        pool_2 = Pooling("pool_2", "activ_2", (2, 2), PoolingType.MAX)
        layers.extend([weights_2, biases_2, conv_2, relu_2, pool_2])

        #reshape
        reshape = Reshape("reshape", "pool_2", (1, 400))
        layers.append(reshape)

        #first fc
        weights_fc_1 = Container("weights_fc_1", Memory((120, 400)))
        biases_fc_1 = Container("biases_fc_1", Memory((120,)))
        weights_fc_1.fill(np.loadtxt(weights_folder + "fc1.weight").reshape((120, 400)))
        biases_fc_1.fill(np.loadtxt(weights_folder + "fc1.bias").reshape((120,)))
        fc_1 = Linear("fc_1", "reshape", "weights_fc_1", "biases_fc_1")
        relu_fc_1 = Activation("relu_fc_1", "fc_1", ActivationFunctions.RELU)
        layers.extend([weights_fc_1, biases_fc_1, fc_1, relu_fc_1])

        #second fc
        weights_fc_2 = Container("weights_fc_2", Memory((84, 120)))
        biases_fc_2 = Container("biases_fc_2", Memory((84,)))
        weights_fc_2.fill(np.loadtxt(weights_folder + "fc2.weight").reshape((84, 120)))
        biases_fc_2.fill(np.loadtxt(weights_folder + "fc2.bias").reshape((84,)))
        fc_2 = Linear("fc_2", "relu_fc_1", "weights_fc_2", "biases_fc_2")
        relu_fc_2 = Activation("relu_fc_2", "fc_2", ActivationFunctions.RELU)
        layers.extend([weights_fc_2, biases_fc_2, fc_2, relu_fc_2])

        #third fc
        weights_fc_3 = Container("weights_fc_3", Memory((10, 84)))
        biases_fc_3 = Container("biases_fc_3", Memory((10,)))
        weights_fc_3.fill(np.loadtxt(weights_folder + "fc3.weight").reshape((10, 84)))
        biases_fc_3.fill(np.loadtxt(weights_folder + "fc3.bias").reshape((10,)))
        fc_3 = Linear("output", "relu_fc_2", "weights_fc_3", "biases_fc_3")
        layers.extend([weights_fc_3, biases_fc_3, fc_3])

        return Network(layers, dump_graph=dump_graph)

    def set_input(self, input_data):
        self.input_layer.fill(input_data)

    def execute(self):
        self.output_values = self.net.execute()
        return self.output_values

    def get_output(self):
        return self.output_values