from __future__ import unicode_literals
import unicodedata
import string
from api.network import Network
from api.container import Container
from api.memory import Memory
from api.linear import Linear
from api.concatenation import Concatenation
from api.softmax import Softmax
from api.reshape import Reshape
import numpy as np


class Model:
    """
    This class defines the NameNEt architecture.
    It uses EnvNet library.
    Goal of this model is to predict country of the name.
    Input size: tensor of letter 1 x n_letters
    Output size: n_categories (number of languages)
    """
    def __init__(self, weights_folder="", dump_graph=False):
        self.input_layer = None
        self.input_hidden_layer = None
        self.output_values = []
        self.net = self.__create_model(weights_folder, dump_graph)
        self.all_letters = string.ascii_letters + " .,;'"
        self.categories = ["Arabic", "Chinese", "Czech", "Dutch", "English",
                           "French", "German", "Greek", "Irish", "Italian",
                           "Japanese", "Korean", "Polish", "Portuguese",
                           "Russian", "Scottish", "Spanish", "Vietnamese"]

    def __create_model(self, weights_folder, dump_graph):
        layers = []
        #input
        self.input_layer = Container("input", Memory((1, 57)))
        self.input_hidden_layer = Container("in_hidden", Memory((1, 128)))
        layers.extend([self.input_layer, self.input_hidden_layer])

        concat = Concatenation("concat", ["input", "in_hidden"], 1)
        layers.append(concat)

        if weights_folder[-1] != "/" or \
            weights_folder[-2:] != "\\":
            weights_folder += "/"

        weights_fc_1 = Container("weights_fc_1", Memory((128, 185)))
        biases_fc_1 = Container("biases_fc_1", Memory((128,)))
        weights_fc_1.fill(np.loadtxt(weights_folder + "i2h.weight").reshape((128, 185)))
        biases_fc_1.fill(np.loadtxt(weights_folder + "i2h.bias").reshape((128,)))
        out_hidden = Linear("i2h", "concat", "weights_fc_1", "biases_fc_1")
        layers.extend([out_hidden, weights_fc_1, biases_fc_1])

        weights_fc_2 = Container("weights_fc_2", Memory((18, 185)))
        biases_fc_2 = Container("biases_fc_2", Memory((18,)))
        weights_fc_2.fill(np.loadtxt(weights_folder + "i2o.weight").reshape((18, 185)))
        biases_fc_2.fill(np.loadtxt(weights_folder + "i2o.bias").reshape((18,)))
        i2o = Linear("i2o", "concat", "weights_fc_2", "biases_fc_2")
        layers.extend([i2o, weights_fc_2, biases_fc_2])

        pre_out_reshape = Reshape("pre_out_reshape", "i2o", (1, 1, 1, 18))
        output = Softmax("output", "pre_out_reshape", do_log=True)
        layers.extend([pre_out_reshape, output])

        return Network(layers, dump_graph=dump_graph)

    def __unicode_to_ascii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn' and c in self.all_letters)

    def word_to_tensor(self, word):
        tensor = np.zeros((len(word), 1, len(self.all_letters)))
        for li, letter in enumerate(word):
            tensor[li][0][self.all_letters.find(letter)] = 1
        return tensor

    def set_input(self, input_data):
        self.input_layer.fill(input_data)

    def set_hidden(self, input_data):
        self.input_hidden_layer.fill(input_data)

    def execute(self):
        self.output_values = self.net.execute()
        return self.output_values

    def get_output(self):
        return self.output_values
