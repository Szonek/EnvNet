import unittest
from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.container import Container
from api.memory import Memory
import numpy as np
import copy


class TestActivation(unittest.TestCase):

    def setUp(self):
        pass

    def relu_func(self, value):
        return max(0, value)

    def get_reference(self, activ_function, input_values):
        input_values_copy = copy.deepcopy(input_values)
        if activ_function == ActivationFunctions.NONE:
            return input_values_copy
        elif activ_function == ActivationFunctions.RELU:
            func = np.vectorize(self.relu_func)
            return func(input_values_copy)
        raise Exception("Not implemented activation func")

    def test_simple_RELU(self):
        input = Container("input", Memory((3,)))
        output = Activation("output", "input", ActivationFunctions.RELU)
        input.fill([-5, 10, -0.5])
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertIsNotNone(net_out["output"])
        self.assertEqual(net_out["output"], [0, 10, 0])

    def test_generic_3d(self):
        shapes = [(3, 32, 32), (10, 1, 1), (1, 2, 3), (5, 224, 224)]
        for activ_func in ActivationFunctions:
            for input_shape in shapes:
                input = Container("input", Memory(input_shape))
                output = Activation("output", "input", activ_func)
                input_values = np.random.uniform(-2, 2, input_shape)
                input.fill(input_values)
                network = Network([output, input])
                net_out = network.execute()
                self.assertEqual(len(net_out), 1)
                self.assertIsNotNone(net_out["output"])
                reference_output = self.get_reference(activ_func, input_values)
                real_output = net_out["output"]
                #TODO : add check here