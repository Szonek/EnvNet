import unittest
from api.network import Network
from api.reshape import Reshape
from api.container import Container
from api.memory import Memory
import numpy as np
import copy


class TestReshape(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple_RELU(self):
        input = Container("input", Memory((1, 1, 5, 5)))
        inp_array = np.array([1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2,
                              3, 3, 3, 3, 3,
                              4, 4, 4, 4, 4,
                              5, 5, 5, 5, 5]).reshape((1, 1, 5, 5))
        input.fill(inp_array)

        reshape = Reshape("reshape", "input", (25,))

        net2 = Network([input, reshape])
        outputs = net2.execute()
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs["reshape"].shape, (25,))

    def test_generic_many_shapes(self):
        input_shape = (8, 3, 32, 32)
        input = Container("input", Memory(input_shape))
        input_values = np.random.uniform(-2, 2, input_shape)
        input.fill(input_values)

        shapes = [(8, 3, 1, 1024), (8, 3, 1024, 1), (24, 1, 32, 32), (1, 1, 1, 24576)]
        for shape in shapes:
            output = Reshape("output", "input", shape)
            network = Network([output, input])
            net_out = network.execute()
            self.assertEqual(len(net_out), 1)
            self.assertIsNotNone(net_out["output"])
            self.assertEqual(net_out["output"].shape, shape)