import unittest
from api.network import Network
from api.softmax import Softmax
from api.container import Container
from api.memory import Memory
import numpy as np


class TestSoftmax(unittest.TestCase):

    def test_simple_batch_1(self):
        input = Container("input", Memory((1, 1, 4, 4)))
        output = Softmax("output", "input")

        inp_array = np.array([1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12,
                             13, 14, 15, 16]).reshape((1, 1, 4, 4))
        input.fill(inp_array)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertEqual(len(net_out["output"].shape), 4)
        self.assertAlmostEqual(1.0, np.sum(net_out["output"]))

    def test_simple_batch_8(self):
        input = Container("input", Memory((8, 2, 4, 4)))
        output = Softmax("output", "input")

        inp_array = np.arange(8*2*4*4).reshape((8, 2, 4, 4))
        input.fill(inp_array)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertEqual(len(net_out["output"].shape), 4)
        out = net_out["output"]
        sum = np.sum(out)
        self.assertAlmostEqual(8.0, sum)