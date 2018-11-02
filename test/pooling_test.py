import unittest
from api.network import Network
from api.pooling import Pooling, PoolingType
from api.container import Container
from api.memory import Memory
import numpy as np


class TestPooling(unittest.TestCase):

    def test_simple_MAX_0(self):
        input = Container("input", Memory((1, 1, 4, 4)))
        output = Pooling("output", "input", (2, 2), PoolingType.MAX)

        inp_array = np.array([1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12,
                             13, 14, 15, 16]).reshape((1, 1, 4, 4))
        input.fill(inp_array)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertEqual(len(net_out["output"].shape), 4)
        out_reference = np.array([6.0, 8.0, 14.0, 16.0]).reshape((1, 1, 2, 2))
        self.assertTrue(np.array_equal(out_reference, net_out["output"]))

    def test_simple_MAX_1(self):
        input = Container("input", Memory((2, 2, 4, 4)))
        output = Pooling("output", "input", (2, 2), PoolingType.MAX)

        inp_array = np.arange(2*2*4*4).reshape((2, 2, 4, 4))
        input.fill(inp_array)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertEqual(len(net_out["output"].shape), 4)
        out_reference = np.array([5, 7, 13, 15, 21, 23, 29, 31, 37, 39, 45, 47, 53, 55, 61, 63]).reshape((2, 2, 2, 2))
        self.assertTrue(np.array_equal(out_reference, net_out["output"]))

    def test_stride(self):
        input = Container("input", Memory((1, 1, 4, 4)))
        output = Pooling("output", "input", (2, 2), PoolingType.MAX, stride=(3, 3))

        inp_array = np.arange(4*4).reshape((1, 1, 4, 4))
        input.fill(inp_array)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertEqual(len(net_out["output"].shape), 4)
        out_reference = np.array([5]).reshape((1, 1, 1, 1))
        self.assertTrue(np.array_equal(out_reference, net_out["output"]))