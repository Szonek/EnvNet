import unittest
from api.network import Network
from api.concatenation import Concatenation
from api.container import Container
from api.memory import Memory
import numpy as np


class TestConcatenation(unittest.TestCase):

    def setUp(self):
        pass

    def test_concat_batch(self):
        input0 = Container("input0", Memory((3, 1, 2, 2)))
        inp0_arr = np.array([0, 0, 0, 0,
                              1, 1, 1, 1,
                              2, 2, 2, 2]).reshape((3, 1, 2, 2))
        input0.fill(inp0_arr)

        input1 = Container("input1", Memory((2, 1, 2, 2)))
        inp1_arr = np.array([13, 13, 13, 13,
                              24, 24, 24, 24]).reshape((2, 1, 2, 2))
        input1.fill(inp1_arr)

        concat = Concatenation("concat", ["input0", "input1"], axis=0)

        layers = [input0, input1, concat]

        network = Network(layers)
        net_out = network.execute()
        real_output = net_out["concat"]
        self.assertEqual(len(net_out), 1)
        self.assertIsNotNone(real_output)
        ref_output = np.concatenate((inp0_arr, inp1_arr), axis=0)
        self.assertEqual(real_output.shape, (5, 1, 2, 2))
        self.assertTrue(np.array_equal(ref_output, real_output))