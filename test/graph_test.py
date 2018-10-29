import unittest
from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.container import Container
from api.memory import Memory


class TestGraph(unittest.TestCase):

    def setUp(self):
        pass

    def test_simple(self):
        input = Container("input", Memory((3,)))
        output = Activation("output", "input", ActivationFunctions.NONE)
        network = Network([output, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 1)
        self.assertIsNotNone(net_out["output"])

    def test_multiple_outputs(self):
        input = Container("A", Memory((3,)))
        b = Activation("B", "A", ActivationFunctions.RELU)
        d = Activation("D", "B", ActivationFunctions.RELU)
        f = Activation("F", "B", ActivationFunctions.RELU)
        c = Activation("C", "A", ActivationFunctions.RELU)
        g = Activation("G", "C", ActivationFunctions.RELU)
        e = Activation("E", "A", ActivationFunctions.RELU)
        network = Network([b, d, f, c, g, e, input])
        net_out = network.execute()
        self.assertEqual(len(net_out), 4)
        self.assertIsNotNone(net_out["D"])
        self.assertIsNotNone(net_out["F"])
        self.assertIsNotNone(net_out["G"])
        self.assertIsNotNone(net_out["E"])