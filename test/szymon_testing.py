from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

c = Container("A", Memory((10,)))
a1 = Activation("B", "A", ActivationFunctions.RELU)
a2 = Activation("D", "B", ActivationFunctions.RELU)
a3 = Activation("F", "B", ActivationFunctions.RELU)
a4 = Activation("C", "A", ActivationFunctions.RELU)
a5 = Activation("G", "C", ActivationFunctions.RELU)
a6 = Activation("E", "A", ActivationFunctions.RELU)

array = [1, 2, 3, -4, -5, -6, 0, 8, 9, -10]

c.fill(array)

net2 = Network([a1, a2, a3, c, a4, a5, a6], dump_graph=True)
outputs = net2.execute()

# ---- END --------

print("end tests")