from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

c = Container("input", Memory((10,)))
a = Activation("Activ", "input", ActivationFunctions.RELU)

array = [1, 2, 3, -4, -5, -6, 0, 8, 9, -10]

c.fill(array)

net2 = Network([c, a])
outputs = net2.execute()

# ---- END --------

print("end tests")