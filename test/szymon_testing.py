from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.concatenation import Concatenation
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

c = Container("A", Memory((10,)))
a1 = Activation("B", "A", ActivationFunctions.RELU)
a2 = Activation("D", "B", ActivationFunctions.RELU)

concat = Concatenation("concat", ["B", "D"], axis=0)

a3 = Activation("F", "concat", ActivationFunctions.RELU)

array = [1, 2, 3, -4, -5, -6, 0, 8, 9, -10]

c.fill(array)

net2 = Network([a1, a2, concat,  a3, c], dump_graph=True)
outputs = net2.execute()
print(outputs)
# ---- END --------

print("end tests")