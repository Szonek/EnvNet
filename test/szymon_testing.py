from api.network import Network
from api.softmax import Softmax
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

c = Container("A", Memory((2, 1, 1, 5)))
a1 = Softmax("output", "A")

inp_array = np.array([1, 2, 3, -4, -5, -6, 0, 8, 9, -10]).reshape((2, 1, 1, 5))

c.fill(inp_array)

net2 = Network([c, a1], dump_graph=True)
outputs = net2.execute()
sum = np.sum(outputs["output"])
print(sum)
print(outputs)
# ---- END --------

print("end tests")