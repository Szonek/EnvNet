from api.network import Network
from api.reshape import Reshape
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

input = Container("input", Memory((1, 1, 5, 5)))

inp_array = np.array([1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5]).reshape((1, 1, 5, 5))
input.fill(inp_array)

reshape = Reshape("reshape", "input", (25,))


net2 = Network([input, reshape], dump_graph=False)
outputs = net2.execute()
print(outputs)
# ---- END --------

print("end tests")