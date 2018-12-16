from api.network import Network
from api.convolution import Convolution
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

input = Container("input", Memory((1, 1, 5, 5)))
weights = Container("weights", Memory((1, 1, 2, 2)))  #out ch, feature, ks_y, ks_x
biases = Container("biases", Memory((1,)))
conv = Convolution("output", "input", "weights", "biases")

inp_array = np.array([1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3,
                      4, 4, 4, 4, 4,
                      5, 5, 5, 5, 5]).reshape((1, 1, 5, 5))

weights_array = np.array([1, 1,
                          1, 1]).reshape((1, 1, 2, 2))

input.fill(inp_array)
weights.fill(weights_array)
biases.fill([1])
net2 = Network([input, weights, biases, conv], dump_graph=True)
outputs = net2.execute()
sum = np.sum(outputs["output"])
print(sum)
print(outputs)
# ---- END --------

print("end tests")