from api.network import Network
from api.linear import Linear
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

input = Container("input", Memory((1, 10)))

inp_array = np.array([1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2]).reshape((1, 10))
input.fill(inp_array)

weights = Container("weights", Memory((2, 10)))
bias = Container("bias", Memory((2,)))

weights_array = np.arange(20).reshape((2, 10))
bias_array = np.zeros((2,))

weights.fill(weights_array)
bias.fill(bias_array)
fc = Linear("fc", "input", "weights", "bias")


net2 = Network([input, weights, bias, fc], dump_graph=False)
outputs = net2.execute()
print(outputs)
# ---- END --------

print("end tests")