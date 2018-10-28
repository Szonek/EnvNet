from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.memory import Memory

import numpy as np


# ----- TEST NETWORK ---------
a = Activation("Activ", [], ActivationFunctions.RELU)
b = Activation("Activ_2", ["Activ"], ActivationFunctions.NONE)

net = Network([a, b])

net.execute()


# ------- END ---------------------



# ------------- TEST MEMORY ------------

mem = Memory((3, 32, 32))

dummy_data = np.ones((3, 32, 32))

#fill wil numpy array
mem.fill(dummy_data)

#also we can fill with pythonic array
python_array = [[[0 for k in range(32)] for j in range(32)] for i in range(3)]

for i in range(3):
    for y in range(32):
        for x in range(32):
            python_array[i][y][x] = 1

mem.fill(python_array)

print("test mem size: ", mem.size())

# ------------ END -----------------

print("end tests")