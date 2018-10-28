from api.network import Network
from api.activation import Activation, ActivationFunctions
from api.container import Container
from api.memory import Memory
import numpy as np

# ---- TEST CONTAINER --------

c = Container("input", Memory((3, 32, 32)))
a = Activation("Activ", "input", ActivationFunctions.RELU)

c.fill(np.ones((3, 32, 32)))

net2 = Network([c, a])
net2.execute() #not working, need to add topological sort
# ---- END --------

print("end tests")