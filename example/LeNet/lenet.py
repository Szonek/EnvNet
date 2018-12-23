from api.network import Network
from api.convolution import Convolution
from api.container import Container
from api.memory import Memory
from api.activation import Activation, ActivationFunctions
from api.pooling import Pooling, PoolingType
from api.reshape import Reshape
from api.linear import Linear
from api.softmax import Softmax
import numpy as np
import pandas as pd
import cv2
import scipy.ndimage.interpolation as sp
class LeNet:
    def __init__(self):
        layers = []
        #input
        self.input_env = Container("input", Memory((1, 1, 32, 32)))

        layers.append(self.input_env)

        #first convolution
        weights_1 = Container("weights_1", Memory((6, 1, 5, 5)))  #out ch, feature, ks_y, ks_x
        biases_1 = Container("biases_1", Memory((6,)))
        weights_1.fill(np.loadtxt("weights2/conv1.weight").reshape((6, 1, 5, 5)))
        biases_1.fill(np.loadtxt("weights2/conv1.bias").reshape((6,)))
        conv_1 = Convolution("conv_1", "input", "weights_1", "biases_1")
        relu_1 = Activation("activ_1", "conv_1", ActivationFunctions.RELU)
        pool_1 = Pooling("pool_1", "activ_1", (2, 2), PoolingType.MAX)
        layers.extend([weights_1, biases_1, conv_1, relu_1, pool_1])

        #2nd convolution
        weights_2 = Container("weights_2", Memory((16, 6, 5, 5)))  #out ch, feature, ks_y, ks_x
        biases_2 = Container("biases_2", Memory((16,)))
        weights_2.fill(np.loadtxt("weights2/conv2.weight").reshape((16, 6, 5, 5)))
        biases_2.fill(np.loadtxt("weights2/conv2.bias").reshape((16,)))
        conv_2 = Convolution("conv_2", "pool_1", "weights_2", "biases_2")
        relu_2 = Activation("activ_2", "conv_2", ActivationFunctions.RELU)
        pool_2 = Pooling("pool_2", "activ_2", (2, 2), PoolingType.MAX)
        layers.extend([weights_2, biases_2, conv_2, relu_2, pool_2])

        #reshape
        reshape = Reshape("reshape", "pool_2", (1, 400))
        layers.append(reshape)

        #first fc
        weights_fc_1 = Container("weights_fc_1", Memory((120, 400)))
        biases_fc_1 = Container("biases_fc_1", Memory((120,)))
        weights_fc_1.fill(np.loadtxt("weights2/fc1.weight").reshape((120, 400)))
        biases_fc_1.fill(np.loadtxt("weights2/fc1.bias").reshape((120,)))
        fc_1 = Linear("fc_1", "reshape", "weights_fc_1", "biases_fc_1")
        relu_fc_1 = Activation("relu_fc_1", "fc_1", ActivationFunctions.RELU)
        layers.extend([weights_fc_1, biases_fc_1, fc_1, relu_fc_1])

        #second fc
        weights_fc_2 = Container("weights_fc_2", Memory((84, 120)))
        biases_fc_2 = Container("biases_fc_2", Memory((84,)))
        weights_fc_2.fill(np.loadtxt("weights2/fc2.weight").reshape((84, 120)))
        biases_fc_2.fill(np.loadtxt("weights2/fc2.bias").reshape((84,)))
        fc_2 = Linear("fc_2", "relu_fc_1", "weights_fc_2", "biases_fc_2")
        relu_fc_2 = Activation("relu_fc_2", "fc_2", ActivationFunctions.RELU)
        layers.extend([weights_fc_2, biases_fc_2, fc_2, relu_fc_2])

        #third fc
        weights_fc_3 = Container("weights_fc_3", Memory((10, 84)))
        biases_fc_3 = Container("biases_fc_3", Memory((10,)))
        weights_fc_3.fill(np.loadtxt("weights2/fc3.weight").reshape((10, 84)))
        biases_fc_3.fill(np.loadtxt("weights2/fc3.bias").reshape((10,)))
        fc_3 = Linear("output", "relu_fc_2", "weights_fc_3", "biases_fc_3")
        layers.extend([weights_fc_3, biases_fc_3, fc_3])

        self.net2 = Network(layers, dump_graph=False)

    def set_input(self, input_data):
        self.input_env.fill(input_data)

    def execute(self):
        return self.net2.execute()
#
# train = pd.read_csv("test.csv").values
# # train = shuffle(train)
# #test  = pd.read_csv("C:\\Users\\szymon\\Desktop\\STUDIA\envnet_reference\\test.csv").values
# nb_batch = 1
# nb_index = 0
# X_data  = train[:, 0:].reshape(train.shape[0], 1, 28, 28)
# #Y = train[:, 0:1]
# X_data  = X_data.astype(float)
# X_data /= 255.0
# #
# net = LeNet()
#
# img_to_network = np.zeros((128, 128, 1))
# while(1):
#     cv2.imshow('image',img_to_network)
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('a'):
#         img = np.pad(X_data[nb_index].reshape(28, 28, 1), ((2, 2), (2, 2), (0, 0)), mode='constant')
#         img_to_network = sp.zoom(img, (4, 4, 1))
#         img_to_network_2 = sp.zoom(img_to_network, (1/4, 1/4, 1))
#         img_to_network_2 = img_to_network_2.reshape((1, 1, 32, 32))
#         net.set_input(img_to_network_2)
#         net_out = net.execute()
#         net_out = net_out["output"]
#         print("net out: ", np.argmax(net_out))
#         #print("reeal out:", Y[nb_index][0])
#         nb_index = nb_index + 1
#     if k == 27:
#         break
# print('end')
