import numpy as np
import scipy as sp
from collections import OrderedDict

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum_(exp_a)
    return exp_a / sum_exp_a



def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size



def sum_squared_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0.5 * np.sum((y-t)**0.5) / batch_size



class Relu(object):
    """docstring forRelu."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid(object):
    """docstring for Sigmoid."""

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

class Affine(object):
    """docstring for Affine."""

    def __init__(self, x, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx



class SoftmaxWithLoss(object):
    """docstring for SoftmaxWithLoss."""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(output_size)


        self.layers = np.OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        return self.lastLayer.forward(x, t)

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grad = {}
        grad["W1"] = self.layers["Affine1"].dW
        grad["b1"] = self.layers["Affine1"].db
        grad["W2"] = self.layers["Affine2"].dW
        grad["b2"] = self.layers["Affine2"].db

        return grad
