import numpy as np
import scipy as sp
from collections import OrderedDict

import cupy as cp

def softmax(x):
    x = x - cp.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
    x = cp.exp(x) / cp.sum(cp.exp(x), axis=-1, keepdims=True)

    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -cp.sum(t * cp.log(y + 1e-12)) / batch_size



def sum_squared_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0.5 * cp.sum(y-t)**2 / batch_size


def sigmoid(x):

    # x = x - np.min(x, axis=-1, keepdims=True)
    out = 1. / (1. + cp.exp(-x))
    return out



class Relu(object):
    """docstring forRelu."""

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0.
        return out

    def backward(self, dout):
        dout[self.mask] = 0.
        dx = dout
        return dx


class Sigmoid(object):
    """docstring for Sigmoid."""

    def __init__(self):
        self.out = None

    def forward(self, x):
        # x = x - cp.max(x, axis=-1, keepdims=True)
        out = 1. / (1. + cp.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

# class Affine(object):
#     """docstring for Affine."""
#
#     def __init__(self, W, b):
#         self.W = W
#         self.b = b
#         self.x = None
#         self.dW = None
#         self.db = None
#
#     def forward(self, x):
#         self.x = x
#         out = np.dot(x, self.W) + self.b
#
#         return out
#
#     def backward(self, dout):
#         dx = np.dot(dout, self.W.T)
#         self.dW = np.dot(self.x.T, dout)
#         self.db = np.sum(dout, axis=0)
#         return dx
class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = cp.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = cp.dot(dout, self.W.T)
        self.dW = cp.dot(self.x.T, dout)
        self.db = cp.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = cp.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = cp.sum(dout, axis=0)
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = cp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = cp.argmax(col, axis=1)
        out = cp.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = cp.zeros((dout.size, pool_size))
        dmax[cp.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

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
        # self.loss = sum_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx

class SigmoidWithLoss(object):
    """docstring for SoftmaxWithLoss."""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.original_x_shape = x.shape
        self.t = t.reshape(self.original_x_shape)
        # print(t.shape)
        self.y = sigmoid(x).reshape(self.original_x_shape)
        # print(self.y.shape)
        self.loss = cross_entropy_error(self.y, self.t)
        # self.loss = sum_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # print(self.y.shape)
        # print(self.t.shape)

        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        # dx[~self.t] *= 0.1
        if dx.shape == (64,64):raise ValueError
        # dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return dx


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):

        self.params = {}
        self.params["W1"] = weight_init_std * cp.random.randn(input_size, hidden_size)
        self.params["b1"] = cp.random.randn(hidden_size)+2
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = cp.random.randn(output_size)+2


        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        # self.lastLayer = SoftmaxWithLoss()
        self.lastLayer = SigmoidWithLoss()

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)


    def accuracy(self, x, t):
        y = self.predict(x)
        y = cp.argmax(y, axis=1)
        if t.ndim != 1: t = cp.argmax(t, axis=1)

        accuracy = cp.sum(y==t)/float(t.shape[0])

        return accuracy

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


class NLayerNet(object):

    def __init__(self, input_size, hidden_size_list, output_size, weight_init_std=0.1):
        self.device = "gpu"
        self.W_params = []
        self.b_params = []
        w = weight_init_std * cp.random.randn(input_size, hidden_size_list[0])
        b = weight_init_std * cp.random.randn(hidden_size_list[0])
        self.layers = []


        N = len(hidden_size_list)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))
        self.layers.append(Relu())

        for n in range(0,N-1):
            w = weight_init_std * cp.random.randn(hidden_size_list[n], hidden_size_list[n+1])
            b = weight_init_std * cp.random.randn(hidden_size_list[n+1])

            self.W_params.append(w)
            self.b_params.append(b)
            self.layers.append(Affine(w,b))
            self.layers.append(Sigmoid())

        w = weight_init_std * cp.random.randn(hidden_size_list[N-1], output_size)
        b = weight_init_std * cp.random.randn(output_size)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))



        self.lastLayer = SigmoidWithLoss()

    def predict(self, x):
        x = cp.asarray(x)
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # x = cp.asarray(x)
        t = cp.asarray(t)
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        t = cp.asarray(t)
        y = self.predict(x)
        y = sigmoid(y)[:,0]
        y = y>0.5
        print(y.shape)
        print(t.shape)
        # print(np.where(y[0]==t))
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1:t = np.argmax(t, axis=1)
        accuracy = cp.sum(y==t) / float(len(t))
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers)
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grad = []
        for layer in self.layers[::2]:
            g = [layer.dW, layer.db]
            grad.append(g)


        return grad
