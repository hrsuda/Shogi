import numpy as np
import scipy as sp
from collections import OrderedDict

import cupy as cp


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True) 
      # オーバーフロー対策
    x = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    return x


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size) #1d->2d
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y)) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)
    # print(y)

    batch_size = y.shape[0]
    return np.sum(-t*np.log(y)-(1-t)*np.log(1-y)) / batch_size


def sum_squared_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size) #1d->2d
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0.5 * np.sum(y-t)**2 / batch_size


def sigmoid(x):
    # print(x)
    # x = x - np.min(x, axis=-1, keepdims=True)
    # x = x - np.max(x, axis=-1, keepdims=True)

    out = 1. / (1. + np.exp(-x))
    return out

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C,filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def im2col_alpha(input_data, filter_h, filter_w, filter_positions, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, len(filter_positions[0]), out_h, out_w))

    for i,p in enumerate(filter_positions.T):
        # print(i)
        x = p[0]
        y = p[1]
        x_max = x + stride*out_w
        y_max = y + stride*out_h
        col[:, :, i, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)


    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]


    return img[:, :, pad:H + pad, pad:W + pad]

def col2im_alpha(col, input_shape, filter_h, filter_w, filter_positions, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (W + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, len(filter_positions[0])).transpose(0, 3, 4, 1, 2)


    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for i,p in enumerate(filter_positions.T):
        x = p[0]
        y = p[1]
        x_max = x + stride*out_w
        y_max = y + stride*out_h
        img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, i, :, :]


    # return img[:, :, pad:H + pad, pad:W + pad][:,:,filter_positions[0],filter_positions[1]]
    return img[:, :, pad:H + pad, pad:W + pad]





class Tanh(object):

    def __init__(self):
        self.mask = None
        self.dW = np.array([0])
        self.db = np.array([0])
        self.W = np.array([0])
        self.b = np.array([0])

        
    def forward(self, x):
        # x = x - np.max(x, axis=-1, keepdims=True) + 1e-12
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = np.cosh(dout)**-2
        return dx


class Relu(object):
    """docstring forRelu."""

    def __init__(self):
        self.mask = None
        self.dW = np.array([0])
        self.db = np.array([0])
        self.W = np.array([0])
        self.b = np.array([0])

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
        self.dW = np.array([0])
        self.db = np.array([0])
        self.W = np.array([0])
        self.b = np.array([0])

    def forward(self, x):
        # x = x - np.max(x, axis=-1, keepdims=True) + 1e-12
        out = 1. / (1. + np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout * (1. - self.out) * self.out
        return dx

class Pass_Layer(object):
    def __init__(self):
        self.out = None
        self.dW = np.array([0])
        self.db = np.array([0])
        self.W = 0
        self.b = 0

    def forward(self, x):
        # x = x - np.max(x, axis=-1, keepdims=True) + 1e-12
        # self.out = out
        return x
    def backward(self, dout):
        return dout

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

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

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
        FN = 1
        C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) 
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        return out



    def backward(self, dout):
        batch_size = dout.shape[0]
        FN = 1 
        C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout.reshape(-1,9,9),axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

class Convolution_with_sigmoid:
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
        FN = 1
        C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        self.out = np.dot(col, col_W) 
        self.out = self.out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = 1/(1-np.exp(self.out))
        return self.out


    def backward(self, dout):
        batch_size = dout.shape[0]
        FN = 1 
        C, FH, FW = self.W.shape
        dout = dout * (1 - self.out) * self.out
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout.reshape(-1,9,9),axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Convolution_alpha_with_Sigmoid:
    def __init__(self, W, b, filter_positions, stride=1, pad=0):
        self.W = W #1d
        self.b = b
        self.stride = stride
        self.pad = pad
        self.filter_positions = filter_positions #not bool

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # FN, C, FH, FW = self.W.shape
        FN = 1
        FH = 17
        FW = 17
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col_alpha(x, FH, FW, self.filter_positions, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        self.out = np.dot(col, col_W)
        self.out = self.out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        self.out = 1/(1-np.exp(self.out))
        return self.out



    def backward(self, dout):
        # batch_size = dout.shape[0]
        FN = 1
        C = self.x.shape[1]
        FH = 17
        FW = 17
        batch_size = dout.shape[0]
        # print(dout.shape)
        dout = dout * (1 - self.out) * self.out
        dout = dout.transpose(0,2,3,1).reshape(-1,1)

        self.db = np.sum(dout.reshape(-1,9,9),axis=0)
        
        self.dW = np.dot(self.col.T, dout)
        # print(self.dW.shape)
        # print(dout.shape)

        self.dW = self.dW.transpose(1, 0).reshape(C*len(self.filter_positions[0]))
        # print(self.dW.shape)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im_alpha(dcol, self.x.shape, FH, FW, self.filter_positions, self.stride, self.pad)

        return dx

class Convolution_alpha:
    def __init__(self, W, b, filter_positions, stride=1, pad=0):
        self.W = W #1d
        self.b = b
        self.stride = stride
        self.pad = pad
        self.filter_positions = filter_positions #not bool

        # 中間データ（backward時に使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # FN, C, FH, FW = self.W.shape
        FN = 1
        FH = 17
        FW = 17
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col_alpha(x, FH, FW, self.filter_positions, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        self.out = np.dot(col, col_W)
        self.out = self.out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        
        return self.out



    def backward(self, dout):
        # batch_size = dout.shape[0]
        FN = 1
        C = self.x.shape[1]
        FH = 17
        FW = 17
        batch_size = dout.shape[0]
        # print(dout.shape)
        dout = dout.transpose(0,2,3,1).reshape(-1,1)

        self.db = np.sum(dout.reshape(-1,9,9),axis=0)
        
        self.dW = np.dot(self.col.T, dout)
        # print(self.dW.shape)
        # print(dout.shape)

        self.dW = self.dW.transpose(1, 0).reshape(C*len(self.filter_positions[0]))
        # print(self.dW.shape)
        dcol = np.dot(dout, self.col_W.T)
        dx = col2im_alpha(dcol, self.x.shape, FH, FW, self.filter_positions, self.stride, self.pad)

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

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = np.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx




class SoftmaxWithLoss(object):
    """docstring for SoftmaxWithLoss."""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.original_x_shape = x.shape
        self.t = t.reshape(self.original_x_shape)
        self.y = softmax(x).reshape(self.original_x_shape)
        # print(self.y)
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
        # print("t")
        # print(t)
        # print('x')
        # print(x)
        # print(t.shape)
        self.x = x
        self.y = sigmoid(x).reshape(self.original_x_shape)
        # print(self.y.shape)
        # print("y")
        # print(self.y)
        self.loss = cross_entropy_error(self.y, self.t)
        # self.loss = sum_squared_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        # print(self.y.shape)
        # print(self.t.shape)

        batch_size = self.t.shape[0]
        # dx = self.y.copy()
        # dx[np.arange(batch_size), self.t] -= 1
        dx = self.y - self.t
        # dx = (self.y - self.t)/ batch_size
        # # dx = self.y**-2 * (self.y**-1 - 1) * dx
        dx = dx * (1 - self.y) * self.y
        dx = dx / batch_size
        # print(dx)

        # dx[~self.t] *= 0.1
        # if dx.shape == (64,64):raise ValueError
        # dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）

        return dx


# class Extend_Input(object):
#     def __init__(self, num):
#         self.shape = None
#         self.dW = 0
#         self.db = 0
#         self.num = num

#     def forward(self, x):
#         self.shape = x.shape

#         self.length = np.prod(self.shape[1:])
#         x = x.reshape(self.shape[0], -1)
#         self.A = np.concatenate([np.identity(self.length)] * self.shape[0], axis=0)
#         self.A_inv = np.linalg.pinv(A)
#     return 




class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):

        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.random.randn(hidden_size)+2
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.random.randn(output_size)+2


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
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y==t)/float(t.shape[0])

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
        w = weight_init_std * np.random.randn(input_size, hidden_size_list[0])
        b = weight_init_std * np.random.randn(hidden_size_list[0])
        self.layers = []


        N = len(hidden_size_list)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))
        self.layers.append(Relu())

        for n in range(0,N-1):
            w = weight_init_std * np.random.randn(hidden_size_list[n], hidden_size_list[n+1])
            b = weight_init_std * np.random.randn(hidden_size_list[n+1])

            self.W_params.append(w)
            self.b_params.append(b)
            self.layers.append(Affine(w,b))
            self.layers.append(Sigmoid())

        w = weight_init_std * np.random.randn(hidden_size_list[N-1], output_size)
        b = weight_init_std * np.random.randn(output_size)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))



        # self.lastLayer = SigmoidWithLoss()
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        x = np.asarray(x)
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # x = np.asarray(x)
        t = np.asarray(t)
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        t = np.asarray(t)
        y = self.predict(x)
        y = sigmoid(y)[:,0]
        y = y>0.5
        # print(y.shape)
        # print(t.shape)
        # print(np.where(y[0]==t))
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1:t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t) / float(len(t))
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


class ShogiLayerNet(object):

    def __init__(self, weight_init_std=0.1):
        self.device = "gpu"
        self.W_params = []
        self.b_params = []
        self.layers = []
        
        self.filter_positions_1 = np.array([[8,8,8,8,8,8,8,8,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,8,8,8,8,8,8,8,8],
                                            [0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,10,11,12,13,14,15,16]])
        self.filter_positions_2 = np.array([[0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0,16],
                                            [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16]])

        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_1[0]))
        b = weight_init_std * np.random.randn(1)
        
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_2[0]))
        b = weight_init_std * np.random.randn(1)

        # print(len(self.filter_positions_2[0]))
        
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

        w = weight_init_std * np.random.randn(2*2*14,3,3)
        b = weight_init_std * np.random.randn(1)
        
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))



        w = weight_init_std * np.random.randn(81+81+81+14*2*2, 1)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))



        self.lastLayer = SigmoidWithLoss()

    def predict(self, x):
        x_a = x[0].reshape([-1,2*2*14,9,9])
        x_b = x[1].reshape([-1,56])
        x1 = self.layers[0].forward(x_a).reshape([-1,81])
        x2 = self.layers[1].forward(x_a).reshape([-1,81])
        x3 = self.layers[2].forward(x_a).reshape([-1,81])

        x = np.concatenate([x1,x2,x3,x_b],axis=1)

        x = self.layers[3].forward(x)

        return x

    def loss(self, x, t):
        # x = np.asarray(x)
        # t = np.asarray(t)
        y = self.predict(x)

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # t = np.asarray(t)
        y = self.predict(x)
        y = sigmoid(y)[:,0]
        y = y>0.5
        # print(y.shape)
        # print(t.shape)
        # print(np.where(y[0]==t))
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1:t = np.argmax(t, axis=1)
        accuracy = (y==t).sum() / float(len(t))
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        dout = self.layers[3].backward(dout)
        # print(dout.shape)
        self.layers[2].backward(dout[:,162:243].reshape([1,-1,9,9]))
        self.layers[1].backward(dout[:,81:162].reshape([1,-1,9,9]))
        self.layers[0].backward(dout[:,0:81].reshape([1,-1,9,9]))
        

        grad_w = []
        grad_b = []
        for layer in self.layers:
            
            grad_w.append(layer.dW)
            grad_b.append(layer.db)


        return grad_w,grad_b

class ShogiLayerNet2(object):

    def __init__(self, weight_init_std=0.1):
        self.device = "gpu"
        self.W_params = []
        self.b_params = []
        self.layers = []
        
        self.filter_positions_1 = np.array([[8,8,8,8,8,8,8,8,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,8,8,8,8,8,8,8,8],
                                            [0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,10,11,12,13,14,15,16]])
        self.filter_positions_2 = np.array([[0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0,16],
                                            [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16]])
        self.filter_positions_3 = np.array([[6,6,8,10,10],
                                            [7,9,8,7,9]])        

#0 hisha 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_1[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

#1 kaku 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_2[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#2 keima 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_3[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#3 sonota 1
        w = weight_init_std * np.random.randn(2*2*14,3,3)
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#4 Last 1
        w = weight_init_std * np.random.randn(81+81+81+81+14*2*2, 100)/(1848+1848+560+504)**0.5
        b = weight_init_std * np.random.randn(100)/(1848+1848+560+504)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))
#5 Rule 1
        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Sigmoid())

#6 hisha 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_1[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

#7 kaku 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#8 keima 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#9 sonota 2
        w = weight_init_std * np.random.randn(4,3,3)/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#10 Last 2
        w = weight_init_std * np.random.randn(81+81+81+81+100, 50)/(4*33*2 + 4*10 + 4*9 + 4*14)**0.5
        b = weight_init_std * np.random.randn(50)/(4*33*2 + 4*10 + 4*9 + 4*14)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))

#11 Rule 2
        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)

        self.layers.append(Sigmoid())

#12 hisha 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_1[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

#13 kaku 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#14 keima 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#15 sonota 3
        w = weight_init_std * np.random.randn(4,3,3)/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#16 Last 3
        w = weight_init_std * np.random.randn(81+81+81+81+50, 1)/(324 + 100)**0.5
        b = weight_init_std * np.random.randn(1)/(324 + 100)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))




        # self.lastLayer = SoftmaxWithLoss()
        self.lastLayer = SigmoidWithLoss()

    def predict(self, x):
        x_a = x[0].reshape([-1,2*2*14,9,9])
        x_b = x[1].reshape([-1,56])
        x1 = self.layers[0].forward(x_a).reshape([-1,81])
        x2 = self.layers[1].forward(x_a).reshape([-1,81])
        x3 = self.layers[2].forward(x_a).reshape([-1,81])
        x4 = self.layers[3].forward(x_a).reshape([-1,81])
        x = np.concatenate([x1,x2,x3,x4,x_b],axis=1)
        x = self.layers[4].forward(x)
        x = self.layers[5].forward(x)

        x_1234 = np.array([x1,x2,x3,x4]).transpose(1,0,2).reshape(-1,4,9,9)
        x1 = self.layers[6].forward(x_1234).reshape([-1,81])
        x2 = self.layers[7].forward(x_1234).reshape([-1,81])
        x3 = self.layers[8].forward(x_1234).reshape([-1,81])
        x4 = self.layers[9].forward(x_1234).reshape([-1,81])

        x = np.concatenate([x1,x2,x3,x4,x],axis=1)

        x = self.layers[10].forward(x)
        x = self.layers[11].forward(x)

        x_1234 = np.array([x1,x2,x3,x4]).transpose(1,0,2).reshape(-1,4,9,9)
        x1 = self.layers[12].forward(x_1234).reshape([-1,81])
        x2 = self.layers[13].forward(x_1234).reshape([-1,81])
        x3 = self.layers[14].forward(x_1234).reshape([-1,81])
        x4 = self.layers[15].forward(x_1234).reshape([-1,81])



        x = np.concatenate([x1,x2,x3,x4,x],axis=1)

        x = self.layers[16].forward(x)
        # print(x)

        return x

    def loss(self, x, t):
        # if np.isnan(x):
        #     raise AssertionError

        # x = np.asarray(x)
        # t = np.asarray(t)
        y = self.predict(x)
        if np.isnan(y.any()):
            raise AssertionError

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # t = np.asarray(t)
        y = self.predict(x)
        y = sigmoid(y)[:,0]
        y = y>0.5
        # print(y.shape)
        # print(t.shape)
        # print(np.where(y[0]==t))
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1:t = np.argmax(t, axis=1)
        accuracy = (y==t).sum() / float(len(t))
        return accuracy

    def gradient(self, x, t):
        # print(self.loss(x, t))
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        # print(dout)
        dout = self.layers[16].backward(dout)
        # print(dout.shape)
        dout9 = self.layers[15].backward(dout[:,243:324].reshape([1,-1,9,9]))
        dout8 = self.layers[14].backward(dout[:,162:243].reshape([1,-1,9,9]))
        dout7 = self.layers[13].backward(dout[:,81:162].reshape([1,-1,9,9]))
        dout6 = self.layers[12].backward(dout[:,0:81].reshape([1,-1,9,9]))

        dout6789 = np.sum(np.array([dout6,dout7,dout8,dout9]),axis=0)/5.

        dout = self.layers[11].backward(dout[:,324:374])
        dout = self.layers[10].backward(dout)/5
       
        dout9 = self.layers[9].backward(dout[:,243:324].reshape([1,-1,9,9]) + dout6789[:,3,:,:])
        dout8 = self.layers[8].backward(dout[:,162:243].reshape([1,-1,9,9]) + dout6789[:,2,:,:])
        dout7 = self.layers[7].backward(dout[:,81:162].reshape([1,-1,9,9]) + dout6789[:,1,:,:])
        dout6 = self.layers[6].backward(dout[:,0:81].reshape([1,-1,9,9]) + dout6789[:,0,:,:])

        dout = self.layers[5].backward(dout[:,324:424])
        dout = self.layers[4].backward(dout)/5

        dout6789 = np.sum(np.array([dout6,dout7,dout8,dout9]),axis=0)/5.
        
        self.layers[3].backward(dout[:,243:324].reshape([1,-1,9,9]) + dout6789[:,3,:,:])
        self.layers[2].backward(dout[:,162:243].reshape([1,-1,9,9]) + dout6789[:,2,:,:])
        self.layers[1].backward(dout[:,81:162].reshape([1,-1,9,9]) + dout6789[:,1,:,:])
        self.layers[0].backward(dout[:,0:81].reshape([1,-1,9,9]) + dout6789[:,0,:,:])
        

        grad_w = []
        grad_b = []
        for layer in self.layers:
            # print(len(layer.dW))
            
            grad_w.append(layer.dW)
            grad_b.append(layer.db)


        return grad_w,grad_b





class ShogiLayerNet3(object):

    def __init__(self, weight_init_std=0.1):
        self.device = "gpu"
        self.W_params = []
        self.b_params = []
        self.layers = []
        
        self.filter_positions_1 = np.array([[8,8,8,8,8,8,8,8,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,8,8,8,8,8,8,8,8],
                                            [0,1,2,3,4,5,6,7,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,9,10,11,12,13,14,15,16]])
        self.filter_positions_2 = np.array([[0,16,1,15,2,14,3,13,4,12,5,11,6,10,7,9,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15,0,16],
                                            [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16,16]])
        self.filter_positions_3 = np.array([[6,6,8,10,10],
                                            [7,9,8,7,9]])        

#0 hisha 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_1[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))



#1 kaku 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_2[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#2 keima 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_3[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#3 sonota 1
        w = weight_init_std * np.random.randn(2*2*14,3,3)
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#4 Last 1
        w = weight_init_std * np.random.randn(81+81+81+81+14*2*2, 100)/(1848+1848+560+504)**0.5
        b = weight_init_std * np.random.randn(100)/(1848+1848+560+504)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))
#5 Rule 1
        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Sigmoid())

#6 hisha 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_1[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

#7 kaku 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#8 keima 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#9 sonota 2
        w = weight_init_std * np.random.randn(4,3,3)/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#10 Last 2
        w = weight_init_std * np.random.randn(81+81+81+81+100, 50)/(4*33*2 + 4*10 + 4*9 + 4*14)**0.5
        b = weight_init_std * np.random.randn(50)/(4*33*2 + 4*10 + 4*9 + 4*14)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))

#11 Rule 2
        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)

        self.layers.append(Sigmoid())

#12 hisha 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_1[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_1, stride=1, pad=8))

#13 kaku 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_2, stride=1, pad=8))

#14 keima 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha( w, b, self.filter_positions_3, stride=1, pad=8))

#15 sonota 3
        w = weight_init_std * np.random.randn(4,3,3)/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution( w, b, stride=1, pad=1))

#16 Last 3
        w = weight_init_std * np.random.randn(81+81+81+81+50, 1)/(324 + 100)**0.5
        b = weight_init_std * np.random.randn(1)/(324 + 100)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w,b))




        # self.lastLayer = SoftmaxWithLoss()
        self.lastLayer = SigmoidWithLoss()

    def predict(self, x):
        x_a = x[0].reshape([-1,2*2*14,9,9])
        x_b = x[1].reshape([-1,56])
        x1 = self.layers[0].forward(x_a).reshape([-1,81])
        x2 = self.layers[1].forward(x_a).reshape([-1,81])
        x3 = self.layers[2].forward(x_a).reshape([-1,81])
        x4 = self.layers[3].forward(x_a).reshape([-1,81])
        x = np.concatenate([x1,x2,x3,x4,x_b],axis=1)
        x = self.layers[4].forward(x)
        x = self.layers[5].forward(x)

        x_1234 = np.array([x1,x2,x3,x4]).transpose(1,0,2).reshape(-1,4,9,9)
        x1 = self.layers[6].forward(x_1234).reshape([-1,81])
        x2 = self.layers[7].forward(x_1234).reshape([-1,81])
        x3 = self.layers[8].forward(x_1234).reshape([-1,81])
        x4 = self.layers[9].forward(x_1234).reshape([-1,81])

        x = np.concatenate([x1,x2,x3,x4,x],axis=1)

        x = self.layers[10].forward(x)
        x = self.layers[11].forward(x)

        x_1234 = np.array([x1,x2,x3,x4]).transpose(1,0,2).reshape(-1,4,9,9)
        x1 = self.layers[12].forward(x_1234).reshape([-1,81])
        x2 = self.layers[13].forward(x_1234).reshape([-1,81])
        x3 = self.layers[14].forward(x_1234).reshape([-1,81])
        x4 = self.layers[15].forward(x_1234).reshape([-1,81])



        x = np.concatenate([x1,x2,x3,x4,x],axis=1)

        x = self.layers[16].forward(x)
        # print(x)

        return x

    def loss(self, x, t):
        # if np.isnan(x):
        #     raise AssertionError

        # x = np.asarray(x)
        # t = np.asarray(t)
        y = self.predict(x)
        if np.isnan(y.any()):
            raise AssertionError

        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        # t = np.asarray(t)
        y = self.predict(x)
        y = sigmoid(y)[:,0]
        y = y>0.5
        # print(y.shape)
        # print(t.shape)
        # print(np.where(y[0]==t))
        # y = np.argmax(y, axis=1)
        # if t.ndim != 1:t = np.argmax(t, axis=1)
        accuracy = (y==t).sum() / float(len(t))
        return accuracy

    def gradient(self, x, t):
        # print(self.loss(x, t))
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        # print(dout)
        dout = self.layers[16].backward(dout)
        # print(dout.shape)
        dout9 = self.layers[15].backward(dout[:,243:324].reshape([1,-1,9,9]))
        dout8 = self.layers[14].backward(dout[:,162:243].reshape([1,-1,9,9]))
        dout7 = self.layers[13].backward(dout[:,81:162].reshape([1,-1,9,9]))
        dout6 = self.layers[12].backward(dout[:,0:81].reshape([1,-1,9,9]))

        dout6789 = np.sum(np.array([dout6,dout7,dout8,dout9]),axis=0)/5.

        dout = self.layers[11].backward(dout[:,324:374])
        dout = self.layers[10].backward(dout)/5
       
        dout9 = self.layers[9].backward(dout[:,243:324].reshape([1,-1,9,9]) + dout6789[:,3,:,:])
        dout8 = self.layers[8].backward(dout[:,162:243].reshape([1,-1,9,9]) + dout6789[:,2,:,:])
        dout7 = self.layers[7].backward(dout[:,81:162].reshape([1,-1,9,9]) + dout6789[:,1,:,:])
        dout6 = self.layers[6].backward(dout[:,0:81].reshape([1,-1,9,9]) + dout6789[:,0,:,:])

        dout = self.layers[5].backward(dout[:,324:424])
        dout = self.layers[4].backward(dout)/5

        dout6789 = np.sum(np.array([dout6,dout7,dout8,dout9]),axis=0)/5.
        
        self.layers[3].backward(dout[:,243:324].reshape([1,-1,9,9]) + dout6789[:,3,:,:])
        self.layers[2].backward(dout[:,162:243].reshape([1,-1,9,9]) + dout6789[:,2,:,:])
        self.layers[1].backward(dout[:,81:162].reshape([1,-1,9,9]) + dout6789[:,1,:,:])
        self.layers[0].backward(dout[:,0:81].reshape([1,-1,9,9]) + dout6789[:,0,:,:])
        

        grad_w = []
        grad_b = []
        for layer in self.layers:
            # print(len(layer.dW))
            
            grad_w.append(layer.dW)
            grad_b.append(layer.db)


        return grad_w,grad_b