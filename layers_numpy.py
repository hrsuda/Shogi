import numpy as np
import scipy as sp
from collections import OrderedDict

import cupy as cp
from functions_numpy import *
from layers_numpy import *








class Tanh(object):

    def __init__(self):
        self.mask = None
        self.dW = np.array([0])
        self.db = np.array([0])
        self.W = 0
        self.b = 0

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
        self.W = np.array([0])
        self.b = np.array([0])

    def forward(self, x):
        # x = x - np.max(x, axis=-1, keepdims=True) + 1e-12
        # self.out = out
        return x
    def backward(self, dout):
        return dout

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

        out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)  + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        return out



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


class Convolution_with_Sigmoid:
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
        self.out = 1/(1+np.exp(self.out))
        return self.out


    def backward(self, dout):
        # print(dout.shape)
        self.out = self.out.transpose(1,0,2,3)
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
        self.out = 1/(1+np.exp(self.out))
        return self.out



    def backward(self, dout):
        # batch_size = dout.shape[0]
        self.out = self.out.transpose(1,0,2,3)
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

class SoftmaxWithLoss(object):
    """docstring for SoftmaxWithLoss."""

    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        t = t*(1-2e-12) + 1e-12
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
        t = t*(1-2e-12) + 1e-12
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


class Convolution_original:
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
        self.dW = -1
        self.db = -1

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        # print(x.shape)
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col_original(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) + self.b
        self.x = x
        self.x_shape = x.shape
        self.col = col
        self.col_W = col_W
        

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im_original(dcol, self.x_shape, FH, FW, self.stride, self.pad)

        return dx

class Flatten(object):
    def __init__(self, W, b, data_num=2):
        self.W = W
        self.b = b
        self.dW = np.array([0])
        self.db = np.array([0])
        self.data_num = data_num
        if data_num == 1:
            self.forward = self._forward
            self.backward = self._backward
        else:
            self.forward = self._forward2
            self.backward = self._backward2

    def _forward(self, x):
        self.x = x
        self.batch_size = self.x.shape[0]
        self.shapes = x.shape

        return x.reshape((self.batch_size,-1))

    def _forward2(self, x):
        self.shapes = []
        xx = []
        self.batch_size = x[0].shape[0]
        for i in range(self.data_num):
            d = x[i]
            self.shapes.append(d.shape)
            xx.append(d.reshape((self.batch_size,-1)))
        x = np.hstack(xx)
        return x

    def _backward1(self, dout):

        dout = dout.reshape(self.shapes)

        return dout

    def _backward2(self, dout):
        out = []
        start = 0
        for i in range(self.data_num):
            shape = self.shapes[i]
            stop = start + np.prod(np.array(shape[1:]))
            out.append(dout[:,start:stop].reshape(shape))
            start = stop

        return out


class Reshape(object):
    def __init__(self, data_shape):
        self.W = np.array([0])
        self.b = np.array([0])
        self.dW = np.array([0])
        self.db = np.array([0])

        self.data_shape = data_shape
        self.batch_size = 1 


    def forward(self, x):

        self.batch_size = x.shape[0]
        self.original_x_shape = x.shape[1:]

        return x.reshape((self.batch_size,) + self.data_shape)

    def backward(self, dout):
        return dout.reshape((self.batch_size,) + self.original_x_shape)



class Select(object):
    """docstring for Select"""
    def __init__(self, layer, ind=0, **kwargs):
        self.layer = layer(**kwargs)
        self.ind = ind

        self.W = self.layer.W
        self.b = self.layer.b
        self.dW = self.layer.dW
        self.db = self.layer.db



    def forward(self, x):
        # print(x[self.ind].shape)
        self.layer.W = self.W
        self.layer.b = self.b
        x[self.ind] = self.layer.forward(x[self.ind]).copy()

        new_x = [xx.copy() for xx in x]


        return new_x

    def backward(self, dout):
        dout[self.ind] = self.layer.backward(dout[self.ind])
        self.dW = self.layer.dW
        self.db = self.layer.db

        return dout


class Duplication(object):
    """docstring for Select"""
    def __init__(self, ind=0, num=2):
        
        self.ind = ind
        self.num = num

        self.W = np.array([0])
        self.b = np.array([0])
        self.dW = np.array([0])
        self.db = np.array([0])



    def forward(self, x):
        # print(x[self.ind].shape)
        

        new_x = [xx.copy() for xx in x]
        for n in range(self.num):
            new_x.insert(self.ind+1, new_x[self.ind].copy())

        return new_x

    def backward(self, dout):
        out = np.sum(np.array(dout[self.ind:self.ind+self.num]), axis=0)/self.num
        
        dout_1 = []
        for i in range(len(dout)):            
            if i == self.ind:
                dout_1.append(out)
                i += self.num
            else:
                dout_1.append(dout[i])

        return dout_1


class Combine(object):
    """docstring for Select"""
    def __init__(self, ind=[0,1]):
        
        self.ind = ind

        self.W = np.array([0])
        self.b = np.array([0])
        self.dW = np.array([0])
        self.db = np.array([0])



    def forward(self, x):
        # print(x[self.ind].shape)
        

        c = [x[i].copy() for i in self.ind]
        c = np.array(c)
        c = np.concatenate(c, axis=1)
        # ax_num = np.arange(len(c.shape))
        # ax_num[[0,1]] = ax_num[[1,0]]
        # c = c.transpose(ax_num)
        new_x = [c]
        for n in range(len(x)):
            if n in self.ind:
                continue
            new_x.append(x[n])

        return new_x

    def backward(self, dout):
        c = dout[0]
        c = np.split(c,len(self.ind),axis=1)
        d = dout[1:]
        for i,j in enumerate(self.ind):            
            
            d.insert(j, c[i])

        return d