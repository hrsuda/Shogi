import numpy as np
import scipy as sp
from collections import OrderedDict

import cupy as cp
from functions_numpy import *
from layers_numpy import *







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
        self.layers.append(convolution_with_Sigmoid( w, b, stride=1, pad=1))



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
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_1, stride=1, pad=8))

#1 kaku 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_2[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_2, stride=1, pad=8))

#2 keima 1
        w = weight_init_std * np.random.randn(2*2*14*len(self.filter_positions_3[0]))
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_3, stride=1, pad=8))

#3 sonota 1
        w = weight_init_std * np.random.randn(2*2*14,3,3)
        b = weight_init_std * np.random.randn(9,9)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_with_Sigmoid( w, b, stride=1, pad=1))

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
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_1, stride=1, pad=8))

#7 kaku 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_2, stride=1, pad=8))

#8 keima 2
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_3, stride=1, pad=8))

#9 sonota 2
        w = weight_init_std * np.random.randn(4,3,3)/(81+81+81+81+14*2*2)**0.5
        b = weight_init_std * np.random.randn(9,9)/(81+81+81+81+14*2*2)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_with_Sigmoid( w, b, stride=1, pad=1))

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
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_1, stride=1, pad=8))

#13 kaku 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_2[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_2, stride=1, pad=8))

#14 keima 3
        w = weight_init_std * np.random.randn(4*len(self.filter_positions_3[0]))/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_alpha_with_Sigmoid( w, b, self.filter_positions_3, stride=1, pad=8))

#15 sonota 3
        w = weight_init_std * np.random.randn(4,3,3)/(4*33*2 + 4*10 + 4*9)**0.5
        b = weight_init_std * np.random.randn(9,9)/(4*33*2 + 4*10 + 4*9)**0.5
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Convolution_with_Sigmoid( w, b, stride=1, pad=1))

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
        

        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Select(layer=Reshape, data_shape=(2*2*14,9,9)))

        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Duplication(ind=0,num=1))

        w = weight_init_std * np.random.randn(1,2*2*14,3,3)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Select(layer=Convolution_original, ind=0, W=w, b=b, stride=1, pad=1))


        w = weight_init_std * np.random.randn(1,2*2*14,9,9)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Select(layer=Convolution_original,ind=1, W=w, b=b, stride=1, pad=4))

        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Combine(ind=[0,1]))

        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Duplication(ind=0,num=1))

        w = weight_init_std * np.random.randn(1,2,3,3)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Select(layer=Convolution_original, ind=0, W=w, b=b, stride=1, pad=1))

        w = weight_init_std * np.random.randn(1,2,9,9)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Select(layer=Convolution_original, ind=1, W=w, b=b, stride=1, pad=4))
        # w = weight_init_std * np.random.randn(9*9*2*2*14,9*9)
        # b = weight_init_std * np.random.randn(9*9)
        # self.W_params.append(w)
        # self.b_params.append(b)
        # self.layers.append(Select(layer=Affine, W=w, b=b))


        

        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Flatten( w, b, data_num=3))


        w = weight_init_std * np.random.randn(2*9*9+2*2*14,50)
        b = weight_init_std * np.random.randn(50)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w, b))



        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Sigmoid())



        w = weight_init_std * np.random.randn(50,25)
        b = weight_init_std * np.random.randn(25)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w, b))


        w = np.array([0.])
        b = np.array([0.])
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Sigmoid())



        w = weight_init_std * np.random.randn(25,1)
        b = weight_init_std * np.random.randn(1)
        self.W_params.append(w)
        self.b_params.append(b)
        self.layers.append(Affine(w, b))


        self.lastLayer = SigmoidWithLoss()



    def predict(self, x):
        # print(x[0].shape)

        for l in self.layers:
            # for xx in x:print(xx.shape)
            # print(l)
            x = l.forward(x)
            # print(x[1].shape)

        # x = self.lastLayer.forward(x)
        return x

    def loss(self, x, t):
        # if np.isnan(x):
        #     raise AssertionError

        # x = np.asarray(x)
        # t = np.asarray(t)
        y = self.predict(x)
        # if np.isnan(y.any()):
        #     raise AssertionError

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
        # self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)


        grad_w = []
        grad_b = []
        for layer in self.layers[::-1]:
            # print(layer)
            dout = layer.backward(dout)
            # for xx in dout:print(xx.shape)
            grad_w.append(layer.dW)
            grad_b.append(layer.db)


        return grad_w[::-1],grad_b[::-1]