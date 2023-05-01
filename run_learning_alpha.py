import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import argparse
import os
import cupy as cp
# # try:
# #     from learning_cupy import *
# #     print("CuPy")
# except:
#     from learning import *
# from learning import *

from learning_cupy import *

import files

def main():
    # args = sys.argv
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    # data_file_name = args[1]
    # out_name = args[2]
    parser.add_argument("input_path", type=str, help="file or directry")
    parser.add_argument("output_path", type=str, help="file")

    parser.add_argument("--network", default=None,action="store", type=str, help="file")
    parser.add_argument("--learning_rate", default=0.1, action="store", type=float, help="")
    parser.add_argument("--batch_size", default=128,  action="store", type=int, help="")
    parser.add_argument("--iters_num", default=10000,  action="store", type=int, help="")
    parser.add_argument("--test_len", default=1000,  action="store", type=int, help="")



    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    iters_num = args.iters_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    test_len = args.test_len


    if os.path.isfile(input_path):
        if input_path[-6:]=="_b.npy":
            data_b = np.load(input_path, allow_pickle=True)
            data_k = np.load(input_path.replace("_b.npy","_k.npy"))
            data_t = np.load(input_path.replace("_b.npy","_t.npy"))
            
        elif input_path[-6:]=="_b.npz":
            data_b = np.load(input_path, allow_pickle=True)["arr_0"]
            data_k = np.load(input_path.replace("_b.npz","_k.npz"), allow_pickle=True)["arr_0"]
            data_t = np.load(input_path.replace("_b.npz","_t.npz"), allow_pickle=True)["arr_0"]
    elif os.path.isdir(input_path):
        filelist = files.get_file_names(input_path,file_type="npz")
        data_b = []
        data_k = []
        data_t = []
        print(filelist)
        for f in filelist:
            if f[-6:]=="_b.npy":
                data_b.append(np.load(input_path))
                data_k.append(np.load(input_path.replace("_b.npy","_k.npy")))
                data_t.append(np.load(input_path.replace("_b.npy","_t.npy")))
            elif f[-6:]=="_t.npz":
                data_b.append(np.load(input_path, allow_pickle=True)["arr_0"])
                data_k.append(np.load(input_path.replace("_b.npz","_k.npz"), allow_pickle=True)["arr_0"])
                data_t.append(np.load(input_path.replace("_b.npz","_t.npz"), allow_pickle=True)["arr_0"])
        
        data_b = np.concatenate(data_b,axis=0)
        data_k = np.concatenate(data_k,axis=0)
        data_t = np.concatenate(data_t,axis=0)

    data_b = cp.asarray(data_b)
    data_k = cp.asarray(data_k)
    data_t = cp.asarray(data_t)

    # data = data.astype(float)
    test_ind = np.random.choice(len(data_b), test_len)
    test_mask = np.zeros(len(data_b),dtype=bool)
    test_mask[test_ind] = True


    x_data_b = data_b[~test_mask]
    x_data_k = data_k[~test_mask]
    # t_data = t_bool
    t_data = data_t.astype(int)[~test_mask]
    x_data_b_good = x_data_b[t_data]
    x_data_k_good = x_data_k[t_data]
    t_data_good = t_data[t_data]
    x_data_b_bad = x_data_b[~t_data]
    x_data_k_bad = x_data_k[~t_data]
    t_data_bad = t_data[~t_data]




    x_test_b = data_b[test_mask]
    x_test_k = data_k[test_mask]

    x_test = [x_test_b, x_test_k] 
    
    # t_test = t_bool[test_mask]
    t_test = data_t[test_mask]
    # print(data[0])
    if args.network is None:
        network = ShogiLayerNet2(weight_init_std=0.3)
    else:
        with open(args.network, "rb") as f:
            network = pickle.load(f)

    data_size = x_data_b.shape[0]


    train_loss_list = []
    train_acc_list = []

    iter_per_epoch = max(data_size // batch_size, 1)
    print(iter_per_epoch)
    alpha = 1.0/4


    for i in range(iters_num):

        # batch_mask = np.random.choice(data_size, batch_size)
        batch_mask_good = np.random.choice(len(x_data_b_good), int(batch_size*alpha))
        batch_mask_bad = np.random.choice(len(x_data_b_bad), int(batch_size*(1-alpha)))

        # x_batch = x_data[batch_mask]
        # t_batch = t_data[batch_mask]
        x_batch_b = cp.concatenate([x_data_b_good[batch_mask_good], x_data_b_bad[batch_mask_bad]], axis=0)
        x_batch_k = cp.concatenate([x_data_k_good[batch_mask_good], x_data_k_bad[batch_mask_bad]], axis=0)
        x_batch = [x_batch_b, x_batch_k]
        t_batch = cp.concatenate([t_data_good[batch_mask_good], t_data_bad[batch_mask_bad]], axis=0)


        grad_w, grad_b= network.gradient(x_batch, t_batch)

        for j in range(len(grad_w)):
            network.W_params[j] -= learning_rate * grad_w[j][0]
            network.b_params[j] -= learning_rate * grad_b[j][0]



        loss = network.loss(x_batch, t_batch)
        # print(loss)


        train_loss_list.append(cp.asnumpy(loss).reshape(1))

        if i % 1000 == 0:
            train_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)

            print(train_acc)
            print(loss)
            # print(grad)

    W_params = []
    b_params = []
    for j in range(len(network.W_params)):
        W_params.append(network.W_params[j].reshape([-1]))
        b_params.append(network.b_params[j].reshape([-1]))
    W_params = cp.asnumpy(cp.concatenate(W_params))
    b_params = cp.asnumpy(cp.concatenate(b_params))
    np.savetxt(output_path + "_W.csv", W_params, delimiter=",")
    np.savetxt(output_path + "_b.csv", b_params, delimiter=",")




    print(train_loss_list)

    plt.plot(range(iters_num),np.concatenate(train_loss_list))

    plt.yscale('log')
    plt.savefig("hoge.pdf")
    # print(network.params)
    # np.save(out_name+"_W",np.concatenate(network.W_params))
    # np.save(out_name+"_b",np.concatenate(network.b_params))
    with open(output_path, 'wb') as f:
        pickle.dump(network, f)


if __name__ == "__main__":
    main()
