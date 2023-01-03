import numpy as np
import sys
from learning import *
import matplotlib.pyplot as plt


def main():
    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]

    iters_num = 10000
    batch_size = 64
    learning_rate = 1e-1
    test_len = 100


    data = np.load(data_file_name, allow_pickle=True)
    data2 = np.load(data_file_name.replace(".npy","_t.npy"), allow_pickle=True)

    # data = data.astype(float)
    test_mask = np.random.choice(len(data), test_len)

    t_bool = np.zeros([len(data2),2])
    t_bool[data2,0] = 1
    t_bool[~data2,1] = 1


    x_data = data
    # t_data = t_bool
    t_data = data2

    x_test = data[test_mask]
    # t_test = t_bool[test_mask]
    t_test = data2[test_mask]
    # print(data[0])
    network = NLayerNet(input_size=2*14*2*10*10, hidden_size_list=[200,150,100,50], output_size=1,weight_init_std=0.1)
    data_size = x_data.shape[0]


    train_loss_list = []
    train_acc_list = []

    iter_per_epoch = max(data_size // batch_size, 1)
    print(iter_per_epoch)


    for i in range(iters_num):

        batch_mask = np.random.choice(data_size, batch_size)

        x_batch = x_data[batch_mask]
        t_batch = t_data[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for j,g in enumerate(grad):
            network.W_params[j] -= learning_rate * g[0]
            network.b_params[j] -= learning_rate * g[1]



        loss = network.loss(x_batch, t_batch)


        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_data, t_data)
            train_acc_list.append(train_acc)

            print(train_acc)
            print(loss)
            # print(grad)
    print(train_acc_list)
    # print(network.params)

    plt.plot(np.arange(iters_num),train_loss_list)

    plt.yscale('log')
    plt.savefig("hoge.pdf")
    np.save(out_name,[network.W_params,network.b_params])

if __name__ == "__main__":
    main()
