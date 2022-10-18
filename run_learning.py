import numpy as np
import sys
from learning import *
import matplotlib.pyplot as plt


def main():
    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]

    data = np.load(data_file_name, allow_pickle=True)
    # data = data.astype(float)
    x_data = data[:,:-2]
    t_data = data[:,-2:]

    print(data.shape)
    network = TwoLayerNet(input_size=160, hidden_size=100, output_size=2,weight_init_std=1000)
    data_size = x_data.shape[0]

    iters_num = 10000
    batch_size = 1000
    learning_rate = 5e-1
    train_loss_list = []
    train_acc_list = []

    iter_per_epoch = max(data_size // batch_size, 1)
    print(iter_per_epoch)


    for i in range(iters_num):

        batch_mask = np.random.choice(data_size, batch_size)

        x_batch = x_data[batch_mask]
        t_batch = t_data[batch_mask]

        grad = network.gradient(x_batch, t_batch)

        for key in ("W1", "b1", "W2", "b2"):
            network.params[key] -= learning_rate * grad[key]


        loss = network.loss(x_batch, t_batch)

        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_data, t_data)
            train_acc_list.append(train_acc)
            print(train_acc_list)
    plt.plot(np.arange(iters_num),train_loss_list)
    plt.yscale('log')
    plt.savefig("hoge.pdf")

if __name__ == "__main__":
    main()
