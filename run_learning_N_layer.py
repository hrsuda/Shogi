import numpy as np
import sys
from learning import *
import matplotlib.pyplot as plt
import pickle
import argparse

def main():
    # args = sys.argv
    argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
    # data_file_name = args[1]
    # out_name = args[2]
    parser.add_argument("input_path", type=str, help="file or directry")
    parser.add_argument("output_path", type=str, help="file")

    parser.add_argument("--network", default=None,action="store", type=str, help="file")
    parser.add_argument("--learning_rate", default=0.1, action="store", type=floot, help="")
    parser.add_argument("--batch_size", default=128,  action="store", type=int, help="")
    parser.add_argument("--iters_num", default=1e5,  action="store", type=int, help="")
    parser.add_argument("--test_len", default=1e3,  action="store", type=int, help="")



    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    iters_num = args.iters_num
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    test_len = args.test_len

    if os.isfile(input_path):
        if input_path[-3:]=="npy":
            data = np.load(input_path, allow_pickle=True)
            data2 = np.load(input_path.replace(".npy","_t.npy"), allow_pickle=True)
        elif input_path[-3:]=="npz":
            data = np.load(input_path, allow_pickle=True)["arr_0"]
            data2 = np.load(input_path.replace(".npz","_t.npz"), allow_pickle=True)["arr_0"]
    elif os.isdir(input_path):
        filelist = files.get_file_names(input_path)
        data = []
        data2 = []
        for f in filelist:
            if input_path[-5:]=="_t.npy":
                data2.append(np.load(f, allow_pickle=True))
                data.append(np.load(input_path.replace("_t.npy",".npy"), allow_pickle=True))
            elif input_path[-5:]=="_t.npz":
                data2.append(np.load(f, allow_pickle=True))["arr_0"]
                data.append(np.load(input_path.replace("_t.npz",".npz"), allow_pickle=True))["arr_0"]
        data = np.concatenate(data,axis=0)
        data2 = np.concatenate(data2,axis=0)

    # data = data.astype(float)
    test_ind = np.random.choice(len(data), test_len)
    test_mask = np.zeros(len(data),dtype=bool)
    test_mask[test_ind] = True


    t_bool = np.zeros([len(data2),2])
    t_bool[data2,0] = 1
    t_bool[~data2,1] = 1


    x_data = data[~test_mask]
    # t_data = t_bool
    t_data = data2.astype(bool)[~test_mask]
    x_data_good = x_data[t_data]
    t_data_good = t_data[t_data]
    x_data_bad = x_data[~t_data]
    t_data_bad = t_data[~t_data]




    x_test = data[test_mask]
    # t_test = t_bool[test_mask]
    t_test = data2[test_mask]
    # print(data[0])
    if network iss None:
        network = NLayerNet(input_size=2*14*2*10*10, hidden_size_list=[200,150,100,50], output_size=1,weight_init_std=0.3)
    else:
        with open(network, "rb") as f:
            network = pickle.load(f)

    data_size = x_data.shape[0]


    train_loss_list = []
    train_acc_list = []

    iter_per_epoch = max(data_size // batch_size, 1)
    print(iter_per_epoch)
    alpha = 1.0/4


    for i in range(iters_num):

        # batch_mask = np.random.choice(data_size, batch_size)
        batch_mask_good = np.random.choice(len(x_data_good), int(batch_size*alpha))
        batch_mask_bad = np.random.choice(len(x_data_bad), int(batch_size*(1-alpha)))

        # x_batch = x_data[batch_mask]
        # t_batch = t_data[batch_mask]
        x_batch = np.concatenate([x_data_good[batch_mask_good], x_data_bad[batch_mask_bad]], axis=0)
        t_batch = np.concatenate([t_data_good[batch_mask_good], t_data_bad[batch_mask_bad]], axis=0)


        grad = network.gradient(x_batch, t_batch)

        for j,g in enumerate(grad):
            network.W_params[j] -= learning_rate * g[0]
            network.b_params[j] -= learning_rate * g[1]



        loss = network.loss(x_batch, t_batch)
        # print(loss)


        train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)

            print(train_acc)
            print(loss)
            # print(grad)
    print(train_acc_list)

    plt.plot(range(iters_num),train_loss_list)

    plt.yscale('log')
    plt.savefig("hoge.pdf")
    # print(network.params)
    # np.save(out_name+"_W",np.concatenate(network.W_params))
    # np.save(out_name+"_b",np.concatenate(network.b_params))
    with open(out_name, 'wb') as f:
        pickle.dump(network, f)


if __name__ == "__main__":
    main()
