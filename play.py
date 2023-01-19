import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import datetime
import os

from shogi_2 import *
import learning
import plot_shogi as ps


def main():
    plt.ion()
    args = sys.argv

    save_dir_path = "./game_log/" + datetime.datetime.now().strftime("%Y%m%d/")
    save_file_name = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)


    network_data_filename = args[1]
    with open(network_data_filename, "rb") as f:
        network = pickle.load(f)


    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    B = Board(pieces)

    sente = np.random.randint(low=0,high=2)
    teban = sente
    players = ["human", "sudanza"]
    game = True
    out = []
    out_t = []
    while game:
        print(teban)
        plt.cla()
        ax = ps.plot_board(B.pieces)
        plt.show()

        moves = B.get_legal_moves()
        move_data = []
        if B.turn==1:
            for m in moves:
                move_data.append(B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:]))

        else:
            for m in moves:
                move_data.append(B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:])[:,::-1,::-1,::-1])


        if teban == 0:
            move_str = ''
            while True:
                move_str = input().upper()
                if (move_str in moves) or (move_str in ["TORYO", "END"]):break

            if move_str in ["TORYO", "END"]:
                break

            for md in move_data:
                out.append([B.out_data,md])

            for m in moves:
                out_t.append(m==move_str)


        else:
            result = []
            for i in range(len(move_data)):
                result.append(learning.sigmoid(network.predict(np.array([B.out_data[:,::2*sente-1,::2*sente-1,::2*sente-1],move_data[i]]).reshape(1,5600))))
            ind = np.argsort(result,axis=0,)[::-1,0,0]
            move_str = moves[ind[0]]


        B.move(start=(int(move_str[0]), int(move_str[1])),goal=(int(move_str[2]),int(move_str[3])),name=move_str[4:])
        teban = 1 - teban
        B.turn = 1 - B.turn

    np.savez_compressed(save_dir_path+save_file_name, out)
    np.savez_compressed(save_dir_path+save_file_name+'_t', out_t)


if __name__ == "__main__":
    main()
