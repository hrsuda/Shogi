import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from shogi_2 import *
import learning
import plot_shogi as ps


def main():
    plt.ion()
    args = sys.argv

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

    while game:
        ax = ps.plot_board(B.pieces)
        plt.show()

        moves = B.get_legal_moves()
        move_data = []
        for m in moves:
            move_data.append(B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:]))


        if teban == 0:
            move_str = ''
            while (not (len(move_str)==6)):
                move_str = input()

        else:
            result = []
            for i in range(len(move_data)):
                result.append(learning.sigmoid(network.predict(np.array([B.out_data,move_data[i]]).reshape(1,5600))))
            ind = np.argsort(result,axis=0,)[::-1,0,0]
            move_str = moves[ind[0]]
        teban = 1 - teban

        B.move(start=(int(move_str[0]), int(move_str[1])),goal=(int(move_str[2]),int(move_str[3])),name=move_str[4:])




if __name__ == "__main__":
    main()
