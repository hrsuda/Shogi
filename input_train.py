from shogi_2 import *
import numpy as np
import files
import pickle
import init_position
import sys
import plot_shogi
import matplotlib.pyplot as plt
import random


def main():
    plt.ion()

    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]

    out = []
    out_t = []

    imax = 100

    data = np.load(data_file_name, allow_pickle=True)


    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    B = Board(pieces)


    for i0 in range(imax):

        print(i0)
        i = random.randint(0,len(data))
        # if good[i] != 1:continue
        d = data[i,0]
        d2 = data[i,1]
        print(np.where(d))


        # B.board_from_data(dict(zip(B.names,d)))

        # ax = plot_shogi.plot_board(B.pieces)
        # ax = plot_shogi.plot_board_from_data(dict(zip(B.names,d)))
        ax = plot_shogi.plot_move(d,d2)
        # plt.show()
        plt.pause(0.1)
        moves = B.get_legal_moves()
        # print(B.pieces)
        a = np.array([list(B.move_data_tmp([int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=m[4:6])) for m in moves])
        move_text =''
        # while len(move_text)==0:
        #     move_text = input()
        move_text = input()
        # move_text = "9998KY"
        if move_text=="":
            ax.clear()
            continue
        start = [int(move_text[0]),int(move_text[1])]
        goal = [int(move_text[2]),int(move_text[3])]
        name = move_text[4:6]

        a_t = (a==move_text)
        B.move(start,goal,name)
        B.board_data()
        aa = np.zeros([len(a),2,14,2,10,10])
        aa[:,0,...] = d
        aa[:,1,...] = a

        out.append(aa)
        out_t.append(a_t)
        ax.clear()


    out = np.concatenate(out,axis=0)
    out_t = np.concatenate(out_t,axis=1)
    np.save(out_filename, out)
    np.save(out_filename+'_t', out_t)













if __name__ == "__main__":
    main()
