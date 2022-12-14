from shogi_2 import *
import numpy as np
import files
import pickle
import init_position
import sys
import plot_shogi
import matplotlib.pyplot as plt


def main():

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


    for i in range(imax):
        print(data.shape)
        d = data[i,1]
        print(np.where(d))
        B.board_from_data(dict(zip(B.names,d)))

        ax = plot_shogi.plot_board(B.pieces)
        plt.show()
        moves = B.get_legal_moves()
        a = np.array([list(B.move_data_tmp([int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=m[4:6]).values()) for m in moves])
        move_text =''
        while len(move_text)!=6:
            move_text = input()
        # move_text = ["9998KY"]
        start = [int(move_text[0]),int(move_text[1])]
        goal = [int(move_text[2]),int(move_text[3])]
        name = move_text[4:6]

        a_t = a==move_text
        B.move()
        B.board_data()
        aa = np.zeros([len(a),2,14,2,10,10])
        aa[:,0,...] = d
        aa[:,1,...] = a

        out.append(aa)
        out_t.append(a_t)


    out = np.concatenate(out,axis=0)
    out_t = np.concatenate(out_t,axis=1)
    np.save(out_filename, out)
    np.save(out_filename+'_t', out_t)













if __name__ == "__main__":
    main()
