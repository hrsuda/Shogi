from shogi_2 import *
import numpy as np
import files
import pickle
import init_position
import sys
import plot_shogi


def main():

    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]

    out = []

    imax = 100

    data = np.load(data_file_name, allow_pickle=True)

    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    B = Board(pieces)


    for i in range(imax):

        d = data[i]
        B.board_from_data(d)

        plot_shogi.plot_board(d)
        moves = self.get_legal_moves()
        a = np.array([list(self.move_data_tmp([int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=l[4:6]).values()) for m in moves])
        # move_text = input()
        move_text = ["9998KY"]
        start = [int(mv[0]),int(mv[1])]
        goal = [int(mv[2]),int(mv[3])]
        name = l[4:6]

        a_t = a==move_text
        B.move()
        B.board_data()

        out.append([d,B.out_data])


    out = np.array(out)










if __name__ == "__main__":
    main()
