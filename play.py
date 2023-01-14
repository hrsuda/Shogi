import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

from shogi_2 import *
import learning
import plot_shogi as ps


def main():
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
        ps.plot_board(B.pieces)

        if teban == 0:
            move_str = None
            while (not len(move_str)==6) or (move_str == "toryo"):
                move_str = input()
