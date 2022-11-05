import os
import sys
import numpy as np


moves = {"FU":[[0,1]],
        "TO":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
        "KY":[[0,i] for i in range(1,10)],
        "KE":[[1,2],[-1,2]],
        "NY":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
        "GI":[[0,1],[1,1],[-1,1],[-1,-1],[1,-1]],
        "NG":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
        "KI":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
        "OU":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1],[1,-1],[-1,-1]],
        "KA":[[i,i] for i in range(-9,10) if i != 0] + [i,-i] for i in range(-9,10) if i != 0],
        "UM":[[i,i] for i in range(-9,10) if i != 0] + [i,-i] for i in range(-9,10) if i != 0] + [[0,1], [1,0], [-1,0], [0,-1]],
        "HI":[[0,i] for i in range(-9,10) if i != 0] + [0,i] for i in range(-9,10) if i != 0],
        "RY":[[0,i] for i in range(-9,10) if i != 0] + [0,i] for i in range(-9,10) if i != 0] + [[1,1], [1,-1], [-1,1], [1,-1]],
        }


def get_file_names(dir_path, file_type="csa"):
    files = os.listdir(dir_path)
    out = [n for n in files if file_type == n[-3:]]
    return out

def get_move_files():
    names = ["FU", "TO", "KY", "NY", "KE", "NK", "GI", "NG", "KI", "HI", "RY", "KA", "UM", "OU"]



def get_move(name):
    move_dict = {}
    move_bool = []

    if name in ["FU", "TO", "NY", "KE", "NK", "GI", "NG", "KI", "OU"]:
        move = moves[name]
        t = np.zeros([1,19,19], dtype=bool)
        for p in move:
            t[0][np.array(p)+9] = True
        return t

    elif name in ["KY", "KA", "HI", ]
