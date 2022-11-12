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
        "HI":[[0,i] for i in range(-9,10) if i != 0] + [i,0] for i in range(-9,10) if i != 0],
        "RY":[[0,i] for i in range(-9,10) if i != 0] + [i,0] for i in range(-9,10) if i != 0] + [[1,1], [1,-1], [-1,1], [1,-1]],
        }


def get_file_names(dir_path, file_type="csa"):
    files = os.listdir(dir_path)
    out = [n for n in files if file_type == n[-3:]]
    return out

def get_move_files():
    names = ["FU", "TO", "KY", "NY", "KE", "NK", "GI", "NG", "KI", "HI", "RY", "KA", "UM", "OU"]
    move_dict = {}
    for n in names:
        move_dict[n] = get_move(n)

    return move_dict


def get_move(name):

    move_bool = []

    if name in ["FU", "TO", "NY", "KE", "NK", "GI", "NG", "KI", "OU"]:
        move = moves[name]
        t = np.zeros([17,17], dtype=bool)
        for p in move:
            t[0][np.array(p)+8] = True
        return t

    elif name in ["KY"]:
        move = moves[name]
        t = []

        for i in range(1,8):
            a = np.zeros([1,17,17], dtype=bool)
            a[0,i+8:8,0] = True
            t.append(a)

        return np.concatenate(np.array(t),axis=0)

    elif name in ["HI"]:
        move = moves[name]
        t = []

        for i in range(1,8):
            a = np.zeros([4,17,17], dtype=bool)
            a[0,i+8:8,0] = True
            a[1,0,i+8:8] = True
            a[2,-i+8:8,0] = True
            a[3,0,-i+8:8] = True
            t.append(a)

        return np.concatenate(np.array(t),axis=0)

    elif name in ["RY"]:
        move = moves[name]
        t = []

        for i in range(1,8):
            a = np.zeros([4,17,17], dtype=bool)
            a[0,i+8:8,0] = True
            a[1,0,i+8:8] = True
            a[2,-i+8:8,0] = True
            a[3,0,-i+8:8] = True
            t.append(a)
        # t = np.concatenate(np.array(t),axis=0)
        aa =  [[1,1], [1,-1], [-1,1], [1,-1]]
        for ii in aa:
            a = np.zeros([1,17,17],dtype=bool)
            a[0][ii] = True
            t.append(a)
        return np.concatenate(np.array(t),axis=0)

    elif name in ["KA"]:
        move = moves[name]
        t = []

        for i in range(1,8):
            a = np.zeros([4,17,17], dtype=bool)
            for ii in range(i):
                a[0,i+8,i+8] = True
                a[1,-i+8,i+8] = True
                a[0,i+8,-i+8] = True
                a[0,-i+8,-i+8] = True
            t.append(a)
        # t = np.concatenate(np.array(t),axis=0)

        return np.concatenate(np.array(t),axis=0)


    elif name in ["UM"]:
        move = moves[name]
        t = []

        for i in range(1,8):
            a = np.zeros([4,17,17], dtype=bool)
            for ii in range(i):
                a[0,i+8,i+8] = True
                a[1,-i+8,i+8] = True
                a[0,i+8,-i+8] = True
                a[0,-i+8,-i+8] = True
            t.append(a)
        # t = np.concatenate(np.array(t),axis=0)
        aa =  [[1,0], [0,-1], [0,1], [0,-1]]
        for ii in aa:
            a = np.zeros([1,17,17],dtype=bool)
            a[0][ii] = True
            t.append(a)
        return np.concatenate(np.array(t),axis=0)
