import os
import sys
import numpy as np
import pickle


# moves = {"FU":[[0,1]],
#         "TO":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
#         "KY":[[0,i] for i in range(1,10)],
#         "KE":[[1,2],[-1,2]],
#         "NY":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
#         "GI":[[0,1],[1,1],[-1,1],[-1,-1],[1,-1]],
#         "NG":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
#         "KI":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1]],
#         "OU":[[0,1],[1,1],[-1,1],[1,0],[-1,0],[0,-1],[1,-1],[-1,-1]],
#         "KA":[[i,i] for i in range(-9,10) if i != 0] + [i,-i] for i in range(-9,10) if i != 0],
#         "UM":[[i,i] for i in range(-9,10) if i != 0] + [i,-i] for i in range(-9,10) if i != 0] + [[0,1], [1,0], [-1,0], [0,-1]],
#         "HI":[[0,i] for i in range(-9,10) if i != 0] + [i,0] for i in range(-9,10) if i != 0],
#         "RY":[[0,i] for i in range(-9,10) if i != 0] + [i,0] for i in range(-9,10) if i != 0] + [[1,1], [1,-1], [-1,1], [1,-1]],
#         }


def read_csa_file(filename):
    players = ["player1", "player2"]

    move_data = []

    with open(filename,encoding="utf-8") as f:
        data = f.read().split('\n')
    for l in data:
        if len(l)==0:
            continue

        elif l[:2]=="N+":
            players[0] = l[2:]
        elif l[:2]=="N-":
            players[1] = l[2:]

        # if (len(l)==7) and ((l[0]=="+") or (l[0]=="-")):

        elif (len(l)>6) and ((l[0]=="+") or (l[0]=="-")):
            # print(l)
            move_data.append(l[1:])


        elif l == "%TORYO":


            # out[:,-2+result] = 1
            # out[:,-4] = "human" in self.players[0]
            # out[:,-3] = "human" in self.players[1]



            return players,move_data

        elif "%" in l:
            return players,move_data




def get_file_names(dir_path, file_type="csa"):
    files = os.listdir(dir_path)
    # out = [dir_path + '/' + n for n in files if file_type == n[-3:]]
    out = []
    for f in files:
        # if file_type != f[-3:]:continue
        path = dir_path + '/' + f
        if os.path.isdir(path):
            files2 = os.listdir(path)
            print(files2)
            if len(files2) == 0:continue    
            out = out + [path + '/' + n for n in files2 if file_type == n[-3:]]
        else:
            out.append(path)


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

        a = np.zeros([17,17], dtype=bool)
        a[0,0:8,8] = True


        return a

    elif name in ["HI"]:
        move = moves[name]
        t = []


        a = np.zeros([4,17,17], dtype=bool)
        a[0,0:8,0] = True
        a[1,0,0:8] = True
        a[2,17:8,0] = True
        a[3,0,17:8] = True


        return a

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


def move_output():
    moves = {}
    moves["FU"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["TO"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["KY"] = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T

    moves["NY"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T

    moves["KE"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["NK"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["GI"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["NG"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["KI"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["OU"] = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["HI"] = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],
                            dtype=bool
                            ).T
    moves["RY"] = np.array([[[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],
                            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                            ],
                            dtype=bool
                            )
    moves["KA"] = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
                            dtype=bool
                            )

    moves["UM"] = np.array([[[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                            [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]],
                            [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
                            ],
                            dtype=bool
                            )
    with open('moves.pkl', 'wb') as f:
        pickle.dump(moves, f)
    return moves
