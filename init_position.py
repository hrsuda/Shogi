import pickle

from shogi_3 import *


def main():
    pieces = [OU(init_position=(5,9), owner=0),
           OU(init_position=(5,1), owner=1),
           HISHA(init_position=(2,8), owner=0),
           HISHA(init_position=(8,2), owner=1),
           KAKU(init_position=(8,8), owner=0),
           KAKU(init_position=(2,2), owner=1),
           KIN(init_position=(4,9), owner=0),
           KIN(init_position=(6,9), owner=0),
           KIN(init_position=(4,1), owner=1),
           KIN(init_position=(6,1), owner=1),
           GIN(init_position=(3,9), owner=0),
           GIN(init_position=(7,9), owner=0),
           GIN(init_position=(3,1), owner=1),
           GIN(init_position=(7,1), owner=1),
           KEIMA(init_position=(2,9), owner=0),
           KEIMA(init_position=(8,9), owner=0),
           KEIMA(init_position=(2,1), owner=1),
           KEIMA(init_position=(8,1), owner=1),
           KYO(init_position=(1,9), owner=0),
           KYO(init_position=(9,9), owner=0),
           KYO(init_position=(1,1), owner=1),
           KYO(init_position=(9,1), owner=1),
           FU(init_position=(1,7), owner=0),
           FU(init_position=(2,7), owner=0),
           FU(init_position=(3,7), owner=0),
           FU(init_position=(4,7), owner=0),
           FU(init_position=(5,7), owner=0),
           FU(init_position=(6,7), owner=0),
           FU(init_position=(7,7), owner=0),
           FU(init_position=(8,7), owner=0),
           FU(init_position=(9,7), owner=0),
           FU(init_position=(1,3), owner=1),
           FU(init_position=(2,3), owner=1),
           FU(init_position=(3,3), owner=1),
           FU(init_position=(4,3), owner=1),
           FU(init_position=(5,3), owner=1),
           FU(init_position=(6,3), owner=1),
           FU(init_position=(7,3), owner=1),
           FU(init_position=(8,3), owner=1),
           FU(init_position=(9,3), owner=1),
           ]
    with open("init_posision.pkl", "wb") as f:
        pickle.dump(pieces,f)

if __name__ == "__main__":
    main()
