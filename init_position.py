import pickle

from shogi_2 import *

def main():
    pieces = [OU(init_position=(5,9), owner=True),
           OU(init_position=(5,1), owner=False),
           KIN(init_position=(4,9), owner=True),
           KIN(init_position=(6,9), owner=True),
           KIN(init_position=(4,1), owner=False),
           KIN(init_position=(6,1), owner=False),
           GIN(init_position=(3,9), owner=True),
           GIN(init_position=(7,9), owner=True),
           GIN(init_position=(3,1), owner=False),
           GIN(init_position=(7,1), owner=False),
           KEIMA(init_position=(2,9), owner=True),
           KEIMA(init_position=(8,9), owner=True),
           KEIMA(init_position=(2,1), owner=False),
           KEIMA(init_position=(8,1), owner=False),
           KYO(init_position=(1,9), owner=True),
           KYO(init_position=(9,9), owner=True),
           KYO(init_position=(1,1), owner=False),
           KYO(init_position=(9,1), owner=False),
           HISHA(init_position=(2,8), owner=True),
           HISHA(init_position=(8,2), owner=False),
           KAKU(init_position=(8,8), owner=True),
           KAKU(init_position=(2,2), owner=False),
           FU(init_position=(1,7), owner=True),
           FU(init_position=(2,7), owner=True),
           FU(init_position=(3,7), owner=True),
           FU(init_position=(4,7), owner=True),
           FU(init_position=(5,7), owner=True),
           FU(init_position=(6,7), owner=True),
           FU(init_position=(7,7), owner=True),
           FU(init_position=(8,7), owner=True),
           FU(init_position=(9,7), owner=True),
           FU(init_position=(1,3), owner=False),
           FU(init_position=(2,3), owner=False),
           FU(init_position=(3,3), owner=False),
           FU(init_position=(4,3), owner=False),
           FU(init_position=(5,3), owner=False),
           FU(init_position=(6,3), owner=False),
           FU(init_position=(7,3), owner=False),
           FU(init_position=(8,3), owner=False),
           FU(init_position=(9,3), owner=False),
           ]
    with open("init_posision.pkl", "wb") as f:
        pickle.dump(pieces,f)

if __name__ == "__main__":
    main()
