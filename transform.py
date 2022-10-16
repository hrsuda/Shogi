from shogi_2 import *
import numpy as np
import files
import pickle
import init_position
import sys


def main():
    init_position.main()
    args = sys.argv
    input_dir = args[1]
    out_filename = args[2]

    file_names = files.get_file_names(input_dir, file_type="csa")
    output = []
    for i,fname in enumerate(file_names):
        with open("init_posision.pkl", "rb") as f:
            pieces = pickle.load(f)
            
            # print(fname)
        B = Board(pieces)
        out = B.read_file(input_dir+"/"+fname)
        print(out)
        if out is not None:
            print(i)
            output.append(out)
        if i==100:
            break

    np.savez_compressed("./test", output)

if __name__ == "__main__":
    main()
