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
    output2 = []

    out_0 = None
    for i,fname in enumerate(file_names):
        with open("init_posision.pkl", "rb") as f:
            pieces = pickle.load(f)
        B = Board(pieces)
        # print(fname)
        # out,good = B.read_file(input_dir+"/"+fname)
        _ = B.read_file_with_bad(input_dir+"/"+fname)
        if _ is None:continue
        out, good = _
        # print(out.shape)
        out = out
        out2 = out.copy()
        # out_g = out[good]
        # out99 = out2[:,::-1,9:0:-1,9:0:-1]
        # out2 = out2[:,:,::-1,:,:]
        out[1::2,:,:,1:,1:] = out[1::2,:,::-1,9:0:-1,9:0:-1]
        # out[1::2,:,:,1:,1:] = out[0::2,:,::-1,9:0:-1,9:0:-1]

        # l = len(out)
        # ind1 = np.arange(0,l-1,2)
        # ind2 = np.arange(1,l-1,2)
        #
        # couple = np.zeros(l-1,2,14,2,10,10)
        #
        # couple[ind1,0] = out[ind1]
        # couple[ind1,1] = out[ind1 + 1]
        # couple[ind2,0] = out2[ind2]
        # couple[ind2,1] = out2[ind2 + 1]
        # couple[0::2,0] = out[0::2:,:,::-1,9:0:-1,9:0:-1]
        #
        if out is not None:
        #     # print(i)
            output.append(out)
            output2.append(good)


        if i==1000:
            break

    # output = np.concatenate(output, axis=0)
    # output2 = np.concatenate(output2, axis=0)
    # print(np.array(output).shape)
    np.save(out_filename, output)
    np.save(out_filename+'_t', output2)


# def main():
#     init_position.main()
#     args = sys.argv
#     input_dir = args[1]
#     out_filename = args[2]
#
#     file_names = files.get_file_names(input_dir, file_type="csa")
#     output = []
#     for i,fname in enumerate(file_names):
#         with open("init_posision.pkl", "rb") as f:
#             pieces = pickle.load(f)
#         B = Board(pieces)
#         # print(fname)
#         out = B.read_file(input_dir+"/"+fname)
#
#         if out is not None:
#             # print(i)
#             output.append(out)
#         # if i==1000:
#         #     break
#
#     output = np.concatenate(output)
#     np.save(out_filename, output)

if __name__ == "__main__":
    main()
